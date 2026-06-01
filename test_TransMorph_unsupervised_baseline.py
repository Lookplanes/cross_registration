import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加当前目录以便引入 TransMorph 相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 引入 TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import utils
from data.datasets import SingleModalityPairedDataset


def load_selected_names(manifest_json_path):
    """Load selected sample names from pipeline manifest JSON."""
    if manifest_json_path is None:
        return None
    if not os.path.isfile(manifest_json_path):
        print(f"[WARN] 未找到样本清单，忽略固定选图: {manifest_json_path}")
        return None

    with open(manifest_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names = []
    for item in data:
        if isinstance(item, dict) and 'name' in item:
            names.append(item['name'])
    names = list(dict.fromkeys(names))

    print(f"[INFO] 已加载固定样本数: {len(names)} from {manifest_json_path}")
    return names if len(names) > 0 else None


def save_selected_samples_manifest(output_dir, selected_samples):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'selected_samples.txt')
    json_path = os.path.join(output_dir, 'selected_samples.json')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for idx, name in selected_samples:
            f.write(f"{idx}\t{name}\n")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{'index': idx, 'name': name} for idx, name in selected_samples], f, indent=2, ensure_ascii=False)
    print(f"可视化样本清单已保存: {txt_path}")
    print(f"可视化样本清单已保存: {json_path}")


def _save_gray_image(path, image_2d, cmap='gray', vmin=0.0, vmax=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image_2d, cmap=cmap, vmin=vmin, vmax=vmax)


def resize_flow_to_target(flow, target_hw):
    """Resize flow to target spatial size and scale displacement magnitudes."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    src_h, src_w = int(flow.shape[2]), int(flow.shape[3])

    if src_h == target_h and src_w == target_w:
        return flow

    flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    scale_y = float(target_h) / float(max(src_h, 1))
    scale_x = float(target_w) / float(max(src_w, 1))
    flow_resized[:, 0, :, :] = flow_resized[:, 0, :, :] * scale_y
    flow_resized[:, 1, :, :] = flow_resized[:, 1, :, :] * scale_x
    return flow_resized


def warp_with_flow_2d(image, flow):
    """Backward warp image with dense flow. image: [B,1,H,W], flow: [B,2,H,W]."""
    b, _, h, w = image.shape
    device = image.device
    dtype = image.dtype

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    base_grid = torch.stack((yy, xx), dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
    new_locs = base_grid + flow

    new_locs_y = 2.0 * (new_locs[:, 0, :, :] / max(h - 1, 1) - 0.5)
    new_locs_x = 2.0 * (new_locs[:, 1, :, :] / max(w - 1, 1) - 0.5)
    grid = torch.stack((new_locs_x, new_locs_y), dim=-1)
    return F.grid_sample(image, grid, align_corners=True, mode='bilinear')

def load_transmorph_model(model_path, img_size=(256, 256), device='cuda'):
    """
    加载预训练的 TransMorph 配准模型
    """
    config = CONFIGS_TM['TransMorph']
    config.img_size = img_size
    config.in_chans = 2  # Fixed + Moving
    model = TransMorph.TransMorph(config)
    
    if not os.path.exists(model_path):
        print(f"警告: 找不到 TransMorph 模型文件 {model_path}，使用随机初始化。")
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 处理可能的 DataParallel module 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print(f"成功加载无监督 TransMorph 模型: {model_path}")
        
    model.to(device)
    model.eval()
    return model

def plot_baseline_results(moving, fixed, moved_pred, moved_gt, idx, sample_tag, save_dir, metrics_text):
    """
    保存无监督 Baseline 可视化结果（仅3个面板，因为没有 Pix2Pix）
    """
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()[0, 0, :, :]

    m = to_numpy(moving) 
    f = to_numpy(fixed)
    mp = to_numpy(moved_pred)
    mg = to_numpy(moved_gt) if moved_gt is not None else None
    diff_pre = np.abs(m - f)
    diff_post = np.abs(mp - f)
    diff_gt_vs_pred = np.abs(mg - mp) if mg is not None else np.zeros_like(m)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    plt.suptitle(metrics_text, fontsize=13, fontweight='bold', y=0.98)

    axes[0].imshow(m, cmap='gray')
    axes[0].set_title('1. Moving\n(Real C1)', fontsize=14, pad=15)
    axes[0].axis('off')

    axes[1].imshow(f, cmap='gray')
    axes[1].set_title('2. Fixed\n(Real Distorted C0)', fontsize=14, pad=15)
    axes[1].axis('off')

    axes[2].imshow(mp, cmap='gray')
    axes[2].set_title('3. Warped Moving\n(Real C1 + Flow)', fontsize=14, pad=15)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    os.makedirs(save_dir, exist_ok=True)

    panel_name = f'{idx:04d}_{sample_tag}'.replace(' ', '_')
    plt.savefig(os.path.join(save_dir, f'baseline_unsup_result_{panel_name}.png'), bbox_inches='tight', dpi=150)
    plt.close()

    single_dir = os.path.join(save_dir, 'single_images', panel_name)
    _save_gray_image(os.path.join(single_dir, '01_moving_real_c1.png'), m)
    _save_gray_image(os.path.join(single_dir, '02_fixed_real_distorted_c0.png'), f)
    _save_gray_image(os.path.join(single_dir, '03_warped_moving_real_c1.png'), mp)
    _save_gray_image(os.path.join(single_dir, '04_pre_reg_diff.png'), diff_pre, cmap='hot', vmin=0.0, vmax=1.0)
    _save_gray_image(os.path.join(single_dir, '05_post_reg_diff.png'), diff_post, cmap='hot', vmin=0.0, vmax=1.0)
    if mg is not None:
        _save_gray_image(os.path.join(single_dir, '06_moving_plus_gt_flow.png'), mg)
    else:
        _save_gray_image(os.path.join(single_dir, '06_moving_plus_gt_flow_na.png'), np.zeros_like(m))
    _save_gray_image(os.path.join(single_dir, '07_gt_vs_pred_diff.png'), diff_gt_vs_pred, cmap='hot', vmin=0.0, vmax=1.0)
    with open(os.path.join(single_dir, 'metrics.txt'), 'w', encoding='utf-8') as f_txt:
        f_txt.write(metrics_text + '\n')


def save_metrics_summary(output_dir, summary):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'metrics_summary.json')
    txt_path = os.path.join(output_dir, 'metrics_summary.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        'Unsupervised Baseline Metrics Summary',
        '=' * 80,
        f"samples: {summary['samples']}",
        '-' * 80,
        '* Cross-modal Metrics (Real C1 vs Real C0):',
        f"  - Pre-Reg  -> NMI: {summary['cross_modal']['pre_nmi']:.4f}  |  Fore-Dice: {summary['cross_modal']['pre_dice']:.4f}  |  cross-ZNCC: {summary['cross_modal']['pre_cross_zncc']:.4f}",
        f"  - Post-Reg -> NMI: {summary['cross_modal']['post_nmi']:.4f}  |  Fore-Dice: {summary['cross_modal']['post_dice']:.4f}  |  cross-ZNCC: {summary['cross_modal']['post_cross_zncc']:.4f}",
        '-' * 80,
        '* Flow Quality Metrics:',
    ]

    if summary['flow_quality']['epe'] is not None:
        lines.append(f"  - EPE (End Point Error)  : {summary['flow_quality']['epe']:.4f}")
    else:
        lines.append('  - EPE (End Point Error)  : N/A (GT flow unavailable)')

    if summary['flow_quality']['folding_ratio'] is not None:
        lines.append(f"  - Folding (Negative Jac) : {summary['flow_quality']['folding_ratio']:.4%}")
    else:
        lines.append('  - Folding (Negative Jac) : N/A')
    lines.append('=' * 80)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"统计指标已保存: {json_path}")
    print(f"统计文本已保存: {txt_path}")

def main():
    # --- 1. 配置参数 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # test_name = "Test_CrossModal_test_shift"
    test_name = "Test_CrossModal_test_id"
    # test_name = "Test_CrossModal_test_ffd"

    # 路径配置（请根据实际的无监督模型路径修改）
    transmorph_model_path = "/data2/xujr/output_model/0330/TransMorph_unsupervised_ncc_1.0_1.0/experiments/model_best.pth"
    # test_data_dir = '/data2/xujr/idr_data/Train_CrossModal/Test/ch1_to_ch0'
    test_data_dir = f'/data2/xujr/idr_data/Test/{test_name}/ch1_to_ch0'

    # output_dir = "./results/baseline_unsupervised_results"
    output_dir = f"./results/baseline_unsupervised_results/{test_name}"

    selected_names = None


    # 可选：快速可视化模式。开启后仅跑指定可视化数量并提前结束，不输出统计文件。
    quick_visual_only = True
    max_visualizations = 10

    sample_count = max_visualizations if quick_visual_only else None
    sample_seed = 42

    # --- 2. 加载模型 ---
    print("正在加载 TransMorph 模型...")
    transmorph = load_transmorph_model(transmorph_model_path, img_size=(256, 256), device=device)

    # --- 3. 准备数据 ---
    test_set = SingleModalityPairedDataset(
        target_dir=test_data_dir,
        moving_folder='moving',  # 原始无形变的图(C1)
        fixed_folder='fixed',   # 扰动后的变形图(C0)
        img_size=(256, 256),
        transforms=None,
        selected_names=selected_names,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # --- 4. 运行 推理 ---
    print(f"开始 Baseline 推理，共有 {len(test_set)} 个样本...")
    
    # Trackers
    eval_zncc_pre = utils.AverageMeter()
    eval_zncc_post = utils.AverageMeter()
    eval_nmi_pre = utils.AverageMeter()
    eval_nmi_post = utils.AverageMeter()
    eval_epe = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_dice_pre = utils.AverageMeter()
    eval_dice_post = utils.AverageMeter()
    selected_samples = []
    vis_saved_count = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if quick_visual_only and vis_saved_count >= max_visualizations:
                print(f"[INFO] quick_visual_only 已达到上限 {max_visualizations}，提前结束。")
                break

            data = [t.to(device) for t in data]
            moving = data[0]  
            fixed = data[1]   

            # 确保通道数为 1
            if moving.size(1) == 3: moving = moving.mean(dim=1, keepdim=True)
            if fixed.size(1) == 3: fixed = fixed.mean(dim=1, keepdim=True)

            # ===== 核心：直接抛弃 GAN，将两个不同模态直接送入无监督模型 =====
            x_in = torch.cat((moving, fixed), dim=1)
            output = transmorph(x_in)
            
            moved_pred_real = output[0]  # 无监督下直接输出的 warped moving
            flow_pred = output[1]        
            moved_gt_real = None

            # --- 中间评价指标：计算 EPE 和 雅可比折叠比例 (Folding Ratio) ---
            if len(data) >= 4:
                valid_mask = data[3].to(device)
            else:
                valid_mask = (fixed > 1e-4).float()
                
            if len(data) >= 3:
                gt_flow = data[2].to(device)
                if gt_flow.ndim == 4 and gt_flow.shape[1] != 2 and gt_flow.shape[-1] == 2:
                    gt_flow = gt_flow.permute(0, 3, 1, 2).contiguous()
                if gt_flow.shape[2:] != moving.shape[2:]:
                    gt_flow = resize_flow_to_target(gt_flow, moving.shape[2:])

                moved_gt_real = warp_with_flow_2d(moving, gt_flow)
                diff = flow_pred - gt_flow
                epe = torch.norm(diff, p=2, dim=1, keepdim=True)
                masked_epe = epe * valid_mask
                mean_epe = masked_epe.sum() / (valid_mask.sum() + 1e-8)
                eval_epe.update(mean_epe.item(), moving.size(0))
            
            flow_np = flow_pred.detach().cpu().numpy()[0].transpose(1, 2, 0)
            jac_det = utils.jacobian_determinant_vxm(flow_np)
            mask_np = valid_mask.detach().cpu().numpy()[0, 0]
            valid_pixels = np.sum(mask_np > 0)
            folding_ratio = np.sum((jac_det <= 0) & (mask_np > 0)) / (valid_pixels + 1e-8)
            eval_det.update(folding_ratio, moving.size(0))

            if i % 10 == 0:
                print(f"正在处理第 {i} 个样本...")
            
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy()[0, 0, :, :]
            
            m = to_numpy(moving) 
            f = to_numpy(fixed)
            mp = to_numpy(moved_pred_real)
            
            # 这里记录原始跨模态间的ZNCC供参考(尽管是无效指标，但用于横向对比)
            zncc_pre = utils.compute_zncc(m, f)
            zncc_post = utils.compute_zncc(mp, f)
            nmi_pre = utils.compute_nmi(m, f)
            nmi_post = utils.compute_nmi(mp, f)
            dice_pre = utils.compute_foreground_dice(m, f)
            dice_post = utils.compute_foreground_dice(mp, f)
            
            eval_zncc_pre.update(zncc_pre, moving.size(0))
            eval_zncc_post.update(zncc_post, moving.size(0))
            eval_nmi_pre.update(nmi_pre, moving.size(0))
            eval_nmi_post.update(nmi_post, moving.size(0))
            eval_dice_pre.update(dice_pre, moving.size(0))
            eval_dice_post.update(dice_post, moving.size(0))
            
            if vis_saved_count < max_visualizations:
                sample_name = os.path.basename(test_set.pairs[i]['moving']) if i < len(test_set.pairs) else f'sample_{i:04d}'
                sample_tag = os.path.splitext(sample_name)[0]
                metrics_text = (
                    f"--- Unsupervised Baseline Metrics ---\n"
                    f"Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}/Cross-ZNCC={zncc_pre:.4f}\n"
                    f"Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}/Cross-ZNCC={zncc_post:.4f}"
                )
                plot_baseline_results(moving, fixed, moved_pred_real, moved_gt_real, i, sample_tag, output_dir, metrics_text)
                selected_samples.append((i, sample_name))
                vis_saved_count += 1
                
    print(f"\nBaseline 推理完成。测试图保存在 {output_dir}\n")
    if quick_visual_only:
        print(f"[INFO] quick_visual_only=True，已跳过全量统计与 summary 文件输出。")
    else:
        print("="*80)
        print("[Method: Unsupervised TransMorph Baseline]")
        print("-" * 80)
        print("* Cross-modal Metrics (Real C1 vs Real C0):")
        print(f"  - Pre-Reg  -> NMI: {eval_nmi_pre.avg:.4f}  |  Fore-Dice: {eval_dice_pre.avg:.4f}  |  cross-ZNCC: {eval_zncc_pre.avg:.4f}")
        print(f"  - Post-Reg -> NMI: {eval_nmi_post.avg:.4f}  |  Fore-Dice: {eval_dice_post.avg:.4f}  |  cross-ZNCC: {eval_zncc_post.avg:.4f}")
        print("-" * 80)
        print("* Flow Quality Metrics:")
        if eval_epe.count > 0:
            print(f"  - EPE (End Point Error)  : {eval_epe.avg:.4f}")
        print(f"  - Folding (Negative Jac) : {eval_det.avg:.4%}")
        print("="*80 + "\n")

        summary = {
            'samples': len(test_set),
            'method': 'unsupervised_transmorph_baseline',
            'cross_modal': {
                'pre_nmi': float(eval_nmi_pre.avg),
                'pre_dice': float(eval_dice_pre.avg),
                'pre_cross_zncc': float(eval_zncc_pre.avg),
                'post_nmi': float(eval_nmi_post.avg),
                'post_dice': float(eval_dice_post.avg),
                'post_cross_zncc': float(eval_zncc_post.avg),
            },
            'flow_quality': {
                'epe': float(eval_epe.avg) if eval_epe.count > 0 else None,
                'folding_ratio': float(eval_det.avg),
            }
        }
        save_metrics_summary(output_dir, summary)
    save_selected_samples_manifest(output_dir, selected_samples)

if __name__ == '__main__':
    main()
