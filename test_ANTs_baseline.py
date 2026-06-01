import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加当前目录以便引入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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

# 尝试导入 ants
try:
    import ants
except ImportError:
    print("\n[错误] 未找到 antspyx 包！\n请在终端中运行以下命令安装:\npip install antspyx\n")
    sys.exit(1)

def plot_ants_results(moving, fixed, moved_pred, moved_gt, idx, sample_tag, save_dir, metrics_text):
    m = moving 
    f = fixed
    mp = moved_pred
    mg = moved_gt
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
    # SyN 指的是 Symmetric Normalization 算法
    axes[2].set_title('3. ANTs Warped Moving\n(SyN + MI)', fontsize=14, pad=15)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    os.makedirs(save_dir, exist_ok=True)

    panel_name = f'{idx:04d}_{sample_tag}'.replace(' ', '_')
    plt.savefig(os.path.join(save_dir, f'baseline_ants_result_{panel_name}.png'), bbox_inches='tight', dpi=150)
    plt.close()

    single_dir = os.path.join(save_dir, 'single_images', panel_name)
    _save_gray_image(os.path.join(single_dir, '01_moving_real_c1.png'), m)
    _save_gray_image(os.path.join(single_dir, '02_fixed_real_distorted_c0.png'), f)
    _save_gray_image(os.path.join(single_dir, '03_ants_warped_moving.png'), mp)
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
        'ANTs Baseline Metrics Summary',
        '=' * 80,
        f"samples: {summary['samples']}",
        '-' * 80,
        '* Cross-modal Metrics (Real C1 vs Real C0):',
        f"  - Pre-Reg  -> NMI: {summary['cross_modal']['pre_nmi']:.4f}  |  Fore-Dice: {summary['cross_modal']['pre_dice']:.4f}  |  cross-ZNCC: {summary['cross_modal']['pre_cross_zncc']:.4f}",
        f"  - Post-Reg -> NMI: {summary['cross_modal']['post_nmi']:.4f}  |  Fore-Dice: {summary['cross_modal']['post_dice']:.4f}  |  cross-ZNCC: {summary['cross_modal']['post_cross_zncc']:.4f}",
        '-' * 80,
        '* Flow Quality Metrics:',
        '  - EPE (End Point Error)  : N/A (traditional ANTs baseline, no GT-flow prediction)',
        '  - Folding (Negative Jac) : N/A (not computed in this script)',
        '-' * 80,
        '* Speed Metrics:',
        f"  - Optimization Time: {summary['speed']['optimization_time_sec_per_pair']:.4f} s/pair",
        '=' * 80,
    ]

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"统计指标已保存: {json_path}")
    print(f"统计文本已保存: {txt_path}")

def main():
    # --- 1. 配置参数 ---
    # 传统方法跑 CPU 就行，不需要显卡，但为了保持加载器一致性保留这部分格式

    
    # test_name = "Test_CrossModal_test_shift"
    # test_name = "Test_CrossModal_test_id"
    test_name = "Test_CrossModal_test_ffd"
    
    test_data_dir = f'/data2/xujr/idr_data/Test/{test_name}/ch1_to_ch0'
    output_dir = f"./results/baseline_ants_results/{test_name}"

    # 不再使用 pipeline manifest 进行固定选图；使用 dataset 的 sample_count/sample_seed 或默认全量
    selected_names = None

    # 可选：快速可视化模式。开启后仅跑指定可视化数量并提前结束，不输出统计文件。
    quick_visual_only = True
    max_visualizations = 10

    sample_count = max_visualizations if quick_visual_only else None
    sample_seed = 42

    # --- 2. 准备数据 ---
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
    # Batch size 必须设为 1，因为 ANTs 是针对单张图片逐个优化的
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # --- 3. 运行 推理 ---
    print(f"开始传统 ANTs (SyN + MI) 推理，共有 {len(test_set)} 个样本...")
    
    # 评价指标 Trackers
    eval_zncc_pre = utils.AverageMeter()
    eval_zncc_post = utils.AverageMeter()
    eval_nmi_pre = utils.AverageMeter()
    eval_nmi_post = utils.AverageMeter()
    eval_dice_pre = utils.AverageMeter()
    eval_dice_post = utils.AverageMeter()
    
    # 【核心杀手锏记录项】：耗时 Trackers
    eval_time = utils.AverageMeter()
    selected_samples = []
    vis_saved_count = 0

    for i, data in enumerate(test_loader):
        if quick_visual_only and vis_saved_count >= max_visualizations:
            print(f"[INFO] quick_visual_only 已达到上限 {max_visualizations}，提前结束。")
            break

        moving = data[0]  
        fixed = data[1]   

        # 确保通道数为 1
        if moving.size(1) == 3: moving = moving.mean(dim=1, keepdim=True)
        if fixed.size(1) == 3: fixed = fixed.mean(dim=1, keepdim=True)

        # ANTs 处理的是单纯的 Numpy 矩阵（二维结构），剥离 Batch 和 Channel 维度
        m_np = moving.detach().cpu().numpy()[0, 0, :, :].astype(np.float32)
        f_np = fixed.detach().cpu().numpy()[0, 0, :, :].astype(np.float32)

        # 转换为 ANTs 对象
        moving_ants = ants.from_numpy(m_np)
        fixed_ants = ants.from_numpy(f_np)

        if i % 10 == 0:
            print(f"正在处理第 {i} 个样本，...")
            
        start_time = time.time()
        
        # 执行非刚性配准 SyN，同时强制设置使用互信息(mattes)应对跨模态
        reg_result = ants.registration(
            fixed=fixed_ants, 
            moving=moving_ants, 
            type_of_transform='SyN',
            syn_metric='mattes'
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        eval_time.update(elapsed_time)

        # 获取配准后的图像矩阵
        mp_np = reg_result['warpedmovout'].numpy()
        moved_gt_np = None

        if len(data) >= 3:
            gt_flow = data[2]
            if gt_flow.ndim == 4 and gt_flow.shape[1] != 2 and gt_flow.shape[-1] == 2:
                gt_flow = gt_flow.permute(0, 3, 1, 2).contiguous()
            gt_flow = gt_flow.float()
            if gt_flow.shape[2:] != moving.shape[2:]:
                gt_flow = resize_flow_to_target(gt_flow, moving.shape[2:])
            moved_gt_t = warp_with_flow_2d(moving.float(), gt_flow)
            moved_gt_np = moved_gt_t.detach().cpu().numpy()[0, 0, :, :].astype(np.float32)

        # =====================================================
        # 计算各种指标
        # =====================================================
        zncc_pre = utils.compute_zncc(m_np, f_np)
        zncc_post = utils.compute_zncc(mp_np, f_np)
        nmi_pre = utils.compute_nmi(m_np, f_np)
        nmi_post = utils.compute_nmi(mp_np, f_np)
        dice_pre = utils.compute_foreground_dice(m_np, f_np)
        dice_post = utils.compute_foreground_dice(mp_np, f_np)
        
        eval_zncc_pre.update(zncc_pre)
        eval_zncc_post.update(zncc_post)
        eval_nmi_pre.update(nmi_pre)
        eval_nmi_post.update(nmi_post)
        eval_dice_pre.update(dice_pre)
        eval_dice_post.update(dice_post)
        
        if vis_saved_count < max_visualizations:
            sample_name = os.path.basename(test_set.pairs[i]['moving']) if i < len(test_set.pairs) else f'sample_{i:04d}'
            sample_tag = os.path.splitext(sample_name)[0]
            metrics_text = (
                f"--- ANTs/SyN Baseline Metrics ---\n"
                f"Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}/ZNCC={zncc_pre:.4f}\n"
                f"Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}/ZNCC={zncc_post:.4f}\n"
                f"(Optimization Time: {elapsed_time:.3f} s/pair)"
            )
            plot_ants_results(m_np, f_np, mp_np, moved_gt_np, i, sample_tag, output_dir, metrics_text)
            selected_samples.append((i, sample_name))
            vis_saved_count += 1
            
    print(f"\nANTs 推理及评价完成。测试图保存在 {output_dir}\n")
    if quick_visual_only:
        print(f"[INFO] quick_visual_only=True，已跳过全量统计与 summary 文件输出。")
    else:
        print("="*80)
        print("[Method: Traditional ANTs/SyN (MI) Baseline]")
        print("-" * 80)
        print("* Cross-modal Metrics (Real C1 vs Real C0):")
        print(f"  - Pre-Reg  -> NMI: {eval_nmi_pre.avg:.4f}  |  Fore-Dice: {eval_dice_pre.avg:.4f}  |  cross-ZNCC: {eval_zncc_pre.avg:.4f}")
        print(f"  - Post-Reg -> NMI: {eval_nmi_post.avg:.4f}  |  Fore-Dice: {eval_dice_post.avg:.4f}  |  cross-ZNCC: {eval_zncc_post.avg:.4f}")
        print("-" * 80)
        print("* Speed Metrics:")
        print(f"  - Optimization Time: {eval_time.avg:.4f} s/pair")
        print("="*80 + "\n")

        summary = {
            'samples': len(test_set),
            'method': 'ants_syn_mi_baseline',
            'cross_modal': {
                'pre_nmi': float(eval_nmi_pre.avg),
                'pre_dice': float(eval_dice_pre.avg),
                'pre_cross_zncc': float(eval_zncc_pre.avg),
                'post_nmi': float(eval_nmi_post.avg),
                'post_dice': float(eval_dice_post.avg),
                'post_cross_zncc': float(eval_zncc_post.avg),
            },
            'flow_quality': {
                'epe': None,
                'folding_ratio': None,
            },
            'speed': {
                'optimization_time_sec_per_pair': float(eval_time.avg),
            }
        }
        save_metrics_summary(output_dir, summary)
    save_selected_samples_manifest(output_dir, selected_samples)

if __name__ == '__main__':
    main()
