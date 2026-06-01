import os
import sys
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加当前目录以便引入 TransMorph 相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 将 pix2pix 的路径加入环境变量以便导入
pix2pix_dir = os.path.join(current_dir, 'pytorch-CycleGAN-and-pix2pix')
sys.path.append(pix2pix_dir)

# 引入 TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import utils
from data.datasets import SingleModalityPairedDataset

# 引入 pix2pix 的 Generator
import importlib.util
networks_path = os.path.join(pix2pix_dir, 'models', 'networks.py')
spec = importlib.util.spec_from_file_location("pix2pix_networks", networks_path)
pix2pix_networks = importlib.util.module_from_spec(spec)
sys.modules["pix2pix_networks"] = pix2pix_networks
spec.loader.exec_module(pix2pix_networks)
define_G = pix2pix_networks.define_G

def load_pix2pix_generator(model_path, input_nc=1, output_nc=1, ngf=64, netG='unet_256', norm='batch', use_dropout=True, device='cuda'):
    """
    加载预训练的 pix2pix 生成器
    """
    # 初始化生成器
    netG_model = define_G(input_nc=input_nc, output_nc=output_nc, ngf=ngf, netG=netG, 
                          norm=norm, use_dropout=use_dropout, init_type='normal', init_gain=0.02)
    
    # 加载权重
    if not os.path.exists(model_path):
        print(f"警告: 找不到 pix2pix 模型文件 {model_path}，当前使用随机初始化权重。")
    else:
        state_dict = torch.load(model_path, map_location=str(device))
        # 兼容不同的权重保存格式
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        netG_model.load_state_dict(state_dict)
        print(f"成功加载 pix2pix 模型: {model_path}")
        
    netG_model.to(device)
    netG_model.eval()
    return netG_model

def load_transmorph_model(model_path, img_size=(256, 256), device='cuda'):
    """
    加载预训练的 TransMorph 配准模型
    """
    config = CONFIGS_TM['TransMorph']
    config.img_size = img_size
    config.in_chans = 2  # Fixed + Moving
    config.if_diffeomorphic = True
    print(f"Diffeomorphic integration enabled for pipeline: {config.if_diffeomorphic}")
    model = TransMorph.TransMorph(config)
    
    if not os.path.exists(model_path):
        print(f"警告: 找不到 TransMorph 模型文件 {model_path}，当前使用随机初始化权重。")
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 处理可能的 DataParallel module 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"成功加载 TransMorph 模型: {model_path}")
        
    model.to(device)
    model.eval()
    return model


def resize_flow_to_target(flow, target_hw):
    """将光流重采样到目标尺寸，并按缩放比例修正位移幅值。"""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    src_h, src_w = int(flow.shape[2]), int(flow.shape[3])

    if src_h == target_h and src_w == target_w:
        return flow

    flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)

    # flow[0] 是 y 位移，flow[1] 是 x 位移。
    scale_y = float(target_h) / float(max(src_h, 1))
    scale_x = float(target_w) / float(max(src_w, 1))
    flow_resized[:, 0, :, :] = flow_resized[:, 0, :, :] * scale_y
    flow_resized[:, 1, :, :] = flow_resized[:, 1, :, :] * scale_x
    return flow_resized

def _save_gray_image(path, image_2d, cmap='gray', vmin=0.0, vmax=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image_2d, cmap=cmap, vmin=vmin, vmax=vmax)


def _save_deformed_grid(path, flow, step=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = flow.shape[1], flow.shape[2]
    grid_x, grid_y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    map_x = grid_x + flow[1]
    map_y = grid_y + flow[0]

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(np.ones((h, w)), cmap='gray', vmin=0, vmax=1)
    for i in range(0, h, step):
        ax.plot(map_x[i, :], map_y[i, :], color='blue', linewidth=0.5)
    for j in range(0, w, step):
        ax.plot(map_x[:, j], map_y[:, j], color='blue', linewidth=0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_pipeline_results(moving, fixed, fake_moving, moved_pred_fake, moved_pred_real, flow_pred, moved_gt_real, idx, sample_tag, save_dir, metrics_text):
    """
    保存可视化结果，并计算评价指标排版展示
    """
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()[0, 0, :, :]

    m = to_numpy(moving) 
    f = to_numpy(fixed)
    fm = to_numpy(fake_moving)
    mp_fake = to_numpy(moved_pred_fake)
    mp_real = to_numpy(moved_pred_real)
    flow = flow_pred.detach().cpu().numpy()[0] # (2, H, W)
    mp_gt = to_numpy(moved_gt_real) if moved_gt_real is not None else None
    
    # 计算配准前后同一模态 (C0) 之间的差异
    diff_pre = np.abs(f - fm)
    diff_post = np.abs(f - mp_fake)
    diff_gt_vs_moving = np.abs(mp_gt - m) if mp_gt is not None else np.zeros_like(m)
    diff_gt_vs_pred = np.abs(mp_gt - mp_real) if mp_gt is not None else np.zeros_like(m)

    # Draw Deformed Grid
    step = 10
    h, w = flow.shape[1], flow.shape[2]
    grid_x, grid_y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    map_x = grid_x + flow[1] # x displacement (w)
    map_y = grid_y + flow[0] # y displacement (h)
    
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flatten()
    
    plt.suptitle(metrics_text, fontsize=15, fontweight='bold', y=0.98)

    axes[0].imshow(m, cmap='gray')
    axes[0].set_title('1. Moving\n(Real C1)', fontsize=14, pad=15)
    axes[0].axis('off')

    axes[1].imshow(fm, cmap='gray')
    axes[1].set_title('2. Fake Moving\n(Masked Fake C0)', fontsize=14, pad=15)
    axes[1].axis('off')

    axes[2].imshow(f, cmap='gray')
    axes[2].set_title('3. Fixed\n(Masked Real Distorted C0)', fontsize=14, pad=15)
    axes[2].axis('off')

    if mp_gt is not None:
        axes[3].imshow(mp_gt, cmap='gray')
        axes[3].set_title('4. Moving + GT Flow\n(Real C1 + GT Flow)', fontsize=14, pad=15)
        axes[3].axis('off')
    else:
        axes[3].imshow(np.zeros_like(m), cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('4. Moving + GT Flow\n(Not Available)', fontsize=14, pad=15)
        axes[3].axis('off')

    axes[4].imshow(mp_fake, cmap='gray')
    axes[4].set_title('5. Warped Fake Moving\n(Masked Fake C0 + Flow)', fontsize=14, pad=15)
    axes[4].axis('off')
    
    axes[5].imshow(mp_real, cmap='gray')
    axes[5].set_title('6. Warped Real Moving\n(Real C1 + Pred Flow)', fontsize=14, pad=15)
    axes[5].axis('off')

    axes[6].imshow(np.ones((h, w)), cmap='gray', vmin=0, vmax=1) # white background
    for i in range(0, h, step):
        axes[6].plot(map_x[i, :], map_y[i, :], color='blue', linewidth=0.5)
    for j in range(0, w, step):
        axes[6].plot(map_x[:, j], map_y[:, j], color='blue', linewidth=0.5)
    axes[6].set_title('7. Deformed Grid\n(Pred Flow)', fontsize=14, pad=15)
    axes[6].set_aspect('equal')
    axes[6].invert_yaxis()
    axes[6].axis('off')
    # set limits to crop out out-of-bounds lines to match image size
    axes[6].set_xlim(0, w-1)
    axes[6].set_ylim(h-1, 0)

    # Keep one slot for visual breathing room in the grouped layout.
    axes[7].axis('off')

    axes[8].imshow(diff_pre, cmap='hot', vmin=0, vmax=1.0)
    axes[8].set_title('8. Pre-Reg Diff\n|Masked Fixed - Masked Fake C0|', fontsize=14, pad=15)
    axes[8].axis('off')

    axes[9].imshow(diff_post, cmap='hot', vmin=0, vmax=1.0)
    axes[9].set_title('9. Post-Reg Diff\n|Masked Fixed - Masked Warped Fake C0|', fontsize=14, pad=15)
    axes[9].axis('off')

    axes[10].imshow(diff_gt_vs_moving, cmap='hot', vmin=0, vmax=1.0)
    axes[10].set_title('10. GT-Warp Diff\n|Moving + GT Flow - Moving|', fontsize=14, pad=15)
    axes[10].axis('off')

    axes[11].imshow(diff_gt_vs_pred, cmap='hot', vmin=0, vmax=1.0)
    axes[11].set_title('11. GT vs Pred Diff\n|Moving + GT Flow - Warped Real Moving|', fontsize=14, pad=15)
    axes[11].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    os.makedirs(save_dir, exist_ok=True)

    panel_name = f'{idx:04d}_{sample_tag}'
    panel_name = panel_name.replace(' ', '_')
    plt.savefig(os.path.join(save_dir, f'pipeline_result_{panel_name}.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Save standalone images for flexible paper composition.
    single_dir = os.path.join(save_dir, 'single_images', panel_name)
    _save_gray_image(os.path.join(single_dir, '01_moving_real_c1.png'), m)
    _save_gray_image(os.path.join(single_dir, '02_fake_moving_c0.png'), fm)
    _save_gray_image(os.path.join(single_dir, '03_fixed_real_distorted_c0.png'), f)
    if mp_gt is not None:
        _save_gray_image(os.path.join(single_dir, '04_moving_plus_gt_flow.png'), mp_gt)
    else:
        _save_gray_image(os.path.join(single_dir, '04_moving_plus_gt_flow_na.png'), np.zeros_like(m))
    _save_gray_image(os.path.join(single_dir, '05_warped_fake_c0.png'), mp_fake)
    _save_gray_image(os.path.join(single_dir, '06_warped_real_c1.png'), mp_real)
    _save_deformed_grid(os.path.join(single_dir, '07_deformed_grid.png'), flow, step=step)
    _save_gray_image(os.path.join(single_dir, '08_pre_reg_diff.png'), diff_pre, cmap='hot', vmin=0.0, vmax=1.0)
    _save_gray_image(os.path.join(single_dir, '09_post_reg_diff.png'), diff_post, cmap='hot', vmin=0.0, vmax=1.0)
    _save_gray_image(os.path.join(single_dir, '10_gt_warp_diff.png'), diff_gt_vs_moving, cmap='hot', vmin=0.0, vmax=1.0)
    _save_gray_image(os.path.join(single_dir, '11_gt_vs_pred_diff.png'), diff_gt_vs_pred, cmap='hot', vmin=0.0, vmax=1.0)

    with open(os.path.join(single_dir, 'metrics.txt'), 'w', encoding='utf-8') as f_txt:
        f_txt.write(metrics_text + '\n')


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


def save_metrics_summary(output_dir, summary):
    """将汇总指标同时保存为 JSON 和可读文本。"""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'metrics_summary.json')
    txt_path = os.path.join(output_dir, 'metrics_summary.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        'Pipeline Metrics Summary',
        '=' * 80,
        f"samples: {summary['samples']}",
        f"use_foreground_mask: {summary['use_foreground_mask']}",
        f"mask_strategy: {summary['mask_strategy']}",
        f"mask_thresh: {summary['mask_thresh']}",
        f"mask_dilate_ks: {summary['mask_dilate_ks']}",
        '-' * 80,
        '* Intra-modal Metrics (Fake C0 vs Real C0):',
        f"  - Pre-Reg  -> ZNCC : {summary['intra_modal']['pre_zncc']:.4f}  |  MSE : {summary['intra_modal']['pre_mse']:.4f}",
        f"  - Post-Reg -> ZNCC : {summary['intra_modal']['post_zncc']:.4f}  |  MSE : {summary['intra_modal']['post_mse']:.4f}",
        '-' * 80,
        '* Cross-modal Metrics (Real C1 vs Real C0):',
        f"  - Pre-Reg  -> NMI  : {summary['cross_modal']['pre_nmi']:.4f}  |  Fore-Dice : {summary['cross_modal']['pre_dice']:.4f}  |  cross-ZNCC : {summary['cross_modal']['pre_cross_zncc']:.4f}",
        f"  - Post-Reg -> NMI  : {summary['cross_modal']['post_nmi']:.4f}  |  Fore-Dice : {summary['cross_modal']['post_dice']:.4f}  |  cross-ZNCC : {summary['cross_modal']['post_cross_zncc']:.4f}",
        '-' * 80,
        '* Flow Quality Metrics:',
    ]

    if summary['flow_quality']['epe'] is not None:
        lines.append(f"  - EPE (End Point Error)  : {summary['flow_quality']['epe']:.4f}")
    else:
        lines.append('  - EPE (End Point Error)  : N/A (GT flow unavailable)')
    lines.append(f"  - Folding (Negative Jac) : {summary['flow_quality']['folding_ratio']:.4%}")
    lines.append('=' * 80)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"统计指标已保存: {json_path}")
    print(f"统计文本已保存: {txt_path}")

def _resize_tensor_2d(tensor_4d, out_hw, mode='bilinear'):
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return F.interpolate(tensor_4d, size=out_hw, mode=mode, align_corners=True)
    return F.interpolate(tensor_4d, size=out_hw, mode=mode)

def main():
    # --- 1. 配置参数 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 前景掩码实验开关
    use_foreground_mask = True
    # 可选: 'post_gan' (推荐), 'pre_and_post'
    mask_strategy = 'post_gan'
    mask_thresh = 0.01
    mask_dilate_ks = 5

    # Coarse template matching options (after modality conversion).
    enable_coarse_match = True
    coarse_match_stride = 2

    # Input crop simulation: make one image a local crop before matching.
    enable_input_crop_simulation = True
    crop_source = 'moving'  # 'moving' or 'fixed'
    crop_hw = (256, 256)
    crop_random = True

    # 固定可视化样本设置（用于论文对齐）
    # 优先级: fixed_vis_indices > fixed_vis_names > 前 max_visualizations 个样本
    quick_visual_only = True
    max_visualizations = 10
    fixed_vis_indices = None  # 示例: [0, 5, 12, 30]
    fixed_vis_names = None    # 示例: ['0001.png', '0020.png']

    sample_count = max_visualizations if quick_visual_only else None
    sample_seed = 42
    
    # 路径配置（请根据实际路径进行修改）
    registrate_model_name = 'Supervised_ablation_full'
    # affine_only full identity no_affine no_appearance
    
    # test_name = "Test_CrossModal_test_shift"
    # test_name = "Test_CrossModal_test_id"
    # test_name = "Test_CrossModal_test_ffd"
    # test_name = "Test_CrossModal_test_id"
    # id ffd shift

    pix2pix_model_path = "/data2/xujr/output_model/pix2pix_idr0003_c1Toc0/latest_net_G.pth"
    # transmorph_model_path = "/data2/xujr/output_model/Aablation/Supervised_ablation_full/experiments/model_best.pth"
    transmorph_model_path = f"/data2/xujr/output_model/Aablation/{registrate_model_name}/experiments/model_best.pth"
    
    # test_data_dir = '/data2/xujr/idr_data/Test/Test_CrossModal_test_id/ch1_to_ch0'
    # output_dir = "./results/pipeline_results"

    # test_data_dir = f'/data2/xujr/idr_data/Test/{test_name}/ch1_to_ch0'
    # output_dir = f"./results/pipeline_results/{registrate_model_name}_{test_name}"

    test_data_dir = f'/data2/xujr/idr_data/Test/Test_CrossModal_test_id_512/ch1_to_ch0'
    output_dir = f"./results/pipeline_results/{registrate_model_name}_crop"

    # --- 2. 加载模型 ---
    print("正在加载 pix2pix 模型...")
    # netG 参数完全匹配 test_pix2pix_0304.sh 设置
    # input_nc=3, output_nc=3, netG="unet_256", norm="instance", no_dropout=True (即use_dropout=False)
    netG = load_pix2pix_generator(
        pix2pix_model_path, 
        input_nc=3, 
        output_nc=3, 
        ngf=64, 
        netG='unet_256', 
        norm='instance', 
        use_dropout=False, 
        device=device
    )
    
    print("正在加载 TransMorph 模型...")
    transmorph = load_transmorph_model(transmorph_model_path, img_size=(256, 256), device=device)
    if use_foreground_mask:
        print(f"启用前景掩码策略: {mask_strategy}, thresh={mask_thresh}, dilate_ks={mask_dilate_ks}")
    else:
        print("未启用前景掩码策略")

    # --- 3. 准备数据 ---
    # 根据我们重新理清的逻辑：
    # - fixed (带有形变特征的扰动后图像): 被用作后向映射的目标(采样范围界定)，所以对应 channel_1，但它是计算变形场时的基底。
    # - moving (未发生变形的原始图像): 被用作像素抓取的源图，所以对应 channel_0。
    # 在数据集读取 `MultiModalityPairedDataset` 中，
    # data[0] 是 moving, data[1] 是 fixed。
    test_set = SingleModalityPairedDataset(
        target_dir=test_data_dir,
        moving_folder='moving',  # 原始无形变的图
        fixed_folder='fixed',   # 扰动后的变形图
        img_size=(256, 256),
        transforms=None,
        selected_names=fixed_vis_names,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    selected_index_set = set(fixed_vis_indices) if fixed_vis_indices is not None else None
    selected_samples = []
    vis_saved_count = 0

    # --- 4. 运行 Pipeline ---
    print(f"开始 Pipeline 推理，共有 {len(test_set)} 个样本...")
    
    # 实例化 SpatialTransformer 以便施加预测的位移场
    from models.TransMorph import SpatialTransformer
    spatial_trans = SpatialTransformer(CONFIGS_TM['TransMorph'].img_size).to(device)

    # Metric Trackers
    eval_zncc_pre = utils.AverageMeter()
    eval_mse_pre = utils.AverageMeter()
    eval_zncc_post = utils.AverageMeter()
    eval_mse_post = utils.AverageMeter()
    eval_nmi_pre = utils.AverageMeter()
    eval_nmi_post = utils.AverageMeter()
    eval_epe = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_dice_pre = utils.AverageMeter()
    eval_dice_post = utils.AverageMeter()
    eval_cross_zncc_pre = utils.AverageMeter()
    eval_cross_zncc_post = utils.AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if quick_visual_only and vis_saved_count >= max_visualizations:
                print(f"[INFO] quick_visual_only 已达到上限 {max_visualizations}，提前结束。")
                break

            # 将数据移动到设备
            data = [t.to(device) for t in data]
            moving = data[0]  # 未变形的原始图像 (C1)
            fixed = data[1]   # 扰动后的变形图像 (C0) 

            moving_crop_hw = None
            if enable_input_crop_simulation:
                ch, cw = crop_hw
                if crop_source == 'moving':
                    src = moving
                else:
                    src = fixed

                h, w = src.shape[2], src.shape[3]
                if ch > h or cw > w:
                    raise ValueError("crop_hw must be <= source image size")

                if crop_random:
                    y0 = int(torch.randint(0, h - ch + 1, (1,)).item())
                    x0 = int(torch.randint(0, w - cw + 1, (1,)).item())
                else:
                    y0 = (h - ch) // 2
                    x0 = (w - cw) // 2

                if crop_source == 'moving':
                    moving = utils.crop_tensor_2d(moving, y0, x0, ch, cw)
                    moving_crop_hw = (ch, cw)
                    if len(data) >= 3:
                        gt_flow = data[2].to(device)
                        if gt_flow.shape[2:] == data[0].shape[2:]:
                            data[2] = utils.crop_tensor_2d(gt_flow, y0, x0, ch, cw)
                    if len(data) >= 4:
                        valid_mask = data[3].to(device)
                        if valid_mask.shape[2:] == data[0].shape[2:]:
                            data[3] = utils.crop_tensor_2d(valid_mask, y0, x0, ch, cw)
                else:
                    fixed = utils.crop_tensor_2d(fixed, y0, x0, ch, cw)

            if use_foreground_mask:
                # MaskA: foreground mask extracted from RealA (moving)
                mask_a = utils.build_foreground_mask(moving, thresh=mask_thresh, dilate_ks=mask_dilate_ks)
            else:
                mask_a = torch.ones_like(moving)

            # ===== 图像配准的后向映射核心逻辑 =====
            
            # 步骤 1：使用 pix2pix 将 moving(源模态 C1) 转换为 fake_moving(目标模态 C0外观，但未变形)
            moving_for_G = moving
            if enable_input_crop_simulation and moving_for_G.shape[2:] != (256, 256):
                # pix2pix unet_256 expects 256x256 inputs
                moving_for_G = _resize_tensor_2d(moving_for_G, (256, 256))
            if use_foreground_mask and mask_strategy == 'pre_and_post':
                moving_for_G = moving_for_G * mask_a
            if moving_for_G.size(1) == 1:
                moving_for_G = moving_for_G.repeat(1, 3, 1, 1) # 扩展到 3 通道
            
            moving_for_G = (moving_for_G - 0.5) * 2.0 # 将 [0, 1] 转换至 [-1, 1]

            fake_moving = netG(moving_for_G)
            
            fake_moving_for_T = (fake_moving + 1.0) / 2.0 # 转换回 [0, 1]
            if fake_moving_for_T.size(1) == 3:
                fake_moving_for_T = fake_moving_for_T.mean(dim=1, keepdim=True) # 转单通道
            if moving_crop_hw is not None and fake_moving_for_T.shape[2:] != moving_crop_hw:
                fake_moving_for_T = _resize_tensor_2d(fake_moving_for_T, moving_crop_hw)

            if enable_coarse_match:
                fm_np = fake_moving_for_T.detach().cpu().numpy()[0, 0]
                f_np = fixed.detach().cpu().numpy()[0, 0]
                fm_h, fm_w = fm_np.shape
                f_h, f_w = f_np.shape
                # Decide which one is template (smaller) to allow inclusion cases.
                if fm_h <= f_h and fm_w <= f_w:
                    template = fm_np
                    search = f_np
                    template_from = 'moving'
                elif f_h <= fm_h and f_w <= fm_w:
                    template = f_np
                    search = fm_np
                    template_from = 'fixed'
                else:
                    template = None
                    template_from = None

                if template is not None:
                    try:
                        (pred_y, pred_x), coarse_score = utils.TemplateMatchTool.find_best_match(
                            search, template, stride=coarse_match_stride
                        )
                        if i % 10 == 0:
                            print(
                                f"[CoarseMatch] idx={i} top-left=(y={pred_y}, x={pred_x}) score={coarse_score:.4f}"
                            )

                        th, tw = template.shape
                        if template_from == 'moving':
                            # Crop fixed to matched location, keep moving as-is (same size as template).
                            fixed = utils.crop_tensor_2d(fixed, pred_y, pred_x, th, tw)
                            # mask_a and gt_flow already align with moving size.
                        else:
                            # Crop moving/fake to matched location, fixed stays as template.
                            moving = utils.crop_tensor_2d(moving, pred_y, pred_x, th, tw)
                            fake_moving_for_T = utils.crop_tensor_2d(fake_moving_for_T, pred_y, pred_x, th, tw)
                            if use_foreground_mask:
                                mask_a = utils.crop_tensor_2d(mask_a, pred_y, pred_x, th, tw)
                            if len(data) >= 3:
                                gt_flow = data[2].to(device)
                                if gt_flow.shape[2:] == data[0].shape[2:]:
                                    data[2] = utils.crop_tensor_2d(gt_flow, pred_y, pred_x, th, tw)
                            if len(data) >= 4:
                                valid_mask = data[3].to(device)
                                if valid_mask.shape[2:] == data[0].shape[2:]:
                                    data[3] = utils.crop_tensor_2d(valid_mask, pred_y, pred_x, th, tw)
                    except ValueError as e:
                        print(f"[CoarseMatch] idx={i} skipped: {e}")
                else:
                    print(f"[CoarseMatch] idx={i} skipped: unmatched sizes fm={fm_np.shape} fixed={f_np.shape}")

            # Resize crops to model input size.
            target_hw = (256, 256)
            moving = _resize_tensor_2d(moving, target_hw)
            fake_moving_for_T = _resize_tensor_2d(fake_moving_for_T, target_hw)
            fixed = _resize_tensor_2d(fixed, target_hw)
            if use_foreground_mask:
                mask_a = _resize_tensor_2d(mask_a, target_hw, mode='nearest')
            if len(data) >= 4:
                data[3] = _resize_tensor_2d(data[3], target_hw, mode='nearest')

            if use_foreground_mask:
                # Apply MaskA to FakeB and FixedB for masked registration visualization/optimization input.
                reg_src = fake_moving_for_T * mask_a
                # reg_tgt = fixed * mask_a
                reg_tgt = fixed
            else:
                reg_src = fake_moving_for_T
                reg_tgt = fixed

            # 步骤 2：进行配准
            # x_in = torch.cat((source, target), dim=1) -> torch.cat((fake_moving, fixed), dim=1)
            x_in = torch.cat((reg_src, reg_tgt), dim=1)
            output = transmorph(x_in)
            
            # 提取配准结果
            moved_pred_fake = output[0]  # 这是网络内部将 fake_moving (C0) 按照预测流场变形后的图
            flow_pred = output[1]        # 预测的流场 (位移场, DDF)
            # output[2] 是 积分前的平稳速度场(SVF)，在这里推理不需要使用

            # 步骤 3：将预测得到的流场施加在原本的真 moving(C1) 上
            moved_pred_real = spatial_trans(moving, flow_pred)
            moved_gt_real = None

            # --- 中间评价指标：计算 EPE 和 雅可比折叠比例 (Folding Ratio) ---
            if len(data) >= 4:
                valid_mask = data[3].to(device)
            else:
                valid_mask = (fixed > 1e-4).float()
            if use_foreground_mask:
                valid_mask = valid_mask * mask_a
                
            # EPE 计算 (需要存在 GT flow, index 2)
            if len(data) >= 3:
                gt_flow = data[2].to(device)
                if gt_flow.shape[2:] != moving.shape[2:]:
                    if i == 0:
                        print(f"检测到 GT flow 尺寸 {tuple(gt_flow.shape[2:])} 与图像尺寸 {tuple(moving.shape[2:])} 不一致，已自动重采样并缩放位移。")
                    gt_flow = resize_flow_to_target(gt_flow, moving.shape[2:])
                moved_gt_real = spatial_trans(moving, gt_flow)
                diff = flow_pred - gt_flow
                epe = torch.norm(diff, p=2, dim=1, keepdim=True)
                masked_epe = epe * valid_mask
                mean_epe = masked_epe.sum() / (valid_mask.sum() + 1e-8)
                eval_epe.update(mean_epe.item(), moving.size(0))
            
            # 雅可比行列式计算 (Folding Ratio)
            flow_np = flow_pred.detach().cpu().numpy()[0].transpose(1, 2, 0)
            jac_det = utils.jacobian_determinant_vxm(flow_np)
            mask_np = valid_mask.detach().cpu().numpy()[0, 0]
            valid_pixels = np.sum(mask_np > 0)
            folding_ratio = np.sum((jac_det <= 0) & (mask_np > 0)) / (valid_pixels + 1e-8)
            eval_det.update(folding_ratio, moving.size(0))

            if i % 10 == 0:
                print(f"正在处理第 {i} 个样本...")
            
            # --- 评价指标计算 ---
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy()[0, 0, :, :]
            
            m = to_numpy(moving) 
            f = to_numpy(fixed)
            fm = to_numpy(fake_moving_for_T)
            mp_fake = to_numpy(moved_pred_fake)
            mp_real = to_numpy(moved_pred_real)
            
            zncc_pre = utils.compute_zncc(fm, f)
            mse_pre = utils.compute_mse(fm, f)
            zncc_post = utils.compute_zncc(mp_fake, f)
            mse_post = utils.compute_mse(mp_fake, f)

            nmi_pre = utils.compute_nmi(m, f)
            nmi_post = utils.compute_nmi(mp_real, f)

            dice_pre = utils.compute_foreground_dice(m, f)
            dice_post = utils.compute_foreground_dice(mp_real, f)
            
            cross_zncc_pre = utils.compute_zncc(m, f)
            cross_zncc_post = utils.compute_zncc(mp_real, f)
            
            # 更新 trackers
            eval_zncc_pre.update(zncc_pre, moving.size(0))
            eval_mse_pre.update(mse_pre, moving.size(0))
            eval_zncc_post.update(zncc_post, moving.size(0))
            eval_mse_post.update(mse_post, moving.size(0))
            eval_nmi_pre.update(nmi_pre, moving.size(0))
            eval_nmi_post.update(nmi_post, moving.size(0))
            eval_dice_pre.update(dice_pre, moving.size(0))
            eval_dice_post.update(dice_post, moving.size(0))
            eval_cross_zncc_pre.update(cross_zncc_pre, moving.size(0))
            eval_cross_zncc_post.update(cross_zncc_post, moving.size(0))
            
            # 保存可视化结果：支持固定索引/固定文件名，确保跨方法一致选图。
            should_save = False
            if selected_index_set is not None:
                should_save = i in selected_index_set
            elif fixed_vis_names is not None:
                should_save = True
            else:
                should_save = vis_saved_count < max_visualizations

            if should_save:
                if i >= len(test_set.pairs):
                    sample_name = f'sample_{i:04d}'
                else:
                    sample_name = os.path.basename(test_set.pairs[i]['moving'])
                sample_tag = os.path.splitext(sample_name)[0]

                metrics_text = (
                    f"--- Model Evaluation Metrics ---\n"
                    f"[Intra-modal (Fake C0 vs Fixed C0)] Pre-Reg: ZNCC={zncc_pre:.4f}/MSE={mse_pre:.4f}  |  Post-Reg: ZNCC={zncc_post:.4f}/MSE={mse_post:.4f}\n"
                    f"[Cross-modal (Real C1 vs Fixed C0)] Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}/cross-ZNCC={cross_zncc_pre:.4f}  |  Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}/cross-ZNCC={cross_zncc_post:.4f}"
                )
                fixed_vis = fixed * mask_a if use_foreground_mask else fixed
                fake_vis = fake_moving_for_T * mask_a if use_foreground_mask else fake_moving_for_T
                moved_fake_vis = moved_pred_fake * mask_a if use_foreground_mask else moved_pred_fake
                plot_pipeline_results(
                    moving,
                    fixed_vis,
                    fake_vis,
                    moved_fake_vis,
                    moved_pred_real,
                    flow_pred,
                    moved_gt_real,
                    i,
                    sample_tag,
                    output_dir,
                    metrics_text,
                )
                selected_samples.append((i, sample_name))
                vis_saved_count += 1
                
    print(f"\nPipeline 推理完成。测试图保存在 {output_dir}\n")
    if quick_visual_only:
        print(f"[INFO] quick_visual_only=True，已跳过全量统计与 summary 文件输出。")
    else:
        print("="*80)
        print("[Method: Our Pipeline]")
        print("-" * 80)
        print("* Intra-modal Metrics (Fake C0 vs Real C0):")
        print(f"  - Pre-Reg  -> ZNCC : {eval_zncc_pre.avg:.4f}  |  MSE : {eval_mse_pre.avg:.4f}")
        print(f"  - Post-Reg -> ZNCC : {eval_zncc_post.avg:.4f}  |  MSE : {eval_mse_post.avg:.4f}")
        print("-" * 80)
        print("* Cross-modal Metrics (Real C1 vs Real C0):")
        print(f"  - Pre-Reg  -> NMI  : {eval_nmi_pre.avg:.4f}  |  Fore-Dice : {eval_dice_pre.avg:.4f}  |  cross-ZNCC : {eval_cross_zncc_pre.avg:.4f}")
        print(f"  - Post-Reg -> NMI  : {eval_nmi_post.avg:.4f}  |  Fore-Dice : {eval_dice_post.avg:.4f}  |  cross-ZNCC : {eval_cross_zncc_post.avg:.4f}")
        print("-" * 80)
        print("* Flow Quality Metrics:")
        if eval_epe.count > 0:
            print(f"  - EPE (End Point Error)  : {eval_epe.avg:.4f}")
        print(f"  - Folding (Negative Jac) : {eval_det.avg:.4%}")
        print("="*80 + "\n")

        summary = {
            'samples': len(test_set),
            'use_foreground_mask': use_foreground_mask,
            'mask_strategy': mask_strategy,
            'mask_thresh': mask_thresh,
            'mask_dilate_ks': mask_dilate_ks,
            'intra_modal': {
                'pre_zncc': float(eval_zncc_pre.avg),
                'pre_mse': float(eval_mse_pre.avg),
                'post_zncc': float(eval_zncc_post.avg),
                'post_mse': float(eval_mse_post.avg),
            },
            'cross_modal': {
                'pre_nmi': float(eval_nmi_pre.avg),
                'pre_dice': float(eval_dice_pre.avg),
                'pre_cross_zncc': float(eval_cross_zncc_pre.avg),
                'post_nmi': float(eval_nmi_post.avg),
                'post_dice': float(eval_dice_post.avg),
                'post_cross_zncc': float(eval_cross_zncc_post.avg),
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
