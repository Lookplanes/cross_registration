import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加当前目录以便引入 TransMorph 相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 引入 TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import utils
from data.datasets import SingleModalityPairedDataset

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

def plot_baseline_results(moving, fixed, moved_pred, idx, save_dir, metrics_text):
    """
    保存无监督 Baseline 可视化结果（仅3个面板，因为没有 Pix2Pix）
    """
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()[0, 0, :, :]

    m = to_numpy(moving) 
    f = to_numpy(fixed)
    mp = to_numpy(moved_pred)
    
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
    
    plt.savefig(os.path.join(save_dir, f'baseline_unsup_result_{idx}.png'), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # --- 1. 配置参数 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置（请根据实际的无监督模型路径修改）
    transmorph_model_path = "/data2/xujr/output_model/0330/TransMorph_unsupervised_ncc_1.0_1.0/experiments/model_best.pth"
    test_data_dir = '/data2/xujr/idr_data/Train_CrossModal/Test/ch1_to_ch0'
    output_dir = "./results/baseline_unsupervised_results"

    # --- 2. 加载模型 ---
    print("正在加载 TransMorph 模型...")
    transmorph = load_transmorph_model(transmorph_model_path, img_size=(256, 256), device=device)

    # --- 3. 准备数据 ---
    test_set = SingleModalityPairedDataset(
        target_dir=test_data_dir,
        moving_folder='moving',  # 原始无形变的图(C1)
        fixed_folder='fixed',   # 扰动后的变形图(C0)
        img_size=(256, 256),
        transforms=None 
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

    def compute_zncc(I, J, eps=1e-5):
        I_mean, J_mean = np.mean(I), np.mean(J)
        cross = np.sum((I - I_mean) * (J - J_mean))
        I_var, J_var = np.sum((I - I_mean)**2), np.sum((J - J_mean)**2)
        return cross / (np.sqrt(I_var * J_var) + eps)

    def compute_nmi(I, J, bins=256):
        hist_2d, _, _ = np.histogram2d(I.flatten(), J.flatten(), bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_nz = px[px > 0]
        py_nz = py[py > 0]
        pxy_nz = pxy[pxy > 0]
        hx = -np.sum(px_nz * np.log(px_nz))
        hy = -np.sum(py_nz * np.log(py_nz))
        hxy = -np.sum(pxy_nz * np.log(pxy_nz))
        return (hx + hy) / hxy if hxy > 0 else 0

    def compute_foreground_dice(I, J, thresh=0.01):
        m_I = I > thresh
        m_J = J > thresh
        intersection = np.sum(m_I & m_J)
        return (2. * intersection) / (np.sum(m_I) + np.sum(m_J) + 1e-8)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
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

            # --- 中间评价指标：计算 EPE 和 雅可比折叠比例 (Folding Ratio) ---
            if len(data) >= 4:
                valid_mask = data[3].to(device)
            else:
                valid_mask = (fixed > 1e-4).float()
                
            if len(data) >= 3:
                gt_flow = data[2].to(device)
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
            zncc_pre = compute_zncc(m, f)
            zncc_post = compute_zncc(mp, f)
            nmi_pre = compute_nmi(m, f)
            nmi_post = compute_nmi(mp, f)
            dice_pre = compute_foreground_dice(m, f)
            dice_post = compute_foreground_dice(mp, f)
            
            eval_zncc_pre.update(zncc_pre, moving.size(0))
            eval_zncc_post.update(zncc_post, moving.size(0))
            eval_nmi_pre.update(nmi_pre, moving.size(0))
            eval_nmi_post.update(nmi_post, moving.size(0))
            eval_dice_pre.update(dice_pre, moving.size(0))
            eval_dice_post.update(dice_post, moving.size(0))
            
            if i < 5:
                metrics_text = (
                    f"--- Unsupervised Baseline Metrics ---\n"
                    f"Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}/Cross-ZNCC={zncc_pre:.4f}\n"
                    f"Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}/Cross-ZNCC={zncc_post:.4f}"
                )
                plot_baseline_results(moving, fixed, moved_pred_real, i, output_dir, metrics_text)
                
    print(f"\nBaseline 推理完成。测试图保存在 {output_dir}")
    print("\n" + "="*50)
    print("==== Unsupervised TransMorph Baseline Final ====")
    print(f"Cross-modal Direct - Before Registration: ZNCC = {eval_zncc_pre.avg:.4f}")
    print(f"Cross-modal Direct - After Registration : ZNCC = {eval_zncc_post.avg:.4f}")
    print(f"Cross-modal Mappings - Before Registration: NMI  = {eval_nmi_pre.avg:.4f}, Fore-Dice = {eval_dice_pre.avg:.4f}")
    print(f"Cross-modal Mappings - After Registration : NMI  = {eval_nmi_post.avg:.4f}, Fore-Dice = {eval_dice_post.avg:.4f}")
    if eval_epe.count > 0:
        print(f"Flow Quality Metrics - EPE = {eval_epe.avg:.4f}")
    print(f"Flow Quality Metrics - Negative Jacobian Ratio (Folding) = {eval_det.avg:.6%}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
