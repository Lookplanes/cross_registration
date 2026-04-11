import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# 添加当前目录以便引入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import utils
from data.datasets import SingleModalityPairedDataset

# 尝试导入 ants
try:
    import ants
except ImportError:
    print("\n[错误] 未找到 antspyx 包！\n请在终端中运行以下命令安装:\npip install antspyx\n")
    sys.exit(1)

def plot_ants_results(moving, fixed, moved_pred, idx, save_dir, metrics_text):
    m = moving 
    f = fixed
    mp = moved_pred
    
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
    
    plt.savefig(os.path.join(save_dir, f'baseline_ants_result_{idx}.png'), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # --- 1. 配置参数 ---
    # 传统方法跑 CPU 就行，不需要显卡，但为了保持加载器一致性保留这部分格式
    test_data_dir = '/data2/xujr/idr_data/Train_CrossModal/Test/ch0_to_ch1'
    output_dir = "./results/baseline_ants_results"

    # --- 2. 准备数据 ---
    test_set = SingleModalityPairedDataset(
        target_dir=test_data_dir,
        moving_folder='moving',  # 原始无形变的图(C1)
        fixed_folder='fixed',   # 扰动后的变形图(C0)
        img_size=(256, 256),
        transforms=None 
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

    for i, data in enumerate(test_loader):
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
        
        if i < 5:
            metrics_text = (
                f"--- ANTs/SyN Baseline Metrics ---\n"
                f"Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}/ZNCC={zncc_pre:.4f}\n"
                f"Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}/ZNCC={zncc_post:.4f}\n"
                f"(Optimization Time: {elapsed_time:.3f} s/pair)"
            )
            plot_ants_results(m_np, f_np, mp_np, i, output_dir, metrics_text)
            
    print(f"\nANTs 推理及评价完成。测试图保存在 {output_dir}\n")
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

if __name__ == '__main__':
    main()
