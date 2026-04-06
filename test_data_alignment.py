import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import SSIM, ssim

# 将当前目录加到 sys.path，保证加载与 train 通道一样
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import datasets
from data.datasets import SingleModalityPairedDataset
from models.TransMorph import CONFIGS as CONFIGS_TM
from models.TransMorph import SpatialTransformer

def main():
    # ==== 1. 配置路径 ====
    # 修改为您实际的数据路径
    data_dir = '/data2/xujr/idr_data/dataset_processed/idr0003_test_samples/screenA' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = CONFIGS_TM['TransMorph']
    
    # ==== 2. 加载数据集 ====
    print(f"Loading dataset from: {data_dir}")
    # 改为使用 SingleModalityPairedDataset，指定具体的通道文件夹
    dataset = SingleModalityPairedDataset(
        target_dir=data_dir,
        moving_folder='channel_0',  # 未扰动图 (配准目标/Source)
        fixed_folder='channel_0',   # 扰动图 (配准基准/Target)  注意这里通常单模态时这两个文件夹内容是同模态的
        img_size=config.img_size,
        transforms=None
    )
    
    if len(dataset) == 0:
        print("Dataset is empty! Check path or subfolders.")
        return
        
    print(f"Dataset loaded. Total validated samples: {len(dataset)}")
    
    # random_seed = 42
    # np.random.seed(random_seed)
    random_idx = np.random.randint(0, len(dataset))
    data = dataset[random_idx]  
    
    moving = data[0].unsqueeze(0).to(device)  # [1, 1, H, W] -> x (未扰动图)
    fixed  = data[1].unsqueeze(0).to(device)  # [1, 1, H, W] -> y (扰动图)

    gt_flow = data[2].unsqueeze(0).to(device) # [1, 2, H, W]
    
    print(f"Loaded Tensors - Moving: {moving.shape}, Fixed: {fixed.shape}, Flow: {gt_flow.shape}")
    
    # ==== 3. 执行确切的空间变换 ====
    # 抽取与模型内完全一模一样的变形算子
    spatial_trans = SpatialTransformer(config.img_size).to(device)
    
    with torch.no_grad():
        # 用 GT flow 将 moving (原图) 扭曲，试图逼近 fixed (带扰动)
        # 如果上下游对齐，这时候出来的 warped_moving 应该和 fixed 肉眼几乎一模一样
        warped_moving = spatial_trans(moving, gt_flow)
        
    # ==== 4. 计算验证指标 ====
    # Since appearance is augmented independently, SSIM between deformed moving and fixed might be low 
    # even with perfect structural alignment. So we evaluate Structural Similarity using robust metrics 
    # (Optional) and focus heavily on checking if flow causes structural overlap using Normalized Cross Correlation (NCC) or just pure Masked MSE on geometry if applicable.
    # We will use Normalized Cross Correlation (NCC) which is more robust to intensity differences than MSE/SSIM.

    def ncc_loss(I, J, eps=1e-5):
        # Compute zero mean normalized cross correlation (ZNCC)
        I_mean = I.mean()
        J_mean = J.mean()
        I_centered = I - I_mean
        J_centered = J - J_mean
        
        cross = (I_centered * J_centered).sum()
        I_var = (I_centered ** 2).sum()
        J_var = (J_centered ** 2).sum()
        
        return cross / (torch.sqrt(I_var * J_var) + eps)
    
    score_ncc_before = ncc_loss(moving, fixed).item()
    score_ncc_after = ncc_loss(warped_moving, fixed).item()
    
    # 额外评估图像梯度的重合度（对光照变化相对鲁棒）
    def grad2d(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    moving_dx, moving_dy = grad2d(warped_moving)
    fixed_dx, fixed_dy = grad2d(fixed)
    
    grad_ncc_x = ncc_loss(moving_dx, fixed_dx).item()
    grad_ncc_y = ncc_loss(moving_dy, fixed_dy).item()
    
    print(f"\n--- Alignment Metrics (Robust to Intensity Augmentation) ---")
    print(f"Before Warping ZNCC: {score_ncc_before:.4f}")
    print(f"After  Warping ZNCC: {score_ncc_after:.4f}")
    print(f"After Warping Gradient ZNCC (X/Y): {grad_ncc_x:.4f} / {grad_ncc_y:.4f}")

    if score_ncc_after > score_ncc_before + 0.1:
        print("\n>>> ALIGNMENT IMPROVED! Flow is fundamentally moving structures in the right direction.")
    else:
        print("\n>>> CAUTION: Alignment did not significantly improve. Flow might be mismatched with SpatialTransformer.")

    # ==== 5. 画图排查 ====
    moving_np = moving[0, 0].cpu().numpy()
    fixed_np = fixed[0, 0].cpu().numpy()
    warped_np = warped_moving[0, 0].cpu().numpy()
    
    # 画差异图辅助判断
    diff_before = abs(moving_np - fixed_np)
    diff_after = abs(warped_np - fixed_np)
    
    plt.figure(figsize=(25, 5))
    
    plt.subplot(1, 5, 1)
    plt.title('1. Moving (Un-deformed)')
    plt.imshow(moving_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title('2. Fixed (Deformed Target)')
    plt.imshow(fixed_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title('3. Warped Result (moving + gt_flow)')
    plt.imshow(warped_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title(f'Diff Before (ZNCC: {score_ncc_before:.2f})')
    plt.imshow(diff_before, cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.title(f'Diff After (ZNCC: {score_ncc_after:.2f})')
    plt.imshow(diff_after, cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    
    save_path = 'data_alignment_check.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    main()
