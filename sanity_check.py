import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import datasets
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import losses

class MaskedFlowMSE(nn.Module):
    def __init__(self):
        super(MaskedFlowMSE, self).__init__()

    def forward(self, pred_flow, gt_flow, valid_mask):
        sq_error = (pred_flow - gt_flow) ** 2
        masked_sq_error = sq_error * valid_mask
        valid_pixel_count = valid_mask.sum() * 2 
        loss = masked_sq_error.sum() / (valid_pixel_count + 1e-8)
        return loss

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 把 Batch Size 设为 2。
    batch_size = 2
    train_dir = '/data2/xujr/idr_data/Train_Supervised_SynthMorph/Train'
    
    config = CONFIGS_TM['TransMorph']
    config.in_chans = 2
    
    print(f"Loading Dataset from: {train_dir}")
    train_set = datasets.MultiModalityPairedDataset(
        root_dir=train_dir,
        img_size=config.img_size,
        transforms=None
    )
    
    if len(train_set) == 0:
        print("Dataset empty! Check your train_dir path.")
        return
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("Initializing TransMorph Model...")
    model = TransMorph.TransMorph(config).to(device)
    model.train()
    
    criterion_mse = MaskedFlowMSE()
    criterion_reg = losses.Grad('l2')
    
    # 2. 跑1个 Iteration
    print("\n" + "="*50)
    print("🚀 STARTING SANITY CHECK (1 Iteration) 🚀")
    print("="*50)
    
    for i, data in enumerate(train_loader):
        data = [t.to(device) for t in data]
        
        x = data[0] # moving
        y = data[1] # fixed
        gt_flow = data[2]
        
        if len(data) >= 4:
            valid_mask = data[3]
            print(f"-> Mask found. Valid pixel count ratio: {(valid_mask.sum()/(valid_mask.numel())):.2%}")
        else:
            valid_mask = (y > 1e-4).float()
            print(f"-> No mask in dataset, generated from fixed image. Valid ratio: {(valid_mask.sum()/(valid_mask.numel())):.2%}")
            
        # 前向传播
        x_in = torch.cat((x, y), dim=1)
        output = model(x_in)
        
        pred_img = output[0]
        pred_flow = output[1]
        
        print("\n[一] SHAPE MATCHING CHECK (通道序列是否对的上):")
        print(f"  Pred Flow Shape: {pred_flow.shape}   |  Expected: [Batch, 2, H, W]")
        print(f"  GT Flow Shape:   {gt_flow.shape}   |  Expected: [Batch, 2, H, W]")
        
        if pred_flow.shape != gt_flow.shape:
            print("  ❌ ERROR: Flow shapes do not match! You might need to transpose GT Flow.")
        else:
            print("  ✅ Shapes match perfectly.")
        
        # 3. 打印 Pred Flow / GT Flow 的 max/min (尺度差了 100 倍？)
        print("\n[二] EXACT VALUES & SCALE CHECK (尺度是否差了百倍):")
        p_max, p_min = pred_flow.max().item(), pred_flow.min().item()
        g_max, g_min = gt_flow.max().item(), gt_flow.min().item()
        
        print(f"  PRED Flow -> Max: {p_max:8.4f}, Min: {p_min:8.4f}, Mean: {pred_flow.mean().item():8.4f}")
        print(f"  GT   Flow -> Max: {g_max:8.4f}, Min: {g_min:8.4f}, Mean: {gt_flow.mean().item():8.4f}")
        
        # 简单诊断：
        # 如果模型刚刚初始化，Pred flow 通常非常接近 0 (小随机数)。
        # 但我们主要看 GT flow 是否在合理范围内 (比如绝对像素位移通常是 [-30, 30] 甚至更大，而如果是归一化可能会是 [-1, 1])
        print(f"  --> Diagnosis on GT Flow: Abs max displacement is {max(abs(g_max), abs(g_min)):.2f}.")
        if max(abs(g_max), abs(g_min)) <= 1.0:
            print("      ⚠️ Warning: GT Flow seems to be normalized [-1, 1]! TransMorph native predicts absolute pixel unit. Ensure they match!")
        else:
            print("      ✅ GT Flow uses absolute pixels (which matches TransMorph default SpatialTransformer logic).")

        # 4. 打印 MSE Loss 和 Reg Loss 的裸值（乘权重之前）
        print("\n[三] LOSS RAW VALUES CHECK (损失裸值是否爆炸):")
        loss_mse = criterion_mse(pred_flow, gt_flow, valid_mask)
        loss_reg = criterion_reg(pred_flow, None)
        
        print(f"  Raw Masked MSE Loss : {loss_mse.item():.4f}")
        print(f"  Raw Smoothness Loss : {loss_reg.item():.4f}")
        
        if loss_mse.item() > 500:
            print("  ⚠️ Warning: Initial MSE is very high! If learning rate is standard (e.g., 1e-4), this might cause gradients to explode. Consider lowering LR or dividing loss weight.")
            
        print("\n" + "="*50)
        print("Sanity Check 完成，你可以根据以上输出排查水土不服点！")
        break

if __name__ == '__main__':
    main()