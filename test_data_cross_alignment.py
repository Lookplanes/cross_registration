import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加当前目录以便引入 TransMorph 相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 将 pix2pix 的路径加入环境变量以便导入
pix2pix_dir = os.path.join(current_dir, 'pytorch-CycleGAN-and-pix2pix')
sys.path.append(pix2pix_dir)

# 引入 TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM
from models.TransMorph import SpatialTransformer
import utils

from data.datasets import SingleModalityPairedDataset
from pipeline_cross_modality_registration import load_pix2pix_generator

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

def main():
    # ==== 1. 配置参数 ====
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 修改为您跨模态测试的具体路径
    test_data_dir = '/data2/xujr/idr_data/dataset_crossmodal_test/Test/ch1_to_ch0'
    pix2pix_model_path = "/data2/xujr/output_model/pix2pix_idr0003_c1Toc0/latest_net_G.pth"
    save_path = 'cross_modal_alignment_check.png'

    config = CONFIGS_TM['TransMorph']
    
    # ==== 2. 加载与 Pipeline 相同的生成器 ====
    print("Loading pix2pix generator...")
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
    
    # ==== 3. 加载数据集 ====
    print(f"Loading dataset from: {test_data_dir}")
    # 用户纠正：moving 是未经扰动的 C1，fixed 是扰动后的 C0。
    # pix2pix 的方向是 C1 to C0
    dataset = SingleModalityPairedDataset(
        target_dir=test_data_dir,
        moving_folder='moving',  
        fixed_folder='fixed',   
        img_size=config.img_size,
        transforms=None 
    )
    
    if len(dataset) == 0:
        print("Dataset is empty! Check path or subfolders.")
        return
        
    print(f"Dataset loaded. Total validated samples: {len(dataset)}")
    
    random_idx = np.random.randint(0, len(dataset))
    data = dataset[random_idx]  
    
    moving = data[0].unsqueeze(0).to(device)  # [1, 1, H, W] (原图/采样源)
    fixed  = data[1].unsqueeze(0).to(device)  # [1, 1, H, W] (扰动图/含形变的网格源)

    if len(data) > 2 and data[2] is not None:
        gt_flow = data[2].unsqueeze(0).to(device) # [1, 2, H, W]
    else:
        print("Error: No Ground Truth flow found for this dataset. This check script requires GT flow.")
        return
        
    print(f"Loaded Tensors - Moving: {moving.shape}, Fixed: {fixed.shape}, Flow: {gt_flow.shape}")
    
    # ==== 4. 测试跨模态的后向映射过程 ====
    spatial_trans = SpatialTransformer(config.img_size).to(device)
    
    with torch.no_grad():
        # A. 通过生成器将 moving (未经扰动的C1) 转换为 fake_moving (具备C0外观但无扰动)
        moving_for_G = moving.repeat(1, 3, 1, 1) # 1ch -> 3ch
        moving_for_G = (moving_for_G - 0.5) * 2.0 # [0,1] -> [-1,1]
        
        fake_moving = netG(moving_for_G)
        
        fake_moving = (fake_moving + 1.0) / 2.0 # [-1,1] -> [0,1]
        fake_moving = fake_moving.mean(dim=1, keepdim=True) # 3ch -> 1ch

        # B. 将转换模态后的 fake_moving (C0无扰动) 按照 GT Flow 变形
        # 变形后它应该结构上与 fixed (扰动后的C0) 一致，由于已经是C0外观，纹理也应该一致
        warped_fake_moving = spatial_trans(fake_moving, gt_flow)
        
        # 为了对比，我们也用 GT Flow 变形原始的 moving(C1) 
        warped_moving_original = spatial_trans(moving, gt_flow)

        
    # ==== 5. 画图排查 ====
    moving_np = moving[0, 0].cpu().numpy()
    fixed_np = fixed[0, 0].cpu().numpy()
    warped_moving_np = warped_moving_original[0, 0].cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('1. Moving')
    plt.imshow(moving_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('2. Fixed')
    plt.imshow(fixed_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('3. Warped Moving (Moving + GT_flow)')
    plt.imshow(warped_moving_np, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    main()