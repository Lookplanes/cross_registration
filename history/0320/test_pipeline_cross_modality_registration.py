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
            
        model.load_state_dict(new_state_dict)
        print(f"成功加载 TransMorph 模型: {model_path}")
        
    model.to(device)
    model.eval()
    return model

def plot_pipeline_results(moving, fixed, fake_moving, moved_pred_fake, moved_pred_real, idx, save_dir, metrics_text):
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
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 7))
    
    plt.suptitle(metrics_text, fontsize=15, fontweight='bold', y=0.98)

    axes[0].imshow(m, cmap='gray')
    axes[0].set_title('1. Moving\n(Real C1)', fontsize=14, pad=15)
    axes[0].axis('off')

    axes[1].imshow(fm, cmap='gray')
    axes[1].set_title('2. Fake Moving\n(Fake C0)', fontsize=14, pad=15)
    axes[1].axis('off')

    axes[2].imshow(f, cmap='gray')
    axes[2].set_title('3. Fixed\n(Real Distorted C0)', fontsize=14, pad=15)
    axes[2].axis('off')

    axes[3].imshow(mp_fake, cmap='gray')
    axes[3].set_title('4. Warped Fake Moving\n(Fake C0 + Flow)', fontsize=14, pad=15)
    axes[3].axis('off')
    
    axes[4].imshow(mp_real, cmap='gray')
    axes[4].set_title('5. Warped Real Moving\n(Real C1 + Flow)', fontsize=14, pad=15)
    axes[4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f'pipeline_result_{idx}.png'), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # --- 1. 配置参数 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置（请根据实际路径进行修改）
    pix2pix_model_path = "/data2/xujr/output_model/pix2pix_idr0003_c0Toc1/latest_net_G.pth"
    transmorph_model_path = "/data2/xujr/output_model/0320/TransMorph_supervised_l1_smooth_1_0.05/experiments/model_best.pth"
    test_data_dir = '/data2/xujr/idr_data/Train_CrossModal/Test/ch0_to_ch1'
    output_dir = "./results/pipeline_results"

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
        transforms=None 
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

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

    # === 计算评价指标（在同一模态下比较 Fake C0 与 Real C0） ===
    def compute_zncc(I, J, eps=1e-5):
        I_mean, J_mean = np.mean(I), np.mean(J)
        cross = np.sum((I - I_mean) * (J - J_mean))
        I_var, J_var = np.sum((I - I_mean)**2), np.sum((J - J_mean)**2)
        return cross / (np.sqrt(I_var * J_var) + eps)
        
    def compute_mse(I, J):
        return np.mean((I - J) ** 2)

    # === 计算跨模态评价指标（归一化互信息 NMI） ===
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
        """用阈值粗略剥离背景，计算前景（组织结构）重叠的 DICE"""
        m_I = I > thresh
        m_J = J > thresh
        intersection = np.sum(m_I & m_J)
        return (2. * intersection) / (np.sum(m_I) + np.sum(m_J) + 1e-8)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # 将数据移动到设备
            data = [t.to(device) for t in data]
            moving = data[0]  # 未变形的原始图像 (C1)
            fixed = data[1]   # 扰动后的变形图像 (C0) 

            # ===== 图像配准的后向映射核心逻辑 =====
            
            # 步骤 1：使用 pix2pix 将 moving(源模态 C1) 转换为 fake_moving(目标模态 C0外观，但未变形)
            moving_for_G = moving
            if moving_for_G.size(1) == 1:
                moving_for_G = moving_for_G.repeat(1, 3, 1, 1) # 扩展到 3 通道
            
            moving_for_G = (moving_for_G - 0.5) * 2.0 # 将 [0, 1] 转换至 [-1, 1]

            fake_moving = netG(moving_for_G)
            
            fake_moving_for_T = (fake_moving + 1.0) / 2.0 # 转换回 [0, 1]
            if fake_moving_for_T.size(1) == 3:
                fake_moving_for_T = fake_moving_for_T.mean(dim=1, keepdim=True) # 转单通道

            # 步骤 2：进行配准
            # x_in = torch.cat((source, target), dim=1) -> torch.cat((fake_moving, fixed), dim=1)
            x_in = torch.cat((fake_moving_for_T, fixed), dim=1)
            output = transmorph(x_in)
            
            # 提取配准结果
            moved_pred_fake = output[0]  # 这是网络内部将 fake_moving (C0) 按照预测流场变形后的图
            flow_pred = output[1]        # 预测的流场 (位移场, DDF)
            # output[2] 是 积分前的平稳速度场(SVF)，在这里推理不需要使用

            # 步骤 3：将预测得到的流场施加在原本的真 moving(C1) 上
            moved_pred_real = spatial_trans(moving, flow_pred)

            # --- 中间评价指标：计算 EPE 和 雅可比折叠比例 (Folding Ratio) ---
            if len(data) >= 4:
                valid_mask = data[3].to(device)
            else:
                valid_mask = (fixed > 1e-4).float()
                
            # EPE 计算 (需要存在 GT flow, index 2)
            if len(data) >= 3:
                gt_flow = data[2].to(device)
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
            
            zncc_pre = compute_zncc(fm, f)
            mse_pre = compute_mse(fm, f)
            zncc_post = compute_zncc(mp_fake, f)
            mse_post = compute_mse(mp_fake, f)

            nmi_pre = compute_nmi(m, f)
            nmi_post = compute_nmi(mp_real, f)
            
            dice_pre = compute_foreground_dice(m, f)
            dice_post = compute_foreground_dice(mp_real, f)
            
            # 更新 trackers
            eval_zncc_pre.update(zncc_pre, moving.size(0))
            eval_mse_pre.update(mse_pre, moving.size(0))
            eval_zncc_post.update(zncc_post, moving.size(0))
            eval_mse_post.update(mse_post, moving.size(0))
            eval_nmi_pre.update(nmi_pre, moving.size(0))
            eval_nmi_post.update(nmi_post, moving.size(0))
            eval_dice_pre.update(dice_pre, moving.size(0))
            eval_dice_post.update(dice_post, moving.size(0))
            
            # 保存可视化结果 (仅前 5 个样本)
            if i < 5:
                metrics_text = (
                    f"--- Model Evaluation Metrics ---\n"
                    f"[Intra-modal (Fake C0 vs Fixed C0)] Pre-Reg: ZNCC={zncc_pre:.4f}/MSE={mse_pre:.4f}  |  Post-Reg: ZNCC={zncc_post:.4f}/MSE={mse_post:.4f}\n"
                    f"[Cross-modal (Real C1 vs Fixed C0)] Pre-Reg: NMI={nmi_pre:.4f}/Dice={dice_pre:.4f}  |  Post-Reg: NMI={nmi_post:.4f}/Dice={dice_post:.4f}"
                )
                plot_pipeline_results(moving, fixed, fake_moving_for_T, moved_pred_fake, moved_pred_real, i, output_dir, metrics_text)
                
    print(f"\nPipeline 推理完成。测试图保存在 {output_dir}")
    print("\n" + "="*50)
    print("==== Final ====")
    print(f"Intra-modal (Fake C0 vs Fixed C0) - Before Registration: ZNCC = {eval_zncc_pre.avg:.4f}, MSE = {eval_mse_pre.avg:.4f}")
    print(f"Intra-modal (Fake C0 vs Fixed C0) - After Registration : ZNCC = {eval_zncc_post.avg:.4f}, MSE = {eval_mse_post.avg:.4f}")
    print(f"Cross-modal (Real C1 vs Fixed C0) - Before Registration: NMI  = {eval_nmi_pre.avg:.4f}, Fore-Dice = {eval_dice_pre.avg:.4f}")
    print(f"Cross-modal (Real C1 vs Fixed C0) - After Registration : NMI  = {eval_nmi_post.avg:.4f}, Fore-Dice = {eval_dice_post.avg:.4f}")
    if eval_epe.count > 0:
        print(f"Flow Quality Metrics - EPE = {eval_epe.avg:.4f}")
    print(f"Flow Quality Metrics - Negative Jacobian Ratio (Folding) = {eval_det.avg:.6%}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
