import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from pytorch_msssim import ssim, SSIM

# Update sys path to include modules if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import utils
from data import datasets, trans
import losses

def ncc_loss(I, J, mask=None, eps=1e-5):
    if mask is not None:
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return torch.tensor(0.0, device=I.device)
        I_mean = (I * mask).sum() / valid_pixels
        J_mean = (J * mask).sum() / valid_pixels
        I_centered = (I - I_mean) * mask
        J_centered = (J - J_mean) * mask
    else:
        I_mean = I.mean()
        J_mean = J.mean()
        I_centered = I - I_mean
        J_centered = J - J_mean
        
    cross = (I_centered * J_centered).sum()
    I_var = (I_centered ** 2).sum()
    J_var = (J_centered ** 2).sum()
    return cross / (torch.sqrt(I_var * J_var) + eps)

def main():
    # Configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    batch_size = 1
    # Check these paths - matching the training script
    test_dir = '/data2/xujr/idr_data/Train_Supervised_SynthMorph_no_appearance/Test' 
    model_dir = "/data2/xujr/output_model/0320/TransMorph_supervised_l1_smooth_1_0.05"

    load_mode = 'best' 
    
    # experiment_dir = "/data2/xujr/output_model/TransMorph_supervised_ssim_1_0.002/experiments/"
    # output_dir = "/data2/xujr/output_model/TransMorph_supervised_ssim_1_0.002/inference_results/"
    experiment_dir =  os.path.join(model_dir, 'experiments')
    output_dir = os.path.join(model_dir, 'inference_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    config.in_chans = 2 # Fixed + Moving
    model = TransMorph.TransMorph(config)

    # Load Weights
    if load_mode == 'best':
        checkpoint_path = os.path.join(experiment_dir, 'model_best.pth')
        if not os.path.exists(checkpoint_path):
             # Fallback to searching files if model_best doesn't exist yet
             print("model_best.pth not found, searching for best SSIM in filenames...")
             files = natsorted(glob.glob(os.path.join(experiment_dir, '*.pth')))
             checkpoint_path = files[-1] if files else None
    elif load_mode == 'latest':
        checkpoint_path = os.path.join(experiment_dir, 'latest_checkpoint.pth')
    else:
        # Load specific or last
        files = natsorted(glob.glob(os.path.join(experiment_dir, '*.pth')))
        checkpoint_path = files[-1] if files else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print('Loading model from: {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
        # if 'state_dict' in checkpoint:
        #     model.load_state_dict(checkpoint['state_dict'])
        # else:
        #     model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found at {}".format(checkpoint_path))
        return

    model.to(device)
    model.eval()

    '''
    Initialize Transformation & Dataset
    '''
    # Use 'bilinear' for image warping
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.to(device)

    # Dataset (Supervised)
    # Reusing the validation dataset loader from training
    test_set = datasets.MultiModalityPairedDataset(
        root_dir=test_dir,
        img_size=config.img_size,
        transforms=None # No random flips for inference
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    '''
    Evaluation Metrics Setup
    '''
    # SSIM calculator
    ssim_calc = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    
    # Metrics aggregators
    eval_ssim = utils.AverageMeter()
    eval_ncc = utils.AverageMeter() # Added NCC
    eval_epe = utils.AverageMeter() # End Point Error for Flow
    eval_det = utils.AverageMeter() # Jacobian Determinant <= 0
    
    # Baseline Metrics aggregators (Before Registration)
    eval_ssim_base = utils.AverageMeter()
    eval_ncc_base = utils.AverageMeter()

    print("Starting Inference on {} samples...".format(len(test_set)))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = [t.to(device) for t in data]
            # data structure from MultiModalityPairedDataset: [x, y, gt_flow]
            x = data[0] # Moving Image
            y = data[1] # Fixed Image (Target)
            gt_flow = data[2] # Ground Truth Flow [B, 2, H, W]

            if len(data) >= 4:
                valid_mask = data[3]
            else:
                valid_mask = (y > 1e-4).float()

            # Inference
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            
            # output[0] is deformed image, output[1] is predicted flow (DDF), output[2] is SVF
            pred_img = output[0]
            pred_flow = output[1]

            # 0. Baseline (Before Registration)
            # Calculate metrics between Moving (original) and Fixed Target
            base_ssim = ssim_calc(x, y)
            eval_ssim_base.update(base_ssim.item(), x.size(0))
            
            base_ncc = ncc_loss(x, y, valid_mask)
            eval_ncc_base.update(base_ncc.item(), x.size(0))

            # 1. SSIM
            # Calculate SSIM between Deformed Moving and Fixed Target
            val_ssim = ssim_calc(pred_img, y)
            eval_ssim.update(val_ssim.item(), x.size(0))

            # 1.5 ZNCC
            val_ncc = ncc_loss(pred_img, y, valid_mask)
            eval_ncc.update(val_ncc.item(), x.size(0))

            # 2. End Point Error (EPE) for Flow

            diff = pred_flow - gt_flow
            epe = torch.norm(diff, p=2, dim=1, keepdim=True)
            masked_epe = epe * valid_mask
            mean_epe = masked_epe.sum() / (valid_mask.sum() + 1e-8)
            eval_epe.update(mean_epe.item(), x.size(0))

            # 3. Jacobian Determinant (Folding)
            flow_np = pred_flow.detach().cpu().numpy()[0] # [2, H, W]
            
            # Transpose to [H, W, 2] for jacobian_determinant_vxm
            flow_np = flow_np.transpose(1, 2, 0)
            
            # Calculate Jac Det
            jac_det = utils.jacobian_determinant_vxm(flow_np)
            
            # Ratio of folding pixels (Only consider folding within valid mask to match physical reality)
            mask_np = valid_mask.detach().cpu().numpy()[0, 0] # [H, W]
            valid_pixels = np.sum(mask_np > 0)
            
            # Check folding only on regions where mask is valid
            folding_ratio = np.sum((jac_det <= 0) & (mask_np > 0)) / (valid_pixels + 1e-8)
            eval_det.update(folding_ratio, x.size(0))

            # Print progress
            if i % 10 == 0:
                print('Sample {}/{}: SSIM: {:.4f}, ZNCC: {:.4f}, EPE: {:.4f}, Folding: {:.4%}'.format(
                    i, len(test_loader), val_ssim.item(), val_ncc.item(), mean_epe.item(), folding_ratio))

            # Optional: Save visual results for the first few samples
            if i < 5: 
                save_visual_result(x, y, pred_img, pred_flow, gt_flow, i, output_dir)
                print(f"Saved visual result for sample {i} to {output_dir}")
                print(f"Sample {i} - SSIM: {val_ssim.item():.4f}, ZNCC: {val_ncc.item():.4f}, EPE: {mean_epe.item():.4f}, Folding Ratio: {folding_ratio:.4%}")


    print('=' * 40)
    print('Baseline Summary (Before Registration):')
    print('Mean Initial SSIM: {:.4f} +- {:.4f}'.format(eval_ssim_base.avg, eval_ssim_base.std))
    print('Mean Initial ZNCC: {:.4f} +- {:.4f}'.format(eval_ncc_base.avg, eval_ncc_base.std))
    print('-' * 40)
    print('Final Summary (After Registration):')
    print('Mean SSIM: {:.4f} +- {:.4f}'.format(eval_ssim.avg, eval_ssim.std))
    print('Mean ZNCC: {:.4f} +- {:.4f}'.format(eval_ncc.avg, eval_ncc.std))
    print('Mean EPE:  {:.4f} +- {:.4f}'.format(eval_epe.avg, eval_epe.std))
    print('Mean Folding Ratio: {:.4%} +- {:.4%}'.format(eval_det.avg, eval_det.std))
    print('=' * 40)

def save_visual_result(moving, fixed, moved, pred_flow, gt_flow, idx, save_dir):
    """
    Helper to save visualization of registration
    """
    # Convert to numpy and normalize for plotting
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()[0, 0, :, :]

    m = to_numpy(moving)
    f = to_numpy(fixed)
    md = to_numpy(moved)
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(m, cmap='gray')
    axes[0].set_title('Moving')
    axes[0].axis('off')

    axes[1].imshow(f, cmap='gray')
    axes[1].set_title('Fixed')
    axes[1].axis('off')

    axes[2].imshow(md, cmap='gray')
    axes[2].set_title('Moved (Pred)')
    axes[2].axis('off')
    
    axes[3].imshow(np.abs(f - m), cmap='gray')
    axes[3].set_title('Diff: Fixed - Moving')
    axes[3].axis('off')

    axes[4].imshow(np.abs(f - md), cmap='gray')
    axes[4].set_title('Diff: Fixed - Moved')
    axes[4].axis('off')
    
    # Optional: Plot Flow (magnitude) if needed
    # ...

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'result_{}.png'.format(idx)))
    plt.close()

if __name__ == '__main__':
    main()
