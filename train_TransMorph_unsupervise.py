from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# MaskedFlowL1 class is removed, using Unsupervised Losses instead.

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # Ensure logs are written immediately

    def flush(self):
        # self.terminal.flush()
        pass

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5' # let the bash script handle it
    
    batch_size = 64

    dir_pth = '/data2/xujr/idr_data/Train_CrossModal'

    # train_dir = os.path.join(dir_pth, 'Train')
    train_dir = '/data2/xujr/idr_data/Train_CrossModal_unsup_full'
    val_dir = os.path.join(dir_pth, 'Val')
    # train_dir = '/data2/xujr/idr_data/Train_CrossModal/Train'
    # val_dir = '/data2/xujr/idr_data/Train_CrossModal/Val'

    weights = [1.0, 1.0] # Weights for [NCC, Smoothness]
    print_freq = 10
    val_interval = 1 # Frequency to run validation and save model
    
    is_supervised = False
    
    save_dir = '/data2/xujr/output_model/0427/'
    if is_supervised:
        save_dir = save_dir + 'TransMorph_supervised_l1_smooth_{}_{}/'.format(weights[0], weights[1])
    else:
        save_dir = save_dir + 'TransMorph_unsupervised_ncc_{}_{}/'.format(weights[0], weights[1])

    if not os.path.exists(save_dir + 'experiments/'):
        os.makedirs(save_dir + 'experiments/')
    if not os.path.exists(save_dir + 'logs/'):
        os.makedirs(save_dir + 'logs/')
    sys.stdout = Logger(save_dir + 'logs/')
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 400 #max traning epoch
    cont_training = True # 改为 True，从崩溃前的模型断点续传

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    config.in_chans = 2 # Fixed image (1 channel) + Moving image (1 channel)
    model = TransMorph.TransMorph(config)
    
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs used")
        model = nn.DataParallel(model)

    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    # Not strictly needed for training logic since we use GT flow, but kept for potential logging
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize training
    '''
    # Use MultiModalityPairedDataset for cross-modal training as well.
    # It reads moving and fixed images.
    train_composed = None 
    
    train_set = datasets.MultiModalityPairedDataset(
        root_dir=train_dir,
        img_size=config.img_size,
        transforms=train_composed
    )
    val_set = datasets.MultiModalityPairedDataset(
        root_dir=val_dir,
        img_size=config.img_size,
        transforms=None
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=50, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    
    # 1. Image Similarity Loss (Unsupervised NCC)
    # Using window size 9 for 2D images
    criterion_sim = losses.NCC_vxm(win=[9, 9])
    
    # 2. Regularization Loss (Smoothness)
    # Penalize large gradients in the flow field
    criterion_reg = losses.Grad('l2', loss_mult=2)

    
    # Validation Metric
    # ssim 用于 validation 计算参考指标
    ssim_calc = SSIM(data_range=1.0, size_average=True, channel=1)
    
    def ncc_loss(I, J, mask=None, eps=1e-5):
        if mask is not None:
            # 只在 mask > 0 的区域计算相关性
            valid_pixels = mask.sum()
            if valid_pixels == 0:
                return torch.tensor(0.0, device=I.device)
            # 计算 mask 区域的均值
            I_mean = (I * mask).sum() / valid_pixels
            J_mean = (J * mask).sum() / valid_pixels
            
            # 中心化（仅对 mask 区域内的像素）
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
    
    best_epe = 1e10 # Initialize with a large number for EPE (lower is better)
    top_best_models = [] # List to track the top 5 BEST models based on EPE

    '''
    Resume from snapshot
    '''
    if cont_training:
        checkpoint_path = save_dir + 'experiments/latest_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            print("Loading latest info from: {}".format(checkpoint_path))
            # weights_only=False is required for PyTorch >= 2.6 when loading complex objects (like numpy scalars)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']
            if 'best_epe' in checkpoint:
                best_epe = checkpoint['best_epe']
            if 'top_best_models' in checkpoint:
                top_best_models = checkpoint['top_best_models']
            print("Resuming from Epoch {}".format(epoch_start))
        else:
            print("Checkpoint not found at {}, starting from scratch".format(checkpoint_path))
            
    # writer = SummaryWriter(log_dir=save_dir + 'logs/')
    
    for epoch in range(epoch_start, max_epoch):
        # Update learning rate
        # adjust_learning_rate(optimizer, epoch, max_epoch, lr) # Optional: disable if constant LR is preferred
        
        print('Epoch {} starts'.format(epoch))
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        loss_sim_meter = utils.AverageMeter()
        loss_reg_meter = utils.AverageMeter()
        
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            
            # Unpack data: [moving, fixed, gt_flow(optional), valid_mask(optional)]
            x = data[0] # Moving
            y = data[1] # Fixed
            
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            
            # Output: [warped_image, flow]
            pred_img = output[0]
            pred_flow = output[1]
            
            # 1. Image Similarity Loss (NCC)
            loss_sim = criterion_sim(y, pred_img) * weights[0]
            
            # 2. Smoothness Regularization
            loss_reg = criterion_reg(pred_flow, y) * weights[1]
            
            loss = loss_sim + loss_reg
            
            loss_sim_meter.update(loss_sim.item(), x.size(0))
            loss_reg_meter.update(loss_reg.item(), x.size(0))

            loss_all.update(loss.item(), x.size(0))
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % print_freq == 0:
                print('Iter {} of {} Loss: {:.6f}, NCC_Loss: {:.6f}, Reg: {:.6f}'.format(
                    idx, len(train_loader), loss.item(), loss_sim.item(), loss_reg.item()))

        # writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} Loss: {:.6f}'.format(epoch, loss_all.avg))
        
        '''
        Validation
        '''
        if (epoch + 1) % val_interval == 0:
            eval_epe = utils.AverageMeter()
            eval_ssim = utils.AverageMeter()
            eval_ncc = utils.AverageMeter()
            
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    data = [t.cuda() for t in data]
                    
                    x = data[0] # Moving
                    y = data[1] # Fixed
                    
                    x_in = torch.cat((x, y), dim=1)
                    output = model(x_in)
                    
                    pred_img = output[0]
                    pred_flow = output[1]
                    
                    if len(data) >= 4:
                        valid_mask = data[3]
                    else:
                        valid_mask = (y > 1e-4).float()
                        
                    # 1. EPE (End Point Error) with Mask (If GT flow is provided in val set)
                    if len(data) >= 3:
                        gt_flow = data[2] # GT Flow
                        diff = pred_flow - gt_flow
                        epe = torch.norm(diff, p=2, dim=1, keepdim=True)
                        masked_epe = epe * valid_mask
                        mean_epe = masked_epe.sum() / (valid_mask.sum() + 1e-8)
                        eval_epe.update(mean_epe.item(), x.size(0))
                    
                    # 2. SSIM (May be invalid for strong cross-modal, but keeping as reference)
                    val_ssim = ssim_calc(pred_img, y)
                    eval_ssim.update(val_ssim.item(), x.size(0))
                    
                    # 3. ZNCC (Robust to appearance changes, with mask)
                    val_ncc = ncc_loss(pred_img, y, valid_mask)
                    eval_ncc.update(val_ncc.item(), x.size(0))
            
            # Use negative ZNCC if EPE is not available to determine best model (lower is better)
            if eval_epe.count > 0:
                current_epe = eval_epe.avg
                print_str = 'Epoch {} Validation: Mean EPE: {:.4f}, Mean SSIM: {:.4f}, Mean ZNCC: {:.4f}'.format(
                    epoch, eval_epe.avg, eval_ssim.avg, eval_ncc.avg)
            else:
                current_epe = -eval_ncc.avg # Lower is better for top 5 queue
                print_str = 'Epoch {} Validation: Mean SSIM: {:.4f}, Mean ZNCC: {:.4f}'.format(
                    epoch, eval_ssim.avg, eval_ncc.avg)
                
            print(print_str)
            
            # --- 核心逻辑 1：维护 Top 5 的 Best 模型 ---
            # 如果列表还没满 5 个，或者当前的 EPE 比列表中最差（最大的）还要好，就加入保留名录
            if len(top_best_models) < 5 or current_epe < max([m['epe'] for m in top_best_models]):
                best_filename = f'model_best_epe_{current_epe:.4f}_epoch_{epoch+1}.pth'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_epe': best_epe if current_epe > best_epe else current_epe,
                    'top_best_models': top_best_models,
                    'optimizer': optimizer.state_dict(),
                }, save_dir=save_dir + 'experiments/', filename=best_filename)
                
                # 登记新的最好模型
                top_best_models.append({'epe': current_epe, 'epoch': epoch + 1, 'path': best_filename})
                # 按照 epe 从小到大排序 (最好的在最前面)
                top_best_models = sorted(top_best_models, key=lambda x: x['epe'])
                
                # 如果超过 5 个，把排名垫底（最大）的踢出列表，并把硬盘里的实体文件删掉
                if len(top_best_models) > 5:
                    worst_model = top_best_models.pop(-1)
                    worst_file_path = os.path.join(save_dir, 'experiments', worst_model['path'])
                    if os.path.exists(worst_file_path):
                        os.remove(worst_file_path)
                
                print(f"--> Saved to Top 5 best models (EPE: {current_epe:.4f})")
                
                # 顺便更新绝对最优纪录 并单独留存一份固定的 model_best.pth 方便下推理复用
                if top_best_models[0]['epoch'] == epoch + 1:
                    best_epe = current_epe
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_epe': best_epe,
                        'optimizer': optimizer.state_dict(),
                    }, save_dir=save_dir + 'experiments/', filename='model_best.pth')
                    print(f"--> New absolute BEST model updated with EPE: {best_epe:.4f}")
            
            # --- 核心逻辑 2：保存用于断点续传的 Latest ---
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_epe': best_epe,
                'top_best_models': top_best_models,
                'optimizer': optimizer.state_dict(),
            }, save_dir=save_dir + 'experiments/', filename='latest_checkpoint.pth')
            
            # --- 核心逻辑 3：每 20 个 Epoch 定期冷备份一份，不会被自动删除 ---
            if (epoch + 1) % 20 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_epe': best_epe,
                    'optimizer': optimizer.state_dict(),
                }, save_dir=save_dir + 'experiments/', filename=f'checkpoint_epoch_{epoch + 1}.pth')
            
def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar'):
    torch.save(state, save_dir + filename)

if __name__ == '__main__':
    main()
