import argparse
import torch
from models import fusion_model
from input_data import ImageDataset
from uitils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import random
import logging
import os.path as osp
from loss_ssim import *
from lr_scheduler import build_scheduler



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="./TNO_Train_crop/ir/", type=str)
parser.add_argument("--visible_dataroot", default="./TNO_Train_crop/vi/", type=str)
parser.add_argument("--data_root", default="./MSRS-main/train/", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--img_size", type=int, default=(128, 128)) 
parser.add_argument("--window_size", type=int, default=8) 
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument('--base_ir', type=float, default=1e-4)  # main 
parser.add_argument('--warmup_ir', type=float, default=1e-6) # init
parser.add_argument("--warmup_epoch", type=int, default=20) 
parser.add_argument("--mlp_ratio", type=int, default=2)
parser.add_argument('--min_ir', type=float, default=1e-7)
parser.add_argument('--optimizer_eps', type=float, default=1e-8)
parser.add_argument("--optimizer_beats", type=float, default=(0.9, 0.99))
parser.add_argument('--weight_decay', type=float, default=0.05)
#parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--lr_sch_name', type=str, default='cosine') 
parser.add_argument("--min_lr", type=float, default=1e-5)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
parser.add_argument('--loss_weight', default='[1, 10, 0.5]', type=str,metavar='N', help='loss weight')
   #  2 10 3


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    
    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    writer = SummaryWriter('./runs/logdir')
    net = fusion_model.build_model(opt).to(device)
    start_epoch = 0

    optimizer = torch.optim.AdamW(net.parameters(), eps=opt.optimizer_eps, betas=opt.optimizer_beats,
                                lr=opt.base_ir, weight_decay=opt.weight_decay)
    
    train_datasets = ImageDataset(opt.data_root, opt.img_size)
    lens = len(train_datasets)
    print('data lens', lens)
    lr_scheduler = build_scheduler(opt, optimizer, lens)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size, shuffle=True)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    global_step = 0
    w1_vis = 1
    i = 0
    grad_loss = 0.0
    inti_loss = 0.0
    ssim_loss = 0.0
    t1, t2, t3 = eval(opt.loss_weight)

    for epoch in range(opt.epoch):
     
        net.train()
        num=0
        for index, data in enumerate(dataloader):
            nc, c, h, w = data[0].size()
            nc2, c2, h2, w2 = data[1].size()
            infrared = data[0].to(device)
            visible = data[1].to(device)
            fused_img = net(infrared, visible)
            
            fused_img = clamp(fused_img)
            int_loss = Int_Loss(fused_img, visible, infrared, w1_vis).to(device)
            gradient_loss = gradinet_Loss(fused_img, visible, infrared).to(device)
            ssim_loss = 1- SSIM_Loss(fused_img, visible, infrared).to(device)
            loss = t1 * int_loss + t2 * gradient_loss + t3 * ssim_loss
            runloss += loss.item()
            grad_loss += gradient_loss.item()
            inti_loss += int_loss.item()
            ssim_loss += ssim_loss.item()
            if epoch == 0 and index == 0:
                writer.add_graph(net, (infrared, visible))
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step_update(epoch * lens + index)  
            if index % 200 == 0:  #
                writer.add_scalar('training loss', runloss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('int loss', inti_loss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('gradient loss', grad_loss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('SSIM loss', ssim_loss / 200, epoch * len(dataloader) + index)
                runloss = 0.
                inti_loss = grad_loss = ssim_loss = 0.
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('learning_rate', current_lr, epoch)
        if epoch % 1 == 0:
            print('write_data, epoch=', epoch)
            print(
                'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, SSIM loss is {:.5}, total loss is  {:.5}'.
                format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
                       gradient_loss.item(), ssim_loss.item(), loss.item()))  
            writer.add_images('IR_images', infrared, dataformats='NCHW')
            writer.add_images('VIS_images', visible, dataformats='NCHW')
            writer.add_images('Fusion_images', fused_img, dataformats='NCHW')
        print(f"Epoch [{epoch+1}/{opt.epoch}], Learning Rate: {current_lr}")
    writer.close()
  
    torch.save(net.state_dict(), './checkpoints/fusion_LACT_4_1_sigm_16_'+str(epoch+1)+'.pth')
    print('training is complete!')



 
 
 

