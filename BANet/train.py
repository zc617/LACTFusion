import argparse
import torch
from models import fusion_model
from input_data import ImageDataset
from uitils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import logging
import os.path as osp
from loss_ssim import *
from models import Semantic
# from cityscapes import CityScapes


# tensorboard --logdir=runs --port=6017


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

parser = argparse.ArgumentParser()
# parser.add_argument("--infrared_dataroot", default="/data/infrared/cc/data/TNO_Train_crop/ir/", type=str)
# parser.add_argument("--visible_dataroot", default="/data/infrared/cc/data/TNO_Train_crop/vi/", type=str)
parser.add_argument("--data_root", default="/data/infrared/cc/data/MSRS-main/train/", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=(80, 80)) # 64 * 64 测试
parser.add_argument("--epoch", type=int, default= 1050)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
parser.add_argument('--loss_weight', default='[1, 10, 0.1, 0]', type=str,metavar='N', help='loss weight')
   #  2 10 3
# Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    x = opt.epoch - 50
    x = 0
    writer = SummaryWriter('./runs/logdir')

    net = fusion_model.FusionNet().to(device)

    # net = nn.DataParallel(net)
    # net =net.to(device)

    ############################################
    modelpth = './checkpoints/'
    n_classes = 9
    segmodel = Semantic.BiSeNet(n_classes=n_classes)
    save_pth = osp.join(modelpth, 'model_final.pth')
    # if logger == None:
    #     logger = logging.getLogger()
    #     setup_logger(modelpth)
    # segmodel.load_state_dict(torch.load(save_pth))
    segmodel.cuda()
    segmodel.eval()
    for p in segmodel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    score_thres = 0.7
    ignore_idx = 255
    n_min = 8 * opt.image_size[0] * opt.image_size[1] // 8
    criteria_p = Semantic_Loss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = Semantic_Loss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    #############################################

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    
    train_datasets = ImageDataset(opt.data_root, opt.image_size)
    lens = len(train_datasets)
    print('data lens', lens)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size,shuffle=True)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    global_step = 0
    w1_vis = 1
    i = 0
    n_min =  opt.image_size[0] * opt.image_size[1]
    Semantic = Semantic_Loss(thresh=0.7, n_min=n_min, ignore_lb=255.0)
    se_loss = 0.0
    t1, t2, t3, t4 = eval(opt.loss_weight)
    for epoch in range(opt.epoch):
        if epoch % 50 == 0 and epoch > 1 and epoch < x:
           opt.lr = opt.lr * 0.1
        if epoch == (x + 1) :
            opt.lr = 1e-6
        if (epoch - x) % 10 == 0 and epoch > x:
            opt.lr = opt.lr * 0.1
        
        net.train()
        num=0
        # train_tqdm = tqdm(dataloader, total=len(dataloader))  # 进度条
        for index, data in enumerate(dataloader):
            if len(data[0].shape) > 4:
                    data[0] = data[0].squeeze(1)
                    data[1] = data[1].squeeze(1)
                    data[2] = data[2].squeeze(1)
                    data[3] = data[3].squeeze(1)
            nc, c, h, w = data[0].size()
            nc2, c2, h2, w2 = data[1].size()
            infrared = data[0].to(device)
            visible = data[1].to(device)
            label = Variable(data[2]).to(device)
            color = data[3]
            color, Cb, Cr = RGB2YCrCb(color)
            color = color.to(device)
            Cb = Cb.to(device)
            Cr = Cr.to(device)
            fused_img, disp_ir_feature, disp_vis_feature = net(infrared, visible)

            fused_img = clamp(fused_img)
            int_loss = Int_Loss(fused_img, visible, infrared, w1_vis).to(device)
            gradient_loss = gradinet_Loss(fused_img, visible, infrared).to(device)
         
            fused_rgb = YCrCb2RGB(fused_img, Cb, Cr)
            fused_rgb = clamp(fused_rgb)
            fused_r = torch.squeeze(fused_rgb, 1)
            lab = torch.squeeze(label, 1)
            # if epoch > 0 :
            if epoch > x:     
                out, mid = segmodel(fused_rgb)
                lossp = criteria_p(out, lab)
                loss2 = criteria_16(mid, lab)
                se_loss = (lossp + 0.1 * loss2) * 0.5
                # t1 = 0.1
                # t2 = 2 
                # t4 = 0.01
            ssim_loss = SSIM_Loss(fused_img, visible, infrared).to(device)

            loss = t1 * int_loss + t2 * gradient_loss  + t3 * se_loss  + t4 * (1- ssim_loss)
            runloss += loss.item()
            if epoch == 0 and index == 0:
                writer.add_graph(net, (infrared, visible))
            # global_step += 1
            if index % 200 == 0:  #
                writer.add_scalar('training loss', runloss / 200, epoch * len(dataloader) + index)
                runloss = 0.

            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch % 1 == 0:
            print('write_data, epoch=', epoch)
            # print(
            #     'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, total loss is  {:.5}, lr: {}'.
            #     format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
            #            gradient_loss.item(), loss.item(), opt.lr))
            # print(
            #     'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, Semantic loss is {:.5} total loss is  {:.5}, lr: {}'.
            #     format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
            #            gradient_loss.item(), se_loss, loss.item(), opt.lr))
            print(
                'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, Semantic loss is {:.5}, SSIM loss is {:.5}, total loss is  {:.5}, lr: {}'.
                format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
                       gradient_loss.item(), se_loss, ssim_loss.item(), loss.item(), opt.lr))
            
            writer.add_images('IR_images', infrared, dataformats='NCHW')
            writer.add_images('VIS_images', visible, dataformats='NCHW')
            writer.add_images('Fusion_images', fused_img, dataformats='NCHW')
        if ((epoch % 20) == 0 and epoch >= 1) or ( epoch >= (x + 1)  and (epoch-x) % 10 == 0):
            torch.save(net.state_dict(), './checkpoints/fusion_wave_cnn_4_2_'+str(epoch+1)+'.pth'.format(opt.lr, log_file[2:]))
            # for name, m in net.named_modules():
            #     if isinstance(m, torch.nn.Conv2d):
            #         m.register_forward_pre_hook(hook_func)
            # net(infrared, visible)
            # m.register_forward_pre_hook(hook_func).remove()
            # corr_ir = torch.reshape(corr_ir, (-1, 1, 128, 128))
            # corr_vis = torch.reshape(corr_vis, (-1, 1, 128, 128))
            #
            # image_corr_ir = corr_ir.cpu().detach().numpy()
            # image_corr_vis = corr_vis.cpu().detach().numpy()

            # plt.figure(figsize=(6.4, 4.8))
            # disp_feature_image(infrared, 'img', 2, 1, 'IR_img')
            # disp_feature_image(disp_ir_feature, 'feature', 2, 2, 'IR_Feature')
            # # disp_feature_image(image_corr_ir, 'cor', 3, 3, 'IR_Corr')
            # disp_feature_image(visible, 'img', 2, 3, 'VIS_img')
            # disp_feature_image(disp_vis_feature, 'feature', 2, 4, 'VIS_Feature')

            # plt.show()
            # plt.close()
    writer.close()
    torch.save(net.state_dict(), './checkpoints/fusion_wave_cnn_4_2_'+str(epoch+1)+'.pth'.format(opt.lr, log_file[2:]))
    print('training is complete!')



    # ch_fusion_wave1 修改conv重建的激活函数和最后一层，修改系数为2:10:1，更换语义读取的代码
    # ch_fusion_wave2 修改了训练的轮数，800+50，语义的Lr调整为每10轮降低0.005（之前是0.01), 
    #                 修改了网络的channel 32,32,64,96  并加入了维度调整的1*1卷积+norm                
    # ch_fusion_wave3 在1的基础上修改，两个模态的特征合并后，加入残差。降低语义Lr，从1e-7起步,每10个降低0.005，patch :8,4
    #  在3的基础上patch改为4,2，维度128 进行训练 效果在1012时视觉效果最好。然后图像出现模糊,此时的超参，1,8,0.1。
    # 在3的基础上继续实验： 超参调为1,10,0.2 训练是每10个降低0.001  测试了指标比较低，趋势是语义训练时间长的效果好一点。

    # 改进：训练图压缩到80*80，patch 为4，stride=1 测试
    # 122 次训练的时候获得了很好的视觉效果。2个指标比较高。现在修改一下。
    # 修改了patchembed的操作。在122版本的基础上修改。
    # 132的时候指标比较高。继续可以改进一下。超参调回。另外发现loss在语义加入后，有波动的趋势。学习率太高效果不好
    # 超参为1,5,0.1 两层卷积，在142时获得了较好的效果。但是vif指标比较低。修改：加入语义后每2个epoch保存一次。
    # 加入更多的训练，目前是232时获得了最好的参数。
    # 调整了patchembed，加入了一个残差求和,效果不好，不稳定。另外语义loss如果一开始就加入训练，效果完全不行。
    # 调整网络结构，融合时加入了实部。效果不明显。
    # 加入ssim后，1,10,0.1,10 训练300epoch，目前在212时做到了psnr,vif,mi最高 保存一个版本，继续下一个实验。
    # 改进方法，调高梯度，降低ssim的比例。看看效果。 1,15,0.1,5 效果不好。
    # 修改一下，继续调高梯度，1,10,0.1,1 现在效果偏向可见光图像，红外的梯度损失严重
    # 二阶段修改一下将其他loss的权重降低10倍。

