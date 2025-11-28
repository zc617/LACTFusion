import argparse
import torchvision.transforms as transforms
from models import fusion_model
from PIL import Image
import time
from uitils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)


parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="/mnt/disk/ch/data/MSRS_over/ir/", type=str)
parser.add_argument("--visible_dataroot", default="/mnt/disk/ch/data/MSRS_over/vi/", type=str)
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="./Ours_LACT_4_over_MSRS", type=str)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
args = parser.parse_args()
windows_size = 8

def model(args):
    model = fusion_model.LACT_Fusion(upscale=2, img_size=128, embed_dim=60,
                   window_size=8, img_range=1.,num_heads=[6, 6, 6, 6],
                   mlp_ratio=2)

    return model


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)

    net = model(args).to(device)
    net.load_state_dict(torch.load("./checkpoints/fusion_LACT_4_1_sigm_16_100.pth"))
    net.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    dirname_ir = os.listdir(opt.infrared_dataroot)
    dirname_vi = os.listdir(opt.visible_dataroot)
    tmp_len = len(dirname_ir)
    # if tmp_len >= 50 :
    #     tmp_len = 50
    with torch.no_grad():
        t = []
        for i in range(tmp_len):
            index = i
            if i != 0:
                start = time.time()
          
            infrared  = Image.open(os.path.join(opt.infrared_dataroot, dirname_ir[i])).convert('L')
            infrared = transform(infrared).unsqueeze(0).to(device)
            visible = Image.open(os.path.join(opt.visible_dataroot, dirname_vi[i]))
            visible = transform(visible)
            visible = visible.unsqueeze(0)
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(visible)
            vis_y_image = vis_y_image.to(device)
            vis_cb_image = vis_cb_image.to(device)
            vis_cr_image = vis_cr_image.to(device)   # show color
            # visible = visible.squeeze(0)
            _, _, h_old, w_old = visible.size()
            h_pad = (h_old // windows_size + 1) * windows_size - h_old
            w_pad = (w_old // windows_size + 1) * windows_size - w_old
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [2])], 2)[:, :, :h_old + h_pad, :]
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [3])], 3)[:, :, :, :w_old + w_pad]
            infrared = torch.cat([infrared, torch.flip(infrared, [2])], 2)[:, :, :h_old + h_pad, :]
            infrared = torch.cat([infrared, torch.flip(infrared, [3])], 3)[:, :, :, :w_old + w_pad] 
            fused_img  = net(infrared,vis_y_image)
            fused_img = fused_img[..., :h_old * args.scale, :w_old * args.scale]
            if i != 0:
                end = time.time()
                print('consume time:', end - start)
                t.append(end - start)
            fused_img = clamp(fused_img)
            # x = torch.squeeze(fused_img, 1)
            # vis_cb_image = torch.squeeze(vis_cb_image, 1)
            # vis_cr_image = torch.squeeze(vis_cr_image, 1)
            fused_img = YCrCb2RGB(fused_img, vis_cb_image, vis_cr_image)  # show color
            fused_img = fused_img.squeeze(0)
            fused_img = transforms.ToPILImage()(fused_img)
            fused_img.save(os.path.join(opt.output_root, str(dirname_ir[i])))  # show color
            # save_image(fused_img, os.path.join(opt.output_root, str(dirname_ir[i])))
        print("mean:%s, std: %s" % (np.mean(t), np.std(t)))

    # from thop import profile, clever_format
    # input_data = torch.randn(1, 1, 640, 512)
    # flops, params = profile(net, (input_data.to(device), input_data.to(device)))
    # flops, params = clever_format([flops, params], "%.3f")
    # total_params = sum(p.numel() for p in net.parameters())
    # print('total parameters:', total_params)
    # print(f"FLOPS: {flops}", f"Params :{params}")

    # from fvcore.nn import FlopCountAnalysis, parameter_count
    # import torch

    # # 输入：构造与你模型输入一致的 dummy 数据
    # image_vis = torch.randn(1, 1, 640, 512).cuda()
    # image_ir  = torch.randn(1, 1, 640, 512).cuda()

    # # 模型本体

    # # 计算 FLOPs 和参数
    # flops = FlopCountAnalysis(net, (image_vis, image_ir))
    # params = parameter_count(net)

    # print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")
    # print(f"Params: {params[''] / 1e6:.2f} M")     