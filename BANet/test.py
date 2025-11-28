import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
from uitils import *
import os.path as osp
import os
from models import Semantic
from visual_label import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(3)
def print_labels(labels):
    output = ", ".join(labels)
    print(output)

# /data1/cc/data/Algorithm-LACT/MSRS
#/data1/cc/data/wg/MSRS-wg/Cddfuse
#/mnt/disk/ch/data/MSRS_test/label
#/mnt/disk/ch/data/MSRS-main/test/Segmentation_labels
parser = argparse.ArgumentParser()
parser.add_argument("--Fusion_image", default="/mnt/disk/ch/data/Algorithm-LACT/MSRS361/TarDAL_MSRS/", type=str)
parser.add_argument("--label", default="/mnt/disk/ch/data/MSRS-main/test/Segmentation_labels/", type=str)
# parser.add_argument("--visible_dataroot", default="/data/infrared/cc/data/RoadScene_test/vi/", type=str)
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="/mnt/disk/ch/pytorch/mmsegmentation/projects/BANet/MSRS_Sem_result/STFNet_MSRS/", type=str)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
args = parser.parse_args()
patch_size = 4
label1 = ["Unlabelled", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump", "mIoU"]
if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)
    modelpth = './checkpoints/'
    n_classes = 9
    labels = ["Unlabelled", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump"]
    net = Semantic.BiSeNet(n_classes=n_classes)
    save_pth = osp.join(modelpth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    net.cuda()
    net.eval()
    score_thres = 0.7
    ignore_idx = 255
    lb_ignore = [255]
    transform = transforms.Compose([transforms.ToTensor()])
    dirname_vi = os.listdir(opt.Fusion_image)
    tmp_len = len(dirname_vi)
    cf = np.zeros((n_classes, n_classes))
    seg_metric = SegmentationMetric(n_classes, device=device)
 
    all_IoU_lists = []
    print('image number:', tmp_len)
    with torch.no_grad():
       
        for i in range(tmp_len):
    
            visible = Image.open(os.path.join(opt.Fusion_image, dirname_vi[i]))
            visible = transform(visible)
            visible = visible.to(device)
            visible = visible.unsqueeze(0)
            _, c ,_ ,_ = visible.shape
            if c == 1:
                visible = visible.repeat(1, 3, 1, 1)
            fused_img, _ = net(visible)
            
            fused_img = fused_img.argmax(1) 
            # fused_img = fused_img.cpu().numpy()   #####visual
            # visualize(os.path.join(opt.output_root, str(dirname_vi[i])), fused_img) # visual

            # print('sem:', dirname_vi[i])
   #计算指标结果
            fused_img = fused_img.squeeze(0)  #  
            labels = np.array(Image.open(os.path.join(opt.label, dirname_vi[i])))# np计算
            labels = torch.tensor(labels).to(device) #tensor
            # fused_img = np.dot(fused_img[..., :3], [0.2989, 0.5870, 0.1140])
            for gtcid in range(n_classes): 
              for pcid in range(n_classes):
                gt_mask      = labels == gtcid 
                pred_mask    = fused_img == pcid
                intersection = gt_mask * pred_mask
                cf[gtcid, pcid] += int(intersection.sum())
            seg_metric.addBatch(fused_img, labels, lb_ignore)
         
            IoU =seg_metric.IntersectionOverUnion()
            IoU = [np.array(a.item()) for a in IoU]
            mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
            Acc = np.array(seg_metric.pixelAccuracy().item())
            IoU_list = IoU
            IoU_list.append(mIoU)
            IoU_list = [np.round(100 * i, 2) for i in IoU_list]
            all_IoU_lists.append(IoU_list)
            # print('IoU:', IoU_list, 'ACC: {:.4f}'.format(Acc))

    all_IoU_arrays = np.array(all_IoU_lists)
    column_means = np.nanmean(all_IoU_arrays, axis=0)
    # last_dir = os.path.basename(os.path.normpath(opt.output_root))
    last_dir = os.path.basename(os.path.normpath(opt.Fusion_image))
    print(last_dir)
    print_labels(label1)
    # print('IoU:', column_means, 'ACC: {:.4f}'.format(Acc))
    # print('IoU:', ", ".join(f"{x:.2f}" for x in column_means) + ',', 'ACC: {:.4f},'.format(Acc))
    print('Results:', ", ".join(f"{x:.3f}" for x in column_means) + ',', 'ACC: {:.4f},'.format(Acc))
    overall_acc, acc, IoU = calculate_result(cf)
    print('| overall accuracy:', overall_acc)
    print('| accuracy of each class:', acc)
    print('| class accuracy avg:', acc.mean())
    print('| IoU:', IoU)
    print('| class IoU avg:', IoU.mean())

    
# 创建一个新的 Excel 工作簿
#     workbook = openpyxl.Workbook()

# # 获取默认的工作表
#     sheet = workbook.active

# # 将数据写入第一列
#     sheet['A1'] = 'Overall Accuracy'
#     sheet['A2'] = overall_acc
#     sheet['A4'] = 'Accuracy'
#     for i, accuracy in enumerate(acc, start=5):
#       sheet[f'A{i}'] = accuracy
#     sheet['A14'] = 'IoU'
#     for i, iou in enumerate(IoU, start=15):
#       sheet[f'A{i}'] = iou

# # 保存 Excel 文件
#     workbook.save('result.xlsx')

            
    print("work over!" )

