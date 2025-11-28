import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import os
# import chainer
from ipdb import set_trace as st
from torch.autograd import Variable
import openpyxl
import torch
import matplotlib.pyplot as plt

def imread(path):
    label = np.array(Image.open(path))
    return label


def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc


# def calculate_result(cf):
#     n_class = cf.shape[0]
#     conf = np.zeros((n_class,n_class))
#     IoU = np.zeros(n_class)
#     conf[:,0] = cf[:,0]/cf[:,0].sum()
#     for cid in range(1,n_class):
#         if cf[:,cid].sum() > 0:
#             conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
#             IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
#     overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
#     acc = np.diag(conf)

#     return overall_acc, acc, IoU

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class, n_class))
    IoU = np.zeros(n_class)
    conf[:, 0] = cf[:, 0] / cf[:, 0].sum()
    non_zero_IoU = []
    for cid in range(1, n_class):
        if cf[:, cid].sum() > 0:
            conf[:, cid] = cf[:, cid] / cf[:, cid].sum()
            intersection = cf[cid, cid]
            union = cf[cid, 1:].sum() + cf[1:, cid].sum() - intersection
            if union != 0:
                IoU[cid] = intersection / union
                non_zero_IoU.append(IoU[cid])
    overall_acc = np.diag(cf[1:, 1:]).sum() / cf[1:, :].sum()
    acc = np.diag(conf)
    acc = acc[acc >= 0.1]
    IoU = IoU[IoU >= 0.1]
    overall_acc = overall_acc[overall_acc >= 0.1]
    return overall_acc, acc, IoU


def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    # background       = [255,255,255]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(names, predictions):
    palette = get_palette()
    img = np.zeros((predictions[0].shape[0], predictions[0].shape[1], 3), dtype=np.uint8)
    for pred in predictions:
        for cid in range(1, 9):  # 遍历所有类别
            img[pred == cid] = palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(names)

def create_palette_image(palette):

    labels = ["Unlabelled", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump"]
    palette = get_palette()
    fig, ax = plt.subplots(1, len(palette), figsize=(19, 0.8))  # 每个小方块一个子图

    for i, (color, label) in enumerate(zip(palette, labels)):
        ax[i].imshow(np.ones((60, 300, 3), dtype=np.uint8) * color)  # 填充颜色
        ax[i].axis('off')
        ax[i].text(150, 30, label, color='white', ha='center', va='center', fontsize=20, fontweight='bold')  # 显示标签，字体较大且粗体

    plt.tight_layout()  # 调整子图布局
    plt.subplots_adjust(top=1, bottom=0.03, left=0, right=1, hspace=0.02, wspace=0.00)
    plt.savefig("1.png") 

def visualize_label(save_name, label):
    palette = get_palette()
    pred = label
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(1, int(label.max()+1)):
        img[pred == cid] = palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(save_name)


if __name__ == '__main__':
  
    label_dir = r'./MSRS_test/label'
    save_dir = r'./Segmentation_visual' ## 可视化结果保存的文件夹

    os.makedirs(save_dir, exist_ok=True)
    n_class = 9
    # file_list = os.listdir(label_dir)
    # for item in file_list:
    #     file_path = os.path.join(label_dir, item)
    #     save_path = os.path.join(save_dir, item)
    #     label = imread(file_path)
    #     visualize_label(save_path, label) # 可视化label

    palette = get_palette()
    create_palette_image(palette) # 画图例