from torch.utils.data import Dataset
from torchvision import transforms
#from flowlib import read, read_weights_file
from PIL import Image
import re
from uitils import *
from augment_image import *
import torchvision
import torchvision.transforms.functional as TF
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# def trans_train_data(im, num, image_size):
#     transform = transforms.Compose([
#         transforms.CenterCrop(image_size),  # crop images
#         # transforms.RandomCrop(size=image_size),
#         # transforms.TenCrop(image_size),
#         transforms.ToTensor()]
#

class ImageDataset(Dataset):
    def __init__(self, dataroot, image_size):
        """

        """
        self.vis_folder = os.path.join(dataroot, 'vi')
        self.ir_folder = os.path.join(dataroot, 'ir')
        # self.label_folder = os.path.join(dataroot, 'Segmentation_labels')
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        self.image_size = image_size

        k = 0
        tmp_len = len(self.vis_list)
        print(self.vis_folder, tmp_len)
        # self.crop = transforms.CenterCrop(self.image_size)

        # self.crop = torchvision.transforms.RandomCrop(self.image_size)
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()])

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        # read image as type Tensor
        vis = Image.open(vis_path).convert('L')
        ir = Image.open(ir_path).convert('L')
        ir = self.transform(ir)
        vis = self.transform(vis)

        return ir, vis
    
    def __len__(self):
        return len(self.vis_list)
    

if __name__ == "__main__":
    date_root="/data/infrared/cc/data/MSRS-main/train/"
    image = ImageDataset(date_root,[256,256])
    print('data lens', len(image))
    dataloader = torch.utils.data.DataLoader(image, batch_size=1)
    for index, item in enumerate(dataloader):
        print('vi:', item[0].shape)
        print('ir:', item[1].shape)

 
    print('OK')





