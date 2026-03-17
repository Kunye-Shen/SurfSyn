import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
    

class DefectDataset(Dataset):
    def __init__(self, data_path, transform, mode='train', dataset=None):
        if mode == 'train':
            self.img_list = glob(data_path + f"{mode}/Img/*")
            self.gt_list = glob(data_path + f"{mode}/GT/*")
        else:
            self.img_list = glob(data_path + f"{dataset}/Img/*")
            self.gt_list = glob(data_path + f"{dataset}/GT/*")
        self.img_list.sort()
        self.gt_list.sort()
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_dir = self.img_list[index]
        gt_dir = self.gt_list[index]

        img = np.array(Image.open(img_dir).convert('RGB')).astype(np.float32)

        gt = np.array(Image.open(gt_dir).convert('L')).astype(np.float32)

        augmented = self.transform(image=img, mask=gt)
        img_aug, gt_aug = augmented['image'], augmented['mask'] / 255.0

        if self.mode=='train':
            return {'img': img_aug, 'gt': gt_aug}
        else:
            return {'img': img_aug, 'img_dir': img_dir}