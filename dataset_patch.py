import os
from glob import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random

class CropPatchDataset(Dataset):
    def __init__(self, imgs_dir: str, masks_dir: str, is_train: bool = True, patch_size: int = 256):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.is_train = is_train
        self.patch_size = patch_size

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir) if not file.startswith('.')]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        
        mask_file = glob(os.path.join(self.masks_dir, idx + '.*'))[0]
        img_file = glob(os.path.join(self.imgs_dir, idx + '.*'))[0]
        
        mask = Image.open(mask_file).convert('L')
        img = Image.open(img_file).convert('RGB')

        # --- PHASE 1: TRAINING (Random 256x256 Patches + Augmentation) ---
        if self.is_train:
            # 1. Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.patch_size, self.patch_size))
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
            
            # 2. Geometric Augmentations
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)
                
            # 3. Environmental Lighting
            color_tf = transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3)
            img = color_tf(img)
            
            if random.random() > 0.7:
                img = transforms.functional.gaussian_blur(img, kernel_size=[3, 3])

        # --- PHASE 2: VALIDATION/TESTING (Full 480x640 Image) ---
        # If is_train=False, we skip cropping and just return the full image
        
        # Convert to Numpy
        img_nd = np.array(img).transpose((2, 0, 1))
        img_trans = (img_nd / 255.0).astype(np.float32)
        
        mask_nd = np.array(mask)
        mask_trans = np.where(mask_nd > 128, 1.0, 0.0).astype(np.float32)

        return {
            'image': torch.from_numpy(img_trans),
            'mask': torch.from_numpy(mask_trans).unsqueeze(0) # Add channel dim
        }