import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import nibabel as nib
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T

class GaussianNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        return image + (torch.randn(image.size()) * self.std) + self.mean
        

class ADNI(Dataset):
    def __init__(self, root,
                 do_resize=False,
                 do_augmentation=False,
                 image_size=224):
        
        self.root = root
        self.files_dir = [root + file_name for file_name in os.listdir(root)]
        self.labels = {"CN": 0, "MC": 1, "AD": 2}
        
        self.scaler = MinMaxScaler()
        self.do_resize = do_resize
        
        if type(image_size) == int:
            self.resize = T.Resize((image_size, image_size))
        else:
            self.resize = T.Resize((image_size[0], image_size[1]))
        
        self.do_augmentation = do_augmentation
        self.gaussian_blur = T.GaussianBlur((3, 3), (0.1, 5))
        self.gaussian_noise = GaussianNoise(0, 0.001)
        self.augmentation = T.RandomChoice([self.gaussian_blur, self.gaussian_noise])
        
    def __len__(self):
        return len(self.files_dir)
    
    def open_scan_sagittal(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[11: 71, :, :]
        # For each scan channel
        for i in range(scan.shape[0]):
            # Normalize intensity between 0 and 1
            scan[i, :, :] = self.scaler.fit_transform(scan[i, :, :])
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[i, :, :], ((0, 0), (8, 8)))

        return image
    
    def open_scan_coronal(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[:, 11: 71, :]
        # For each scan channel
        for i in range(scan.shape[1]):
            # Normalize intensity between 0 and 1
            scan[:, i, :] = self.scaler.fit_transform(scan[:, i, :])
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = cv2.resize(scan[:, i, :], (95, 95))
            
        return image
    
    def open_scan_axial(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[:, :, 11: 71]
        # For each scan channel
        for i in range(scan.shape[2]):
            # Normalize intensity between 0 and 1
            scan[:, :, i] = self.scaler.fit_transform(scan[:, :, i])
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[:, :, i], ((8, 8), (0, 0)))
            
        return image
    
    def make_grid(self, image):
        # Put "image" channels in a grid-like image
        image_grid = np.zeros((10 * 95, 6 * 95))
        for row in range(10):
            for col in range(6):
                idx = row * 6 + col
                if idx < 60:
                    image_grid[row * 95: (row * 95) + 95, col * 95: (col * 95) + 95] = image[:, :, idx]
        
        return image_grid
        
    def __getitem__(self, index):
        # Open scan at index along sagittal axis
        ch1 = self.open_scan_sagittal(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch1 = self.make_grid(ch1)
        image_grid_ch1 = cv2.rotate(image_grid_ch1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Open scan at index along coronal axis
        ch2 = self.open_scan_coronal(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch2 = self.make_grid(ch2)
        image_grid_ch2 = cv2.rotate(image_grid_ch2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Open scan at index along axial axis
        ch3 = self.open_scan_axial(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch3 = self.make_grid(ch3)
        image_grid_ch3 = cv2.rotate(image_grid_ch3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        
        # Make an image with 3 channels
        image_3ch = np.stack((image_grid_ch1, image_grid_ch2, image_grid_ch3), axis = 2)
        # Convert to torch tensor and permute dimensions (C, H, W)
        image_3ch = torch.tensor(image_3ch).permute(2, 0, 1)
        # Resize image if needed
        if self.do_resize:
            image_3ch = self.resize(image_3ch)
        # Data augmentation
        if self.do_augmentation:
            image_3ch = self.augmentation(image_3ch)
        
        # Get the label and convert to torch tensor
        label = self.labels[os.path.split(self.files_dir[index])[1][: 2]]
        label = torch.tensor(label)
        
        return image_3ch.float(), label