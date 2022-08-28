import os
import numpy as np

import nibabel as nib
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset


class ADNI(Dataset):
    def __init__(self, root, transforms=None, extra_channel_dim=False, rotate=False):
        self.root = root
        self.files_dir = [root + file_name for file_name in os.listdir(root)]
        # "MC" for MCI
        self.labels = {"CN": 0, "MC": 1, "AD": 2}
        self.transforms = transforms
        self.extra_channel_dim = extra_channel_dim
        self.rotate = rotate
        
    def __len__(self):
        return len(self.files_dir)
    
    def normalize_voxel(self, image):
        # Find min and max along channels (voxel)
        vmax = image.max(dim=0, keepdim=True)[0]
        vmin = image.min(dim=0, keepdim=True)[0]
        
        
        # Repeat vmax and vmin to make shape (60x95x79)
        vmax = vmax.repeat([image.shape[0], 1, 1])
        vmin = vmin.repeat([image.shape[0], 1, 1])
        
        # Scaling
        image = (image - vmin) / (vmax - vmin)
        image = image.nan_to_num()
        
        return image
    
    def open_scan_axial(self, index):
        # Load the scan
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = nib.load(self.files_dir[index]).get_fdata()[:, :, 11: 71]
        # image = nib.load(self.files_dir[index]).get_fdata()
        
        if self.rotate:
            image = np.rot90(image)
        
        return image
    
    def __getitem__(self, index):
        # Open scan at index along axial axis (Depth, Height, Width)
        # "copy" is needed if "self.rotate=True"
        image = torch.tensor(self.open_scan_axial(index).copy()).permute(2, 0, 1)
        # image = torch.tensor(self.open_scan_axial(index).copy())
        
        if self.transforms:
            image=self.transforms(image)
        
        image = self.normalize_voxel(image)
        
        if self.extra_channel_dim:
            image = image.unsqueeze(0)
        
        # Get the label and convert to torch tensor
        label = self.labels[os.path.split(self.files_dir[index])[1][: 2]]
        label = torch.tensor(label)
        
        return image.float(), label


class ADNI3Channels(Dataset):
    def __init__(self, root, transforms=None, rotate=False, duplicate_channels=False):
        self.root = root
        self.files_dir = [root + file_name for file_name in os.listdir(root)]
        # "MC" for MCI
        self.labels = {"CN": 0, "MC": 1, "AD": 2}
        self.transforms = transforms
        self.rotate = rotate
        self.duplicate_channels = duplicate_channels
        
    def __len__(self):
        return len(self.files_dir)
    
    def normalize_voxel(self, image):
        # Find min and max along channels (voxel)
        vmax = image.max(dim=0, keepdim=True)[0]
        vmin = image.min(dim=0, keepdim=True)[0]
        
        
        # Repeat vmax and vmin to make shape (60x95x79)
        vmax = vmax.repeat([image.shape[0], 1, 1])
        vmin = vmin.repeat([image.shape[0], 1, 1])
        
        # Scaling
        image = (image - vmin) / (vmax - vmin)
        image = image.nan_to_num()
        
        return image
    
    def normalize_channel(self, image):
        for i in range(image.shape[0]):
            vmax = image[i, :, :].max()
            vmin = image[i, :, :].min()
            image[i, :, :] = (image[i, :, :] - vmin) / (vmax - vmin)
        
        return image
    
    
    def open_scan_sagittal(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[11: 71, :, :]
        # For each scan channel
        for i in range(scan.shape[0]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[i, :, :], ((0, 0), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        
        image = self.normalize_voxel(image)

        return image
    
    def open_scan_coronal(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[:, 11: 71, :]
        # For each scan channel
        for i in range(scan.shape[1]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = cv2.resize(scan[:, i, :], (95, 95))
            # Padding is a better option!
            # image[:, :, i] = np.pad(scan[:, i, :], ((8, 8), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        
        image = self.normalize_voxel(image)
            
        return image
    
    def open_scan_axial(self, index):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.files_dir[index]).get_fdata()[:, :, 11: 71]
        # For each scan channel
        for i in range(scan.shape[2]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[:, :, i], ((8, 8), (0, 0)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        
        image = self.normalize_voxel(image)
        
        return image

    def make_grid(self, image):
        # Put "image" channels in a grid-like image
        image_grid = torch.zeros((10 * 95, 6 * 95))
        for row in range(10):
            for col in range(6):
                idx = row * 6 + col
                if idx < 60:
                    image_grid[row * 95: (row * 95) + 95, col * 95: (col * 95) + 95] = image[idx, :, :]
        
        return image_grid
        
    def __getitem__(self, index):
        # Open scan at index along sagittal axis
        ch1 = self.open_scan_sagittal(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch1 = self.make_grid(ch1)
        
        
        # Open scan at index along coronal axis
        ch2 = self.open_scan_coronal(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch2 = self.make_grid(ch2)
        
        # Open scan at index along axial axis
        ch3 = self.open_scan_axial(index)
        # Put channels in a grid-like image and rotate
        image_grid_ch3 = self.make_grid(ch3)
        
        if self.duplicate_channels:
            image_3ch = torch.stack((image_grid_ch3, image_grid_ch3, image_grid_ch3), axis = 0)
        else:
            image_3ch = torch.stack((image_grid_ch1, image_grid_ch2, image_grid_ch3), axis = 0)
            
        if self.rotate:
            image_3ch = torch.rot90(image_3ch, 1, [1, 2])
        
        if self.transforms:
            image_3ch = self.transforms(image_3ch)
            image_3ch = self.normalize_channel(image_3ch)
        
        # Get the label and convert to torch tensor
        label = self.labels[os.path.split(self.files_dir[index])[1][: 2]]
        label = torch.tensor(label)
        
        return image_3ch.float(), label