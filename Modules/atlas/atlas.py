import numpy as np
import nibabel as nib
import cv2
import torch

class AAL():
    def __init__(self, aal_dir, labels_dir, rotate=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.rotate = rotate        
    
    def open_aal_axial(self):
        # Load the AAL
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        
        if self.rotate:
            image = np.rot90(image)
        
        return image
    
    def get_data(self):
        image = torch.tensor(self.open_aal_axial().copy()).permute(2, 0, 1)
            
        labels = {}
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                labels[key] = int(value)
        
        return image.float(), labels


class AAL3Channels():
    def __init__(self, aal_dir, labels_dir, transforms=None, rotate=False, duplicate_channels=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.rotate = rotate
        self.duplicate_channels = duplicate_channels
    
    
    def open_scan_sagittal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[11: 71, :, :]
        # For each scan channel
        for i in range(scan.shape[0]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[i, :, :], ((0, 0), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)

        return image
    
    def open_scan_coronal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, 11: 71, :]
        # For each scan channel
        for i in range(scan.shape[1]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            # WARNING: Use INTER_NEAREST to preserve atlas values!
            image[:, :, i] = cv2.resize(src=scan[:, i, :], dsize=(95, 95), interpolation=cv2.INTER_NEAREST)
            # image[:, :, i] = np.pad(scan[:, i, :], ((8, 8), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                    
        return image
    
    def open_scan_axial(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        # For each scan channel
        for i in range(scan.shape[2]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[:, :, i], ((8, 8), (0, 0)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                
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
        
    def get_data(self):
        # Open scan at index along sagittal axis
        ch1 = self.open_scan_sagittal()
        # Put channels in a grid-like image
        image_grid_ch1 = self.make_grid(ch1)
        
        # Open scan at index along coronal axis
        ch2 = self.open_scan_coronal()
        # Put channels in a grid-like image
        image_grid_ch2 = self.make_grid(ch2)
        
        # Open scan at index along axial axis
        ch3 = self.open_scan_axial()
        # Put channels in a grid-like image
        image_grid_ch3 = self.make_grid(ch3)
        
        if self.duplicate_channels:
            image_3ch = torch.stack((image_grid_ch2, image_grid_ch2, image_grid_ch2), axis = 0)
        else:
            image_3ch = torch.stack((image_grid_ch1, image_grid_ch2, image_grid_ch3), axis = 0)
            
        if self.rotate:
            image_3ch = torch.rot90(image_3ch, 1, [1, 2])
        
        if self.transforms:
            image_3ch = self.transforms(image_3ch)
        
        
        labels = {}
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                labels[key] = int(value)
        
        return image_3ch.float(), labels
    
    
    
class ReadersAtlas():
    def __init__(self, aal_dir, labels_dir, rotate=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.rotate = rotate
        self.regions = {
            
            'Primary sensorimotor cortex': ['Precentral_L',
                                            'Precentral_R',
                                            'Postcentral_L',
                                            'Postcentral_R',
                                            'Supp_Motor_Area_L',
                                            'Supp_Motor_Area_R',
                                            'Paracentral_Lobule_L',
                                            'Paracentral_Lobule_R'],
                   
           'Frontal lobe': ['Frontal_Sup_L',
                            'Frontal_Sup_R',
                            'Frontal_Sup_Orb_L',
                            'Frontal_Sup_Orb_R',
                            'Frontal_Mid_L',
                            'Frontal_Mid_R',
                            'Frontal_Mid_Orb_L',
                            'Frontal_Mid_Orb_R',
                            'Frontal_Inf_Oper_L',
                            'Frontal_Inf_Oper_R',
                            'Frontal_Inf_Tri_L',
                            'Frontal_Inf_Tri_R',
                            'Frontal_Inf_Orb_L',
                            'Frontal_Inf_Orb_R',
                            'Frontal_Sup_Medial_L',
                            'Frontal_Sup_Medial_R',
                            'Frontal_Med_Orb_L',
                            'Frontal_Med_Orb_R',
                            'Rectus_L',
                            'Rectus_R'],

            'Anterior cingulate': ['Cingulum_Ant_L',
                                   'Cingulum_Ant_R'],

            'Posterior cingulate': ['Cingulum_Post_L',
                                    'Cingulum_Post_R'],
    
            'Primary and associative visual cortex': ['Calcarine_L',
                                                      'Calcarine_R',
                                                      'Cuneus_L',
                                                      'Cuneus_R',
                                                      'Lingual_L',
                                                      'Lingual_R',
                                                      'Occipital_Sup_L',
                                                      'Occipital_Sup_R',
                                                      'Occipital_Mid_L',
                                                      'Occipital_Mid_R',
                                                      'Occipital_Inf_L',
                                                      'Occipital_Inf_R'],

            'Bilateral posterior parietotemporal': ['Parietal_Sup_L',
                                                    'Parietal_Sup_R',
                                                    'Parietal_Inf_L',
                                                    'Parietal_Inf_R'],

            'Basal ganglia': ['Caudate_L',
                              'Caudate_R',
                              'Putamen_L',
                              'Putamen_R',
                              'Pallidum_L',
                              'Pallidum_R',
                              'Thalamus_L',
                              'Thalamus_R'],

            'Anterior temporal lobe': ['Temporal_Pole_Sup_L',
                                       'Temporal_Pole_Sup_R',
                                       'Temporal_Pole_Mid_L',
                                       'Temporal_Pole_Mid_R'],
        }
    
    def open_aal_axial(self):
        # Load the AAL
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        
        if self.rotate:
            image = np.rot90(image)
        
        return image
    
    def get_data(self):
        aal_labels = {}
        labels = {}
        
        aal = torch.tensor(self.open_aal_axial().copy()).permute(2, 0, 1)
        
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                aal_labels[key] = int(value)
                
        image = torch.zeros(aal.shape)
        
        for i, region in enumerate(self.regions, 1):
            for sub_region in self.regions[region]:
                image += torch.where(aal==aal_labels[sub_region], i*1000, 0)
            labels[region] = i*1000
        
        return image.float(), labels
    
    
class ReadersAtlas3Channels():
    def __init__(self, aal_dir, labels_dir, transforms=None, rotate=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.rotate = rotate
        self.regions = {
            
            'Primary sensorimotor cortex': ['Precentral_L',
                                            'Precentral_R',
                                            'Postcentral_L',
                                            'Postcentral_R',
                                            'Supp_Motor_Area_L',
                                            'Supp_Motor_Area_R',
                                            'Paracentral_Lobule_L',
                                            'Paracentral_Lobule_R'],
                   
           'Frontal lobe': ['Frontal_Sup_L',
                            'Frontal_Sup_R',
                            'Frontal_Sup_Orb_L',
                            'Frontal_Sup_Orb_R',
                            'Frontal_Mid_L',
                            'Frontal_Mid_R',
                            'Frontal_Mid_Orb_L',
                            'Frontal_Mid_Orb_R',
                            'Frontal_Inf_Oper_L',
                            'Frontal_Inf_Oper_R',
                            'Frontal_Inf_Tri_L',
                            'Frontal_Inf_Tri_R',
                            'Frontal_Inf_Orb_L',
                            'Frontal_Inf_Orb_R',
                            'Frontal_Sup_Medial_L',
                            'Frontal_Sup_Medial_R',
                            'Frontal_Med_Orb_L',
                            'Frontal_Med_Orb_R',
                            'Rectus_L',
                            'Rectus_R'],

            'Anterior cingulate': ['Cingulum_Ant_L',
                                   'Cingulum_Ant_R'],

            'Posterior cingulate': ['Cingulum_Post_L',
                                    'Cingulum_Post_R'],
    
            'Primary and associative visual cortex': ['Calcarine_L',
                                                      'Calcarine_R',
                                                      'Cuneus_L',
                                                      'Cuneus_R',
                                                      'Lingual_L',
                                                      'Lingual_R',
                                                      'Occipital_Sup_L',
                                                      'Occipital_Sup_R',
                                                      'Occipital_Mid_L',
                                                      'Occipital_Mid_R',
                                                      'Occipital_Inf_L',
                                                      'Occipital_Inf_R'],

            'Bilateral posterior parietotemporal': ['Parietal_Sup_L',
                                                    'Parietal_Sup_R',
                                                    'Parietal_Inf_L',
                                                    'Parietal_Inf_R'],

            'Basal ganglia': ['Caudate_L',
                              'Caudate_R',
                              'Putamen_L',
                              'Putamen_R',
                              'Pallidum_L',
                              'Pallidum_R',
                              'Thalamus_L',
                              'Thalamus_R'],

            'Anterior temporal lobe': ['Temporal_Pole_Sup_L',
                                       'Temporal_Pole_Sup_R',
                                       'Temporal_Pole_Mid_L',
                                       'Temporal_Pole_Mid_R'],
        }
    
    def open_scan_sagittal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[11: 71, :, :]
        # For each scan channel
        for i in range(scan.shape[0]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[i, :, :], ((0, 0), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)

        return image
    
    def open_scan_coronal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, 11: 71, :]
        # For each scan channel
        for i in range(scan.shape[1]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            # WARNING: Use INTER_NEAREST to preserve atlas values!
            image[:, :, i] = cv2.resize(src=scan[:, i, :], dsize=(95, 95), interpolation=cv2.INTER_NEAREST)
            # image[:, :, i] = np.pad(scan[:, i, :], ((8, 8), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                    
        return image
    
    def open_scan_axial(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        # For each scan channel
        for i in range(scan.shape[2]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[:, :, i], ((8, 8), (0, 0)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                
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
    
    def get_data(self):
        aal_labels = {}
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                aal_labels[key] = int(value)
                
        labels = {'Primary sensorimotor cortex': 1000,
                  'Frontal lobe': 2000,
                  'Anterior cingulate': 3000,
                  'Posterior cingulate': 4000,
                  'Primary and associative visual cortex': 5000,
                  'Bilateral posterior parietotemporal': 6000,
                  'Basal ganglia': 7000,
                  'Anterior temporal lobe': 8000}
        
        ch1 = self.open_scan_sagittal()        
        image = torch.zeros(ch1.shape)
        for region in self.regions:
            for sub_region in self.regions[region]:
                image += torch.where(ch1==aal_labels[sub_region], labels[region], 0)
        image_grid_ch1 = self.make_grid(image)
        
        ch2 = self.open_scan_coronal()        
        image = torch.zeros(ch2.shape)
        for region in self.regions:
            for sub_region in self.regions[region]:
                image += torch.where(ch2==aal_labels[sub_region], labels[region], 0)
        image_grid_ch2 = self.make_grid(image)
        
        ch3 = self.open_scan_axial()        
        image = torch.zeros(ch3.shape)
        for region in self.regions:
            for sub_region in self.regions[region]:
                image += torch.where(ch3==aal_labels[sub_region], labels[region], 0)
        image_grid_ch3 = self.make_grid(image)
        
        image_3ch = torch.stack((image_grid_ch1, image_grid_ch2, image_grid_ch3), axis = 0)
            
        if self.rotate:
            image_3ch = torch.rot90(image_3ch, 1, [1, 2])
        
        if self.transforms:
            image_3ch = self.transforms(image_3ch)
        
        return image_3ch.float(), labels