import torch
from torch.utils.data import DataLoader

class ADNILoader(DataLoader):
    def __init__(self, **hparams):
        self.hparams = hparams
        
    def collate_fn(self, samples):
        pixel_values = torch.stack([sample[0] for sample in samples])
        labels = torch.tensor([sample[1] for sample in samples])
        return pixel_values, labels
        
    def train_dataloader(self):
        ds = self.hparams['train_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['train_batch_size'],
                                shuffle=self.hparams['train_shuffle'],
                                num_workers=self.hparams['num_workers'],
                                drop_last=self.hparams['train_drop_last'],
                                collate_fn = self.collate_fn
                               )
        return dataloader
    
    def validation_dataloader(self):
        ds = self.hparams['valid_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['valid_batch_size'],
                                shuffle=self.hparams['valid_shuffle'],
                                num_workers=self.hparams['num_workers'],
                                drop_last=self.hparams['valid_drop_last'],
                                collate_fn = self.collate_fn
                               )
        return dataloader
    
    def test_dataloader(self):
        ds = self.hparams['test_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['test_batch_size'],
                                shuffle=self.hparams['test_shuffle'],
                                num_workers=self.hparams['num_workers'],
                                drop_last=self.hparams['test_drop_last'],
                                collate_fn = self.collate_fn
                               )
        return dataloader