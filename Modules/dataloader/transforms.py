import torch

class GaussianNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        noisy_image = image + (torch.randn(image.size()) * self.std) + self.mean
        return noisy_image