import os
import torch
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)

def save_model(stat_dict, save_dir, file_name):
    files = os.listdir(save_dir)
    
    if file_name in files:
        i = input("The filename exists. Do you want to replace it? (y/n)")
        if i == 'y':
            torch.save(stat_dict, save_dir + file_name)
            print("Model replaced!")
        else:
            print("Skipped!")
    
    else:
        torch.save(stat_dict, save_dir + file_name)
        print("Model saved!")
        
        
def image_split(image, channel):    
    crops = []
    
    for row in range(3):
        for col in range(3):
            crops.append(image[channel, row*128:(row*128)+128, col*128:(col*128)+128])
    
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    ax.imshow(image[channel, :, :])
    ax.axis("off")
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(2, 2), dpi=300)
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            axes[row, col].imshow(crops[idx])
            axes[row, col].axis("off");

    fig, axes = plt.subplots(ncols=9, figsize=(8, 2), dpi=300)
    for idx in range(9):
            axes[idx].imshow(crops[idx])
            axes[idx].axis("off");
            
def image_split_multi_channel(image, channel):    
    crops = [image[channel, 0:31, 0:26],
             image[channel, 0:31, 26:52],
             image[channel, 0:31, 52:78],
             image[channel, 31:62, 0:26],
             image[channel, 31:62, 26:52],
             image[channel, 31:62, 52:78],
             image[channel, 62:93, 0:26],
             image[channel, 62:93, 26:52],
             image[channel, 62:93, 52:78]]
    
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    ax.imshow(image[channel, :, :])
    ax.axis("off")
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(1, 1), dpi=300)
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            axes[row, col].imshow(crops[idx])
            axes[row, col].axis("off");
            
    # plt.savefig(f'{channel}.png')

    fig, axes = plt.subplots(ncols=9, figsize=(8, 2), dpi=300)
    for idx in range(9):
            axes[idx].imshow(crops[idx])
            axes[idx].axis("off");
            
    # plt.savefig(f'{channel}_flatten.png')