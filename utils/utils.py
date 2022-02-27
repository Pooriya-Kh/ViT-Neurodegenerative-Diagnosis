import os
import torch

def count_parameters(model):
    return sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)

def save_model(stat_dict, save_dir, file_name):
    files = os.listdir(save_dir)
    
    if file_name in files:
        i = input("The filename exists. Do you want to replace it?")
        if i == 'y':
            torch.save(stat_dict, save_dir + file_name)
            print("Model replaced!")
        else:
            print("Skipped!")
    
    else:
        torch.save(stat_dict, save_dir + file_name)
        print("Model saved!")