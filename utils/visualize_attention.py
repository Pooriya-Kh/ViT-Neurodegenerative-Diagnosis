import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

# Code borrowed and modified from "https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb"
def visualize_attention(image, attention, device, show_map=False, rotate=False, dpi=300):
    image_size = image.shape[-1]
    
    att_mat = torch.stack(attention).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n].to(device), joint_attentions[n-1].to(device))

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), (image_size, image_size))[..., np.newaxis]
    image = image.permute(1, 2, 0)[:, :, 2].unsqueeze(2).numpy()
    result = mask * image
    
    if rotate:
        image = np.rot90(image)
    
    if show_map:
        fig, axes = plt.subplots(ncols=3, nrows=len(joint_attentions), figsize=(5, 20), dpi=dpi)
        for i, v in enumerate(joint_attentions):
            # Attention from the output token to the input space.
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), (image_size, image_size))[..., np.newaxis]
            result = mask * image

            axes[i, 0].imshow(image, cmap='hot')
            axes[i, 1].imshow(mask, cmap='hot')
            axes[i, 2].imshow(result.squeeze(), cmap='hot')

            axes[i, 0].set_title('Original', fontsize=5)
            axes[i, 1].set_title(f'Attention Map Layer {i+1}', fontsize=5)
            axes[i, 2].set_title('Overlay', fontsize=5)

            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
        
    else:

        fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(4.5, 6), dpi=dpi)
        for row in range(4):
            for col in range(3):
                idx = row * 3 + col
                v = joint_attentions[idx]

                mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
                mask = cv2.resize(mask / mask.max(), (image_size, image_size))[..., np.newaxis]
                result = mask * image

                axes[row, col].imshow(result.squeeze(), cmap='hot')
                axes[row, col].set_title(f'Attention Map Layer {idx+1}', fontsize=5)
                axes[row, col].axis("off")
    
        
        