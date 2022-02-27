# Code borrowed and adapted from "https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb"

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

def visualize_attention(image, attention, device):
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
    image = image.permute(1, 2, 0)[:, :, 0].unsqueeze(2).numpy()
    result = (mask * image).astype("uint8")

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (image_size, image_size))[..., np.newaxis]
        result = (mask * image).astype("uint8")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16), dpi=150)
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i+1))
        _ = ax1.imshow(image)
        _ = ax2.imshow(mask)
    