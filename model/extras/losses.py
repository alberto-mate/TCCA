import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss


class TruncatedMeanSquareLoss(nn.Module):
    def __init__(self, max_clip=16):
        super(TruncatedMeanSquareLoss, self).__init__()
        self.max_clip = max_clip
        self.mse = MSELoss(reduction='none')

    def forward(self, logits, mask):
        # Logits shape: (B, T, C)
        # Mask shape: (B, T)

        # Compute the MSE loss for each time step delayed by one
        mse_loss = self.mse(F.log_softmax(logits[:, 1:, :], dim=2), F.log_softmax(logits[:, :-1, :], dim=2))
        # Truncate the loss
        mse_loss = torch.clamp(mse_loss, min=0, max=self.max_clip)
        # Multiply by the mask
        mse_loss = mse_loss * mask[:, 1:].unsqueeze(2)
        # Compute the mean loss
        mean_loss = torch.mean(mse_loss)
        return mean_loss
        