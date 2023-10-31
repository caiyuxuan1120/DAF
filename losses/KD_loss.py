import os
import sys
from PIL.Image import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms 
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils.util import ssim_norm

class BalanceCosineLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(BalanceCosineLoss, self).__init__()
        self.margin = margin
    
    def forward(self,
                T_feature: torch.Tensor,
                S_feature: torch.Tensor, 
                gt: torch.Tensor):
        
        '''
        Args:
            T_feature: shape :math:`(N, C, H, W)`
            fuse_feature: shape :math:`(N, C, H, W)`
            gt: shape :math:`(N, 1, H, W)`
        '''
        b, channel, feature_size, _ = T_feature.shape
                
        resize_op = transforms.Resize((feature_size,feature_size),Image.NEAREST)
        gt = resize_op(gt)

        positive = gt.byte()
        negative = (1 - gt).byte()
        negative_count = int(negative.float().sum())
        
        loss = F.cosine_similarity(T_feature, S_feature, dim=1)  # B,H,W
        negative_loss = loss * negative.squeeze(1).float()  # normal area

        negative_loss = negative_loss.sum() / negative_count
        
        return 1-negative_loss

class BalanceSSIMLoss(nn.Module):
    def __init__(self, margin=0.3, eps=1e-10):
        super(BalanceSSIMLoss, self).__init__()
        self.margin = 0.3
        self.eps = eps
    
    def forward(self,
                T_feature: torch.Tensor,
                S_feature: torch.Tensor, 
                gt: torch.Tensor):
        
        '''
        Args:
            T_feature: shape :math:`(N, C, H, W)`
            fuse_feature: shape :math:`(N, C, H, W)`
            gt: shape :math:`(N, 1, H, W)`
        '''
        
        b, channel, feature_size, _ = T_feature.shape
        resize_op = transforms.Resize((feature_size,feature_size),Image.NEAREST)
        gt = resize_op(gt)

        positive = gt.byte()
        negative = (1 - gt).byte()
        
        stu_norm = ssim_norm(S_feature, self.eps)
        tea_norm = ssim_norm(T_feature, self.eps)

        loss_, ssim_map = ssim(stu_norm, tea_norm, data_range=1.0,size_average=True)

        negative_ssim_map = ssim_map * negative.float()
        negative_count = channel * int(negative.float().sum())
        inv_negative_ssim_map = 1 - negative_ssim_map
        negative_loss = inv_negative_ssim_map.sum() / negative_count

        return negative_loss
