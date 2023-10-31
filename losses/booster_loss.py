import torch
import torch.nn as nn
import torch.nn.functional as F

class BalanceCELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    '''

    def __init__(self, eps=1e-6, bce_scale=1, **kwargs):
        super(BalanceCELoss, self).__init__()
        from losses.balance_cross_entropy_loss import BalanceCrossEntropyLoss

        self.bce_loss = BalanceCrossEntropyLoss()

        self.bce_scale = bce_scale
        self.booster = kwargs['booster']

    def forward(self, pred, batch):
        
        binary_mask = torch.ones_like(batch['pixel_gt']).squeeze(1)

        bce_loss = self.bce_loss(pred['Binary'], batch['pixel_gt'], binary_mask)
        metrics = dict(bce_loss=bce_loss)
        
        loss = bce_loss

        if self.booster:
            _,_, gt_dim,_ = batch['pixel_gt'].shape

            boost1 = F.interpolate(pred['boost1'], (gt_dim,gt_dim), mode='bilinear', align_corners=False)
            boost2 = F.interpolate(pred['boost2'], (gt_dim,gt_dim), mode='bilinear', align_corners=False)
            boost3 = F.interpolate(pred['boost3'], (gt_dim,gt_dim), mode='bilinear', align_corners=False)

            booster_loss1 = self.bce_loss(boost1, batch['pixel_gt'], binary_mask)
            booster_loss2 = self.bce_loss(boost2, batch['pixel_gt'], binary_mask)
            booster_loss3 = self.bce_loss(boost3, batch['pixel_gt'], binary_mask)
            booster_loss = booster_loss1+booster_loss2+booster_loss3

            return loss, metrics, booster_loss
        else:
            return loss, metrics