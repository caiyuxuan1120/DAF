from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
from pytorch_msssim.ssim import ssim
from utils.util import ssim_norm

L1_distance = nn.SmoothL1Loss(reduction='none')
BatchNorm2d = nn.BatchNorm2d
LayerNorm2d = nn.LayerNorm


class DAFModel(nn.Module):
    def __init__(self, Normal_Tmodel,Smodel, Segmodel, L1_channel, L2_channel, L3_channel, In_channel, metrics):
        super(DAFModel, self).__init__()
        
        self.Tmodel = Normal_Tmodel

        self.Smodel = Smodel 
        self.Segmodel = Segmodel
        
        self.metrics = metrics
        assert self.metrics in ['Cosine', 'SSIM', 'Cos_SSIM']
        
        for _, param in self.Tmodel.named_parameters():
            param.requires_grad = False

        self.Smooth_layer = nn.Sequential(
            nn.Conv2d(L3_channel, In_channel, 1, bias=False),
            nn.BatchNorm2d(In_channel),
            nn.ReLU(inplace=True))
        self.Smooth_layer.apply(self.weights_init)
        
        self.boost_head1 = nn.Sequential(
            nn.Conv2d(L1_channel, L1_channel//4 , kernel_size=3, padding=1),
            nn.BatchNorm2d(L1_channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(L1_channel//4 , 1, kernel_size=1),
            nn.Sigmoid())
        
        self.boost_head2 = nn.Sequential(
            nn.Conv2d(L2_channel, L2_channel//4 , kernel_size=3, padding=1),
            nn.BatchNorm2d(L2_channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(L2_channel//4 , 1, kernel_size=1),
            nn.Sigmoid())
        
        self.boost_head3 = nn.Sequential(
            nn.Conv2d(L3_channel, L3_channel//4 , kernel_size=3, padding=1),
            nn.BatchNorm2d(L3_channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(L3_channel//4 , 1, kernel_size=1),
            nn.Sigmoid())
        self.boost_head3.apply(self.weights_init)
        self.boost_head2.apply(self.weights_init)
        self.boost_head1.apply(self.weights_init)

        self.up_2x = nn.Upsample(scale_factor=2)
        self.up_4x = nn.Upsample(scale_factor=4)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    def forward(self, ori_image, syn_image):
        with torch.no_grad():
            Lay1T_out, Lay2T_out, Lay3T_out, _ = self.Tmodel(ori_image)
        Lay1S_out, Lay2S_out, Lay3S_out = self.Smodel(syn_image)
        
        if self.metrics == 'Cos_SSIM':
            _, Lay1_Smap = ssim(ssim_norm(Lay1S_out),ssim_norm(Lay1T_out),data_range=1.0,size_average=False)
            _, Lay2_Smap = ssim(ssim_norm(Lay2S_out),ssim_norm(Lay2T_out),data_range=1.0,size_average=False)
            _, Lay3_Smap = ssim(ssim_norm(Lay3S_out),ssim_norm(Lay3T_out),data_range=1.0,size_average=False)
            
            Lay1_Cmap = 1-F.cosine_similarity(Lay1S_out, Lay1T_out, dim=1).unsqueeze(1)
            Lay2_Cmap = 1-F.cosine_similarity(Lay2S_out, Lay2T_out, dim=1).unsqueeze(1)
            Lay3_Cmap = 1-F.cosine_similarity(Lay3S_out, Lay3T_out, dim=1).unsqueeze(1)

            Lay1_map = 1 - Lay1_Smap + Lay1_Cmap
            Lay2_map = 1 - Lay2_Smap + Lay2_Cmap
            Lay3_map = 1 - Lay3_Smap + Lay3_Cmap

        # simple repeat 
        score_fuse = Lay1_map.repeat(1,4,1,1) + self.up_2x(Lay2_map).repeat(1,2,1,1) + self.up_4x(Lay3_map)
        score_fuse = self.Smooth_layer(score_fuse)
        Binary = self.Segmodel(score_fuse, Lay1S_out, Lay2S_out, Lay3S_out)
        
        boost1 = self.boost_head1(Lay1S_out)
        boost2 = self.boost_head2(Lay2S_out)
        boost3 = self.boost_head3(Lay3S_out)

        result = OrderedDict(Lay2_map=Lay2_map, Lay3_map=Lay3_map,Lay1_map=Lay1_map,Binary=Binary)
        result.update(Lay1_Sfeat=Lay1S_out, Lay2_Sfeat=Lay2S_out, Lay3_Sfeat=Lay3S_out)
        result.update(Lay1_Tfeat=Lay1T_out, Lay2_Tfeat=Lay2T_out, Lay3_Tfeat=Lay3T_out)
        result.update(boost1=boost1,boost2=boost2, boost3=boost3)

        return result

"""Segmentaion Head"""
class Segmodel_UnetLike(nn.Module):
    def __init__(self, L1_channel, L2_channel, L3_channel, In_channel):
        super(Segmodel_UnetLike, self).__init__()

        self.binarize3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(In_channel + L3_channel, In_channel, 3, padding=1, bias=False),
            BatchNorm2d(In_channel),
            nn.ReLU(inplace=True))
        
        self.binarize2 = nn.Sequential(
            nn.ConvTranspose2d(In_channel + L2_channel, In_channel//2, 2, 2),
            BatchNorm2d(In_channel//2),
            nn.ReLU(inplace=True))

        self.binarize1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(In_channel//2 + L1_channel, In_channel//4, 1, 1, bias=False),
            BatchNorm2d(In_channel//4),
            nn.ReLU(inplace=True))

        self.Seghead = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(In_channel//4, 1, kernel_size=1),
            nn.Sigmoid())

        self.binarize3.apply(self.weights_init)
        self.binarize2.apply(self.weights_init)
        self.binarize1.apply(self.weights_init)
        self.Seghead.apply(self.weights_init)
        
        self.up_2x = nn.Upsample(scale_factor=2)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    def forward(self, score_fuse, l1,l2,l3):
        _, _ , l3_shape, _ = l3.shape
        score_fuse = F.interpolate(score_fuse, (l3_shape,l3_shape), mode='bilinear', align_corners=False)
        d3_in = torch.cat((score_fuse, l3), dim=1)
        d3_out = self.binarize3(d3_in)
        d2_in = torch.cat((d3_out,l2), dim=1)
        d2_out = self.binarize2(d2_in)
        d1_in = torch.cat((d2_out,l1), dim=1)
        d1_out = self.binarize1(d1_in)
        binary = self.Seghead(d1_out)
        return binary

