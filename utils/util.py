import torch.nn.functional as F
import torch
import numpy as np
import random

def ssim_norm(feature, eps=1e-10):
    
    b,channel,_,_ = feature.shape

    pad_feature = F.pad(feature,(5,5,5,5), mode='reflect')
    feature_ = pad_feature.view(b,channel,-1)
    # feature_ = feature.view(b,channel,-1)
    perc_max = feature_.max(2)[0].unsqueeze(2).unsqueeze(2)
    norm_feature = pad_feature / (perc_max + eps)
    # norm_feature = feature / (perc_max + eps)
    return norm_feature

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)
    random.seed(seeds)