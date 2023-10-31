from pathlib import Path
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from trainer.trainer import Trainer

sys.path.append('../')
from logger.logger import setup_logging, get_logger
from utils.util import set_seed,ssim_norm
from data.data_loader import MVTecDRAEMTestDataset,MVTecDRAEMTrainDataset
from model.DAF import DAFModel,Segmodel_UnetLike
from model.Student import STUdent
from model.Teacher import resnet18
from warmup_scheduler.scheduler import GradualWarmupScheduler

def conf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/DatasetPath') # DatasetPath 
    parser.add_argument('--source_path', type=str, default='/SourceDatasetPath') # SourceDatasetPath
    parser.add_argument('--defect_cls', type=str, default='toothbrush')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--In_channel', type=int, default=256)
    parser.add_argument('--L1_channel', type=int, default=64,choices=[128,512])
    parser.add_argument('--L2_channel', type=int, default=128,choices=[256,1024,512])
    parser.add_argument('--L3_channel', type=int, default=256,choices=[256,1024,512])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--Resume', type=bool, default=False)
    parser.add_argument('--metrics', type=str, default='Cos_SSIM')

    parser.add_argument('--model_save', type=str, default='./ckpts/DAF')
    parser.add_argument('--log_dir', type=str, default='./log_out/DAF')
    parser.add_argument('--board_dir', type=str, default='./board_out/DAF')

    args = parser.parse_args()

    return args

def main(args):
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_dir)
    logger = get_logger('train')
    set_seed(1126)

    source_path = args.source_path
    train_set = MVTecDRAEMTrainDataset(
                                root_dir=os.path.join(args.root_path,args.defect_cls, 'train/good'), \
                                anomaly_source_path=source_path,
                                resize_shape=[256, 256])
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size,
                              shuffle=False)

    test_dataset = MVTecDRAEMTestDataset(root_dir=os.path.join(args.root_path, args.defect_cls, 'test'),
                                         resize_shape=[256,256])
    test_dataload = DataLoader(test_dataset, 
                               batch_size=1, 
                               shuffle=False)
    
    Teacher = resnet18(pretrained=True)
    Student = STUdent(pretrained=False)
    seg_model = Segmodel_UnetLike(args.L1_channel, args.L2_channel, args.L3_channel, args.In_channel)
    
    model = DAFModel(Teacher,Student, seg_model, 
                     L1_channel=args.L1_channel, 
                     L2_channel=args.L2_channel,  
                     L3_channel=args.L3_channel, 
                     In_channel=args.In_channel, metrics=args.metrics).to(args.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,  model.parameters()), 
                                  lr=args.lr, 
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [650,950,], gamma=0.2)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler)

    _trainer = Trainer(model, optimizer, 
                       args.device, args.epochs, args.batch_size, args.defect_cls, args.model_save,
                       train_set, train_loader, test_dataload, 
                       args.board_dir, scheduler=scheduler_warmup,
                       Resume=args.Resume, metrics=args.metrics)
    logger.info('training start')
    _trainer.train()


if __name__ == '__main__':
    args = conf_args()
    main(args)