import pandas as pd
import torch
import sys
import os
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from warnings import simplefilter
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')
from logger.logger import get_logger
from losses.booster_loss import BalanceCELoss
from losses.KD_loss import BalanceSSIMLoss, BalanceCosineLoss

L1_distance = nn.SmoothL1Loss(reduction='none')
niter = 0
simplefilter(action='ignore', category=FutureWarning)


class Trainer:
    def __init__(self,  model, optim,
                        device,epochs,batch_size,defect_cls,save_dir,
                        train_set, train_loader, test_loader,board_dir,scheduler=None,
                        log_step=40, Resume=False, metrics='all',
                        bc_criteria=BalanceCELoss(booster=True)):
        super(Trainer,self).__init__()

        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.niter = niter
        self.bc_criteria = bc_criteria

        self.resume = Resume
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.defect_cls = defect_cls
        self.save_dir = Path(save_dir)

        self.train_set = train_set
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if not os.path.exists(os.path.join(board_dir, defect_cls)):
            os.makedirs(os.path.join(board_dir, defect_cls), exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=os.path.join(board_dir, defect_cls))
        self.logger = get_logger('trainer')
        self.log_step = log_step
        self.metrics = metrics
        
        self.up_4x = nn.Upsample(scale_factor=4)
        self.up_8x = nn.Upsample(scale_factor=8)
        self.up_16x = nn.Upsample(scale_factor=16)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.Tmodel.eval()

        for step_idx, batch in enumerate(tqdm(self.train_loader)):
    
            for key, input_value in batch.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    batch[key] = input_value.to(self.device)
            nbatches, _, _, _ = batch['ori_img'].shape
            
            result = self.model(batch['ori_img'], batch['aug_img'])

            lay1_loss = BalanceCosineLoss()(result['Lay1_Tfeat'], result['Lay1_Sfeat'], batch['pixel_gt']) + \
                        BalanceSSIMLoss()(result['Lay1_Tfeat'], result['Lay1_Sfeat'], batch['pixel_gt'])
            lay2_loss = BalanceCosineLoss()(result['Lay2_Tfeat'], result['Lay2_Sfeat'], batch['pixel_gt']) + \
                        BalanceSSIMLoss()(result['Lay2_Tfeat'], result['Lay2_Sfeat'], batch['pixel_gt'])
            lay3_loss = BalanceCosineLoss()(result['Lay3_Tfeat'], result['Lay3_Sfeat'], batch['pixel_gt']) + \
                        BalanceSSIMLoss()(result['Lay3_Tfeat'], result['Lay3_Sfeat'], batch['pixel_gt']) 
            
            seg_loss, _, boost_loss = self.bc_criteria(result, batch)  # TODO

            total_loss = lay1_loss + lay2_loss + lay3_loss + seg_loss + boost_loss
            
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            self.niter += 1
            # log messages
            if step_idx % self.log_step == 0 or step_idx == 1:
                self.logger.info(f"Train Epoch:[{epoch}/{self.epochs}] Step:[{step_idx}/{len(self.train_loader)}] "
                                f"Total_loss:{total_loss.item():.6f} Layer1 loss:{lay1_loss.item():6f} Layer2 loss:{lay2_loss.item():.6f} Layer3 loss:{lay3_loss.item():6f}  Segment loss:{seg_loss.item():6f}  Booster loss:{boost_loss.item():6f} ")

            if self.niter % 100 == 0:
                self.writer.add_scalar('total_loss/layer1_loss', lay1_loss.item(), global_step=self.niter)
                self.writer.add_scalar('total_loss/layer2_loss', lay2_loss.item(), global_step=self.niter)
                self.writer.add_scalar('total_loss/layer3_loss', lay3_loss.item(), global_step=self.niter)
                self.writer.add_scalar('total_loss/segment_loss', seg_loss.item(), global_step=self.niter)
                self.writer.add_scalar('total_loss/boost_loss', boost_loss.item(), global_step=self.niter)


        if self.scheduler is not None:
            self.scheduler.step()
            print('learning rate:',self.optim.state_dict()['param_groups'][0]['lr'])


    def train(self):
        self.logger.info('train on cls:{}'.format(self.defect_cls))
        
        if not self.resume:
            start_epoch = -1
        else:
            break_ckpt = torch.load((self.save_dir / self.defect_cls / 'model.pth'), map_location=self.device)
            self.model.load_state_dict(break_ckpt['net'])
            self.optim.load_state_dict(break_ckpt['optimizer'])
            self.scheduler.load_state_dict(break_ckpt['lr_scheduler'])
            start_epoch = break_ckpt['epoch']

        for epoch in range(start_epoch+1, self.epochs):
            self.train_epoch(epoch)

            if (epoch+1) == self.epochs:
                self.resume_ckpt('model_{}.pth'.format(epoch), epoch)


    def resume_ckpt(self, ckpt_name,epoch):
        save_dir = (self.save_dir / self.defect_cls)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        if self.scheduler is not None:
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer':self.optim.state_dict(),
                'lr_scheduler':self.scheduler.state_dict(),
                "epoch": epoch
                }
        else:
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer':self.optim.state_dict(),
                "epoch": epoch
                } 
        
        torch.save(checkpoint, (save_dir / ckpt_name))
