from pathlib import Path
import os
import sys
import argparse
import torch
import csv
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from warnings import simplefilter

sys.path.append('../')
from logger.logger import setup_logging, get_logger
from utils.util import set_seed
from utils.test_funcs import *
from data.data_loader import MVTecDRAEMTestDataset,MVTecDRAEMTrainDataset
from model.DAF import DAFModel, Segmodel_UnetLike
from model.Student import STUdent
from model.Teacher import resnet18

up_4x = nn.Upsample(scale_factor=4)
up_8x = nn.Upsample(scale_factor=8)
up_16x = nn.Upsample(scale_factor=16)
simplefilter(action='ignore', category=FutureWarning)


def conf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/DatasetPath')
    parser.add_argument('--source_path', type=str, default='/SourceDatasetPath')
    parser.add_argument('--defect_cls', type=str, default='toothbrush')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--metrics', type=str, default='Cos_SSIM')

    parser.add_argument('--In_channel', type=int, default=256)
    parser.add_argument('--L1_channel', type=int, default=64,choices=[128,512])
    parser.add_argument('--L2_channel', type=int, default=128,choices=[256,1024,512])
    parser.add_argument('--L3_channel', type=int, default=256,choices=[256,1024,512])

    parser.add_argument('--model_save', type=str, default='./ckpts/DAF')
    parser.add_argument('--log_dir', type=str, default='./log_out/DAF')

    args = parser.parse_args()

    return args

def main(args):
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_dir)
    set_seed(1126)

    test_dataset = MVTecDRAEMTestDataset(root_dir=os.path.join(args.root_path, args.defect_cls, 'test'),
                                                resize_shape=[256,256])
    test_dataload = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    Teacher = resnet18(pretrained=True)
    Student = STUdent(pretrained=False)
    seg_model = Segmodel_UnetLike(args.L1_channel, args.L2_channel,  args.L3_channel, args.In_channel) 

    model = DAFModel(Teacher,Student, seg_model, L1_channel=args.L1_channel, L2_channel=args.L2_channel,  L3_channel=args.L3_channel,In_channel=args.In_channel, metrics=args.metrics).to(args.device)
    ckpts = torch.load(os.path.join(args.model_save, args.defect_cls, 'model_1199.pth'), map_location=args.device)
    model.load_state_dict(ckpts['net'])
    model.eval()

    img_label_list, img_score_list, gt_list, pix_pred_list = [], [], [], []

    for idx, batch in enumerate(tqdm(test_dataload)):
        for key, input_value in batch.items():
            if input_value is not None and isinstance(input_value, torch.Tensor):
                batch[key] = input_value.to(args.device)
        
        with torch.no_grad():
            result = model(batch['ori_img'], batch['ori_img'])
            ST_scoremap = up_4x(result['Lay1_map']).mean(1) + up_8x(result['Lay2_map']).mean(1) + \
                up_16x(result['Lay3_map']).mean(1)
            pix_scoremap = ST_scoremap.unsqueeze(1) + 3*result['Binary']
            pix_scoremap = gaussian_filter(pix_scoremap.cpu(), 4)
            img_score = torch.topk(torch.tensor(pix_scoremap).view(pix_scoremap.shape[0], -1), k=50, dim=-1)[0].mean(1)

        gt_list.append(batch['ground_truth'].cpu())
        pix_pred_list.append(pix_scoremap)
        
        img_label_list.append(batch['has_anomaly'].cpu())
        img_score_list.append(img_score)

    gt = torch.cat(gt_list, 0).numpy()
    pix_pred = np.concatenate(pix_pred_list, 0)
    img_gt = torch.cat(img_label_list, 0).numpy()
    img_pred = torch.cat(img_score_list, 0).numpy()

    pix_auc, pix_pro, pix_ap, img_auc = eval_metric(gt, pix_pred.squeeze(1), img_gt,img_pred)


    header = ['cls','Img_AUC','Pix_AUC', 'PRO', 'P_mAP']
    data = [{'cls':args.defect_cls , 'Img_AUC':round(img_auc,3),'Pix_AUC':round(pix_auc,3),'PRO':round(pix_pro,3), 'P_mAP':round(pix_ap, 3)}]
    with open ('results.csv','a',encoding='utf-8',newline='') as fp:
        writer =csv.DictWriter(fp,header)
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    args = conf_args()
    main(args)