from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import interpolate
from ptflops import get_model_complexity_info
from utils.calc_pro import compute_pro


def eval_metric(gt, pred, img_gt, img_pred):
    _rocauc = roc_auc_score(gt.flatten(), pred[:,None,:,:].flatten())
    _ap = average_precision_score(gt.flatten(), pred[:,None,:,:].flatten())
    _pro_auc = compute_pro(gt.squeeze(), pred)
    _img_rocauc = roc_auc_score(img_gt.flatten(), img_pred.flatten())        
    
    return _rocauc, _pro_auc, _ap, _img_rocauc

def norm_t2np(data):
    data = data.permute(1,2,0).detach().cpu().numpy()
    data = (data - data.min()) / (data.max() - data.min())
    return data

def get_parameter_number(model): # params
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total params': total_num / 1e6, 'Trainable params': trainable_num/ 1e6}

def cal_param_flop(model):
    # if module_name == 'encoder':
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    return flops, params
    