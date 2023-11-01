# DAF
An officical implementation of ["A Discrepancy Aware Framework for Robust Anomaly Detection"](https://ieeexplore.ieee.org/document/10272031)

![image](https://github.com/caiyuxuan1120/DAF/blob/main/framework.png)

# Datasets
Download the MVTecAD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).  
Download the DAGM dataset from [here](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection).  
In some experiments, we use the [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) as the source of anomaly data.

# Install
First, [Install PyTorch>= 1.11.0](https://pytorch.org/get-started/previous-versions/) and torchvision, and then install additional dependencies according to the requirements.txt. For instance,
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

# Training
Before training, some custom parameters need to be configured. For example:
```
python train.py --root_path '/YourMVTecPath' --source_path '/YourDTDPath' --batch_size 8 --lr 2e-4 --defect_cls bottle
```

# Evaluate
The checkpoints is avaliable at [Google Drive](https://drive.google.com/file/d/1tZr24lECmGZEeb-VokrWWsvDlkhLKg6L/view?usp=sharing)

To evaluate the performance with checkpoints:
```
bash test_DAF.sh
```

# TODO List
- [ ] Update the complete code for training and evaluation
- [ ] Update the checkpoints
- [ ] ...

# Citation
If you find this work helpful, please consider to cite our paper:
```
@ARTICLE{10272031,
  author={Cai, Yuxuan and Liang, Dingkang and Luo, Dongliang and He, Xinwei and Yang, Xin and Bai, Xiang},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={A Discrepancy Aware Framework for Robust Anomaly Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/TII.2023.3318302}}
```
