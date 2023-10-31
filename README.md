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
# TODO List
- [ ] Update the complete code for training and evaluation
- [ ] Update the checkpoints
- [ ] ...
