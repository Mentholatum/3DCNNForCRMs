# 3DCNNForCRMs
Code for paper: Automated segmentation and detection of complicated cystic renal masses using 3D V-Net convolutional neural network on multiparametric MRI

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Data Preprocessing
The input images required for model creation, along with data augmentation operations, are conducted offline.
``` bash
python [certain modality]_process.py
```

# Training

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch:
``` bash
python train_[certain modality].py
--batchSz=2
--nEpochs=500
--classes=2
--inChannels=1
--dropout_rate=0.5
--lr=1e-3
--opt='adamw'
--model='VNET' or 'UNET3D'
--cuda
```
# Validation
You can use the following command to initiate model inference using PyTorch:
``` bash
python eval_[certain modality].py
--batchSz=1
--classes=2
--inChannels=1
--dropout_rate=0.5
--model='VNET' or 'UNET3D'
--resume='model_saved_path/model.pth'
--cuda
