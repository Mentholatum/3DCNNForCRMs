# 3DCNNForCRMs
Code for Paper: [Automated segmentation of complicated cystic renal masses using 3D V-Net convolutional neural network on MRI](https://academic.oup.com/bjr/article-abstract/99/1180/812/8465957?redirectedFrom=fulltext)

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
```


# Citation
If this code or article has inspired you, please cite it:
```bash
@article{kang2026automated,
  title={Automated Segmentation of Complicated Cystic Renal Masses Using 3D V-Net Convolutional Neural Network on MRI},
  author={Kang, Huanhuan and Jia, Chuang and Wang, Zhongyi and Huang, Bin and Wang, He and Jiang, Jiahui and Liu, Zhe and Cui, Mengqiu and Zhao, Jian and Bai, Xu and others},
  journal={British Journal of Radiology},
  pages={811-822},
  year={2026},
  url={https://doi.org/10.1093/bjr/tqag027},
  publisher={Oxford University Press}
}
```
