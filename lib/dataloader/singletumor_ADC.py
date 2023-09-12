import glob
import os
import random

import torch
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np


class SingleTumorADC(Dataset):
    def __init__(self, mode, dataset_path="/media/data1/jiachuang/projects/kidney/data"):
        self.mode = mode
        self.root = str(dataset_path)
        self.train_list = []
        self.label_list = []
        self.training_path = self.root + '/single/ADC/train/img'
        self.training_label_path = self.root + '/single/ADC/train/label'
        self.testing_path = self.root + '/single/ADC/val/img'
        self.testing_label_path = self.root + '/single/ADC/val/label'

        list_train_IDsADC = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
        label_train_IDsADC = sorted(glob.glob(os.path.join(self.training_label_path, '*.nii.gz')))
        self.train_datanum = len(list_train_IDsADC)

        list_val_IDsADC = sorted(glob.glob(os.path.join(self.testing_path, '*.nii.gz')))
        label_val_IDsADC = sorted(glob.glob(os.path.join(self.testing_label_path, '*.nii.gz')))
        self.val_datanum = len(list_val_IDsADC)

        assert len(list_train_IDsADC) == len(label_train_IDsADC)
        split_idx = int(self.train_datanum * 0.8)

        random.seed(42)
        random.shuffle(list_train_IDsADC)
        random.seed(42)
        random.shuffle(label_train_IDsADC)

        if self.mode == 'train':
            print('Single Tumor-ADC Dataset for Training. Total data:', int(self.train_datanum * 0.8))
            self.train_list = list_train_IDsADC[:split_idx]
            self.label_list = label_train_IDsADC[:split_idx]

        elif self.mode == 'val':
            print('Single Tumor-ADC Dataset for Validating. Total data:', int(self.train_datanum * 0.2))
            self.train_list = list_train_IDsADC[split_idx:]
            self.label_list = label_train_IDsADC[split_idx:]

        elif self.mode == 'test':
            print('Single Tumor-ADC Dataset for Test. Total data:', self.val_datanum)
            self.train_list = list_val_IDsADC
            self.label_list = label_val_IDsADC

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):
        img = nib.load(self.train_list[item]).get_fdata()
        lab = nib.load(self.label_list[item]).get_fdata()

        img = np.array(img).astype(np.float32)
        lab = np.array(lab).astype(np.float32)

        return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(lab).unsqueeze(0)
