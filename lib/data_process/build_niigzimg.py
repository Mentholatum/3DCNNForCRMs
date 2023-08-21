import cv2
import os
import numpy as np
from PIL import Image
from skimage import io, transform, exposure
import torch
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import mutils.datautils as datautils
import pydicom
from numba import jit
import SimpleITK as sitk

target_file_path = "./data/nxzl/__data_4.19/train125/"
save_file_path = "./data/nxzl/ds_4.19/ds_train/"

folder1 = "corticomedullary phase"
folder2 = "excretory phase"
folder3 = "nephrographic phase"
folder4 = "DWI"
folder5 = "T1WI"
folder6 = "T2WI"
folder6 = "ADC"


@jit
def find_centroid(img):
    x = 0
    y = 0
    z = 0
    sum = 0
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if img[i, j, k] > 0:
                    sum += 1
                    y += k
                    x += j
                    z += i
    x = x // sum
    y = y // sum
    z = z // sum
    return z, x, y


@jit
def buildLabel(data, tx, ty, tz):  # 64*128*128 / 16  = 4 * 8 * 8  = 256
    ret = np.zeros((256, 4))
    index = 0
    data = np.squeeze(data)
    z, x, y = data.shape
    for iz in range(z // 16):
        for ix in range(x // 16):
            for iy in range(y // 16):
                target = data[iz * 16:(iz + 1) * 16, ix * 16: (ix + 1) * 16, iy * 16:(iy + 1) * 16]
                ret[index, 0] = np.sum(target)
                ret[index, 1] = tz
                ret[index, 2] = tx
                ret[index, 3] = ty
                index += 1
    # ret[:,0:1] = softmax(ret[:,0:1], axis = 0)
    ret[:, 0:1] /= np.sum(ret[:, 0:1])
    return ret


def build(id, folder):
    print(id, folder)
    file_path = target_file_path + id + "/" + folder
    slices = os.listdir(file_path)
    sets = []
    centroid_suffix = ""
    for slice in slices:
        if "nii" in slice:
            img = nib.load(file_path + "/" + slice).get_data()
            img = np.array(img)
            img = img.astype(np.float32)
            img = img.transpose((2, 1, 0))
            z, x, y = find_centroid(img)

            img = nib.Nifti1Image(img, np.eye(4))
            if not os.path.exists(save_file_path + folder + "/"):
                os.makedirs(save_file_path + folder + "/")
            nib.save(img, save_file_path + folder + "/" + "label/" + id + ".nii.gz")
        else:
            dicom = pydicom.dcmread(file_path + "/" + slice)
            sets.append(dicom)

    sets = sorted(sets, key=lambda s: s.SliceLocation)  # 按照slice Location排序
    for i in range(len(sets) - 1):
        if sets[i].SliceLocation == sets[i + 1].SliceLocation:
            sets[i].SliceLocation = sets[i].SliceLocation - 1000
    sets = sorted(sets, key=lambda s: s.SliceLocation)  # 按照slice Location排序

    for i in range(len(sets)):
        sets[i] = sets[i].pixel_array
    img = np.array(sets)
    img = img.astype(np.float32)
    img = nib.Nifti1Image(img, np.eye(4))

    if not os.path.exists(save_file_path + folder + "/"):
        os.makedirs(save_file_path + folder + "/")
    nib.save(img, save_file_path + folder + "/" + "train/" + id + ".nii.gz")
    return


def main():
    patients_ids = os.listdir(target_file_path)
    for id in patients_ids:
        modes = os.listdir(target_file_path + id)
        for mode in modes:
            if "corti" in mode or "Co" in mode:  build(id, folder1)
            if "excre" in mode or "Ex" in mode:  build(id, folder2)
            if "nephr" in mode or "Ne" in mode:  build(id, folder3)
            if "DWI" in mode or "dwi" in mode:  build(id, folder4)
            if "T1" in mode or "t1" in mode:  build(id, folder5)
            if "T2" in mode or "t2" in mode:  build(id, folder6)
            if "ADC" in mode or "adc" in mode:  build(id, folder6)
    return


if __name__ == '__main__':
    main()
