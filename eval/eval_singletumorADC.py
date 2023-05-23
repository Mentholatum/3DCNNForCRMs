import sys

sys.path.append('..')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import lib.utils.general as utils
from monai.metrics import DiceMetric, ConfusionMatrixMetric, HausdorffDistanceMetric
from torch.utils.data import DataLoader
from lib.utils.model_creater import create_model
from lib.dataloader.singletumor_ADC import SingleTumorADC
from lib.utils.general import prepare_input
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

seed = 42


def get_arguments():
    parser = argparse.ArgumentParser(description='Single Tumor ADC Segmentation')
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="singletumorADC")
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume',
                        # default="../train/saved_models/VNET_checkpoints/VNET_singletumorADC/VNET_singletumorADC_BEST.pth",
                        # default="../train/saved_models/UNET3D_checkpoints/UNET3D_singletumorADC/UNET3D_singletumorADC_BEST.pth",

                        # default='../train/saved_models/VNET_checkpoints/VNET_singletumorADC_08_26_01_51/VNET_singletumorADC_08_26_01_51_BEST.pth',
                        # default='../train/saved_models/UNET3D_checkpoints/UNET3D_singletumorADC_08_26_11_09/UNET3D_singletumorADC_08_26_11_09_BEST.pth',
                        default='../train/saved_models/VNET_checkpoints/VNET_singletumorADC_10_09_10_50/VNET_singletumorADC_10_09_10_50_BEST.pth',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model',
                        type=str, default='VNET',
                        # type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'adamw', 'rmsprop'))

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    model, _ = create_model(args)

    # load checkpoint
    if args.resume is None:
        print("The model path in the inference phase is not available!")
        return

    else:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    # Move to GPU, if available
    if args.cuda:
        model = model.cuda()

    # Custom data loaders
    val_loader = SingleTumorADC('test')
    val_generator = DataLoader(val_loader, batch_size=1, shuffle=False, num_workers=2)
    train_loader = SingleTumorADC('train')
    training_generator = DataLoader(train_loader, batch_size=1, shuffle=False, num_workers=2)

    # Metric
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    acc_acc = ConfusionMatrixMetric(include_background=True,
                                    metric_name="accuracy",
                                    compute_sample=True,
                                    reduction=MetricReduction.MEAN,
                                    get_not_nans=False)
    spe_acc = ConfusionMatrixMetric(include_background=True,
                                    metric_name="specificity",
                                    compute_sample=True,
                                    reduction=MetricReduction.MEAN,
                                    get_not_nans=False)
    pre_acc = ConfusionMatrixMetric(include_background=True,
                                    metric_name="precision",
                                    compute_sample=True,
                                    reduction=MetricReduction.MEAN,
                                    get_not_nans=False)
    hausd_acc = HausdorffDistanceMetric(include_background=True,
                                        percentile=95.0,
                                        reduction=MetricReduction.MEAN,
                                        get_not_nans=False)
    post_label = AsDiscrete(to_onehot=args.classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.classes)

    dice = 0.0
    hausd = 0.0
    acc = 0.0
    spe = 0.0
    sen = 0.0
    pre = 0.0

    with torch.no_grad():
        for idx, input_tuple in enumerate(val_generator):
            data, target = prepare_input(input_tuple=input_tuple, args=args)
            logits = model(data)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            # Dice
            dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc1 = dice_acc.aggregate()
            avg_acc1 = acc1[0].detach().cpu().numpy()
            # print(float(avg_acc1))
            dice += avg_acc1

            # Accuracy
            acc_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc3 = acc_acc.aggregate()
            avg_acc3 = acc3[0].detach().cpu().numpy()
            # print(float(avg_acc3))
            acc += avg_acc3

            # Specificity
            spe_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc4 = spe_acc.aggregate()
            avg_acc4 = acc4[0].detach().cpu().numpy()
            # print(float(avg_acc4))
            spe += avg_acc4

            # Precision
            pre_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc5 = pre_acc.aggregate()
            avg_acc5 = acc5[0].detach().cpu().numpy()
            # print(float(avg_acc5))
            pre += avg_acc5

            # Hausdorff
            hausd_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc6 = hausd_acc.aggregate()
            avg_acc6 = acc6[0].detach().cpu().numpy()
            print(avg_acc6)
            hausd += avg_acc6

        avg_acc1 = dice / len(val_generator)
        avg_acc5 = pre / len(val_generator)
        avg_acc6 = hausd / len(val_generator)
        print('DSC:', avg_acc1, 'PRE:', avg_acc5, 'HDD:', avg_acc6)


if __name__ == '__main__':
    main()
