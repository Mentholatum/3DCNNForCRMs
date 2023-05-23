import sys

sys.path.append('..')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import lib.utils.general as utils
from lib.utils.trainer import train_epoch, val_epoch, save_checkpoint
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch.utils.data import DataLoader
from monai.transforms import AsDiscrete
from lib.utils.model_creater import create_model
from lib.dataloader.singletumor_ADC import SingleTumorADC


seed = 42


def get_arguments():
    parser = argparse.ArgumentParser(description='Single Tumor ADC Segmentation')
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="singletumorADC")
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume',
                        default=None,
                        type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str,
                        # default='UNET3D',
                        default='VNET',
                        choices=('VNET', 'VNET2', 'UNET3D'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'adamw', 'rmsprop'))

    args = parser.parse_args()

    args.save = './saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}'.format(
        args.dataset_name, utils.datestr())
    return args


def main():
    global start_epoch, epochs_since_improvement, best_dice, best_hauds, loss
    args = get_arguments()
    utils.reproducibility(args, seed)
    model, optimizer = create_model(args)
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if args.resume is None:
        utils.make_dirs(args.save)
        start_epoch = 1
        best_dice = 0
        best_hauds = 100
        loss = 100

    else:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        best_hauds = checkpoint['hauds']
        model.load_state_dict(checkpoint['model_state_dict'])

    # Move to GPU, if available
    if args.cuda:
        model = model.cuda()

    # Loss function
    criterion = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=0.0,
                           smooth_dr=1e-6)

    # Metric
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    post_label = AsDiscrete(to_onehot=args.classes)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=args.classes)

    # Custom data loaders
    train_loader = SingleTumorADC('train')
    val_loader = SingleTumorADC('val')
    training_generator = DataLoader(train_loader, batch_size=args.batchSz, shuffle=True, num_workers=2)
    val_generator = DataLoader(val_loader, batch_size=1, shuffle=True, num_workers=2)

    # Epochs
    for epoch in range(start_epoch, args.nEpochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 40
        if epochs_since_improvement == 40:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            utils.adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_epoch(model=model,
                    loader=training_generator,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss_func=criterion,
                    args=args)

        # One epoch's validation
        recent_dice, hauds, new_loss = val_epoch(model=model,
                                                loader=val_generator,
                                                epoch=epoch,
                                                loss_func=criterion,
                                                post_label=post_label,
                                                post_pred=post_pred,
                                                acc_func=dice_acc,
                                                args=args)

        # Check if there was an improvement
        is_best = new_loss < loss

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0
            best_dice = recent_dice
            best_hauds = hauds
            loss = new_loss

        # Save checkpoint
        save_checkpoint(directory=args.save, model=model,
                        epochs_since_improvement=epochs_since_improvement,
                        epoch=epoch, best_dice=best_dice, hauds=best_hauds,
                        is_best=is_best, optimizer=optimizer)


if __name__ == '__main__':
    main()
