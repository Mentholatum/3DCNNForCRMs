import os
import time

import numpy as np
import torch
from lib.utils.general import prepare_input
from monai.data import decollate_batch
from monai.utils import Average
from monai.utils.enums import MetricReduction
from monai.metrics import ROCAUCMetric, HausdorffDistanceMetric


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model, loader, optimizer, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, input_tuple in enumerate(loader):
        optimizer.zero_grad()
        data, target = prepare_input(input_tuple=input_tuple, args=args)
        data.requires_grad = True
        for param in model.parameters(): param.grad = None
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        run_loss.update(loss.item(), n=args.batchSz)
        print('Epoch: {}/{} {}/{}'.format(epoch, args.nEpochs, idx + 1, len(loader)),
              'loss: {:.4f}'.format(run_loss.avg),
              'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters(): param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, loss_func, post_label=None,
              post_pred=None):
    model.eval()
    dice = 0.0
    hausd = 0.0
    run_loss = AverageMeter()
    hausd_func = HausdorffDistanceMetric(include_background=True,
                                         reduction=MetricReduction.MEAN,
                                         percentile=95,
                                         get_not_nans=True)
    print("==== Validation ====", args.model, "====", args.dataset_name)
    with torch.no_grad():
        for idx, input_tuple in enumerate(loader):
            data, target = prepare_input(input_tuple=input_tuple, args=args)
            logits = model(data)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc1 = acc_func.aggregate()
            avg_acc1 = acc1[0].detach().cpu().numpy()
            dice += float(avg_acc1)

            # Hausdorff distance
            hausd_func(y_pred=val_output_convert, y=val_labels_convert)
            acc3 = hausd_func.aggregate()
            avg_acc3 = acc3[0].detach().cpu().numpy()
            hausd += float(avg_acc3)

            loss = loss_func(logits, target)
            run_loss.update(loss.item(), n=args.batchSz)


        avg_acc1 = dice / len(loader)
        avg_acc3 = hausd / len(loader)
        print('Epoch: {}/{}'.format(epoch, args.nEpochs),
              ' DSC:', avg_acc1, 'HDD:', avg_acc3,
              'loss: {:.4f}'.format(run_loss.avg),
              )
    return avg_acc1, avg_acc3, run_loss.avg


def save_checkpoint(directory, model, epochs_since_improvement, epoch,
                    best_dice, hauds, is_best, optimizer=None, name=None):
    """
    Saves checkpoint at a certain global step during training.
    :param directory: path to save checkpoint
    :param model: training model to be saved
    :param epochs_since_improvement: epochs since improvement
    :param epoch: epoch
    :param best_dice: best dice similarity coefficient
    :param roc:
    :param auc:
    :param hauds: hausdorff distance
    :param is_best: is best so far?
    :param optimizer: optimizer
    :param name: model name
    :return: None
    """
    # Create directory to save to
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Build checkpoint dict to save.
    ckpt_dict = {
        'epoch':
            epoch,
        'best_dice':
            best_dice,
        'hauds':
            hauds,
        'epochs_since_improvement':
            epochs_since_improvement,
        'model_state_dict':
            model.state_dict(),
        'optimizer_state_dict':
            optimizer.state_dict() if optimizer is not None else None,
    }

    # Save the file with specific name
    if name is None:
        name = "{}_{}_epoch.pth".format(
            os.path.basename(directory),  # netD or netG
            'last')

    torch.save(ckpt_dict, os.path.join(directory, name))
    if is_best:
        name = "{}_BEST.pth".format(
            os.path.basename(directory))
        torch.save(ckpt_dict, os.path.join(directory, name))
