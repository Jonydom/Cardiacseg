import math
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from utils.rmi_loss import RMILoss_3D


class LossFunction():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.sigmoid_rmi = args.sigmoid_rmi
        self.rmi_epoch = args.rmi_epoch
        self.ds_method = args.rmi_ds
        self.stride = args.rmi_stride
        self.radius = args.rmi_radius

    def sigmoid(self, epoch):
        s = 1 / (1 + math.exp(self.rmi_epoch - epoch))
        return s
    
    def delta(self, epoch):
        if epoch <= self.rmi_epoch:
            s = 0
        else:
            s = 1
        return s

    def __call__(self, pred, gt, epoch):
        dc_loss = self.Dice(pred, gt)
        ce_loss = self.CrossEntropy(pred, gt)
        if self.sigmoid_rmi:
            rmi_loss = self.sigmoid(epoch) * self.RMI(pred, gt, self.num_classes, downsampling_method=self.ds_method, stride=self.stride, radius=self.radius)
        else:
            rmi_loss = self.delta(epoch) * self.RMI(pred, gt, self.num_classes, downsampling_method=self.ds_method, stride=self.stride, radius=self.radius)
            
        return (dc_loss, ce_loss, rmi_loss)

    def Dice(self, x, y):
        dc_loss = DiceLoss(to_onehot_y=True, softmax=True)(x, y)
        return dc_loss

    def CrossEntropy(self, x, y):
        y = y.squeeze(dim=1).long()
        ce_loss = nn.CrossEntropyLoss()(x, y)
        return ce_loss

    def RMI(self, pred, target, num_classes, downsampling_method='avg', stride=2, radius=3):
        target = F.one_hot(target.squeeze(dim=1).long(), num_classes=num_classes).permute(0,4,1,2,3).float()
        rmi_loss = RMILoss_3D(
                                with_logits=True, downsampling_method=downsampling_method, bce_weight=0,
                                stride=stride, radius=radius,
                            )(pred, target)
        return rmi_loss