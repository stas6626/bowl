import cv2
import numpy as np
import torch
from skimage.morphology import label
from torch import nn
from scipy import ndimage as ndi
import torch.nn.functional as F

class LossBinary:
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def default_loss_composition(loss1, loss2, loss3):
    return loss1 + loss2 + loss3


def iou_t(target, pred):
    target_y = label(target > 0.5)
    pred_y = label(pred > 0.5)
    true_objs = len(np.unique(target_y))
    pred_objs = len(np.unique(pred_y))
    intersection = np.histogram2d(target_y.flatten(),
                                pred_y.flatten(),
                                bins=(true_objs, pred_objs))[0]
  
    area_true = np.histogram(target_y, bins = true_objs)[0]
    area_pred = np.histogram(pred_y, bins = pred_objs)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)