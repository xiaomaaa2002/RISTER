import torch.nn.functional as F
from torch import nn

import torch
import torch.nn.functional as F


class ARLossPrompt(nn.Module):

    def __init__(self, label_smoothing=0.1, ignore_index=0, **kwargs):
        super(ARLossPrompt, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, batch):
        alpha = 0.7
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]
        tgt = tgt.reshape([-1])
        
        logit1 = pred[0]
        logit1 = logit1.flatten(0, 1)
        loss1 = F.cross_entropy(
            logit1,
            tgt,
            reduction='mean',
            label_smoothing=self.label_smoothing,
            ignore_index=logit1.shape[1] + 1,
        )  # self.loss_func(pred, tgt)
        
        logit2 = pred[1]
        logit2 = logit2.flatten(0, 1)
        loss2 = F.cross_entropy(
            logit2,
            tgt,
            reduction='mean',
            label_smoothing=self.label_smoothing,
            ignore_index=logit2.shape[1] + 1,
        )  # self.loss_func(pred, tgt)
        
        loss = alpha * loss1 + (1 - alpha) * loss2
        return {'loss': loss}
