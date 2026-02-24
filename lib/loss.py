import torch.nn as nn
import segmentation_models_pytorch as smp


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.loss = smp.losses.TverskyLoss(
            mode='binary', alpha=alpha, beta=beta, from_logits=True
        )

    def forward(self, pred, target):
        return self.loss(pred, target.float())
