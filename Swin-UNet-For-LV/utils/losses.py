import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):


    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        pred_sum = probs.sum(dim=1)
        target_sum = targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask_available: torch.Tensor) -> torch.Tensor:
        valid_logits = logits[mask_available]
        valid_targets = targets[mask_available]

        if valid_logits.numel() == 0:
            return 0.0 * logits.sum()

        bce = self.bce_loss(valid_logits, valid_targets)
        dice = self.dice_loss(valid_logits, valid_targets)
        return self.bce_weight * bce + self.dice_weight * dice



class EndToEndLoss(nn.Module):

    def __init__(self, seg_bce_weight=0.5, seg_dice_weight=0.5, ef_loss_weight=0.1):
        super().__init__()
        self.seg_loss = CombinedLoss(bce_weight=seg_bce_weight, dice_weight=seg_dice_weight)
        self.ef_loss = nn.MSELoss()
        self.ef_loss_weight = ef_loss_weight

    def forward(self, seg_logits, ef_pred, seg_targets, ef_target, mask_available):

        segmentation_loss = self.seg_loss(seg_logits, seg_targets, mask_available)


        ef_regression_loss = self.ef_loss(ef_pred, ef_target)


        total_loss = segmentation_loss + self.ef_loss_weight * ef_regression_loss

        return total_loss, segmentation_loss, ef_regression_loss
