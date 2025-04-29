import segmentation_models_pytorch as smp
from torch import nn

class LossAndMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.dice_loss_fn = smp.losses.DiceLoss(mode="multiclass")
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, preds, targets):
        dice_loss = self.dice_loss_fn(preds, targets)
        ce_loss = self.ce_loss_fn(preds, targets)
        total_loss = dice_loss + ce_loss
        return total_loss

    def compute_metrics(self, preds, targets):
        preds_classes = preds.argmax(dim=1)
        targets = targets.long()

        metrics = {}
        eps = 1e-7

        for cls in range(self.num_classes):
            pred_mask = (preds_classes == cls).float()
            target_mask = (targets == cls).float()

            tp = (pred_mask * target_mask).sum().item()
            fp = (pred_mask * (1 - target_mask)).sum().item()
            fn = ((1 - pred_mask) * target_mask).sum().item()
            tn = ((1 - pred_mask) * (1 - target_mask)).sum().item()

            iou = tp / (tp + fp + fn + eps)
            dice = (2 * tp) / (2 * tp + fp + fn + eps)

            metrics[cls] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "IoU": iou,
                "Dice": dice,
            }
        return metrics
