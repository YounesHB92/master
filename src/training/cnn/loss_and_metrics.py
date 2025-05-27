import torch.nn as nn
import numpy as np


class CnnLossAndMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, preds, targets):
        return self.ce_loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        """
        Computes TP, FP, FN, TN, precision, recall, F1 for each class.

        Args:
            preds (Tensor): shape [B, C]
            targets (Tensor): shape [B]

        Returns:
            metrics (dict): {class_idx: {...}, ...}
        """
        pred_classes = preds.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        metrics = {}
        for cls in range(self.num_classes):
            tp = np.sum((pred_classes == cls) & (targets == cls))
            fp = np.sum((pred_classes == cls) & (targets != cls))
            fn = np.sum((pred_classes != cls) & (targets == cls))
            tn = np.sum((pred_classes != cls) & (targets != cls))

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = (2 * precision * recall) / (precision + recall + 1e-7)

            metrics[cls] = {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4)
            }

        return metrics

    def compute_average_metrics(self, metrics_dict):
        """
        Computes the macro average of all numeric metrics across all classes.

        Args:
            metrics_dict (dict): Output from compute_metrics()

        Returns:
            avg_metrics (dict): {metric_name: average_value}
        """
        keys = ["TP", "FP", "FN", "TN", "Precision", "Recall", "F1-Score"]
        avg_metrics = {}

        for key in keys:
            values = [metrics_dict[cls][key] for cls in metrics_dict]
            avg_metrics[f"avg_{key}"] = round(np.mean(values), 4)

        return avg_metrics
