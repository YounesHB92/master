import torch
from tqdm import tqdm

class EpochRunner:
    def __init__(self, model, loss_metrics, device):
        self.model = model.to(device)
        self.loss_metrics = loss_metrics
        self.device = device

    def _run_one_epoch(self, dataset_loader, train=False, optimizer=None):

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_metrics = []

        mode = "train" if train else "val-test"
        loop = tqdm(dataset_loader, desc=f"{mode.upper()} Loop", leave=False)

        for images, masks in loop:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.set_grad_enabled(train):
                outputs = self.model(images)
                loss = self.loss_metrics.compute_loss(outputs, masks)
                if train:
                    if optimizer is None:
                        raise ValueError("Optimizer must be provided for training.")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                metrics = self.loss_metrics.compute_metrics(outputs.detach(), masks.detach())
                all_metrics.append(metrics)

        avg_loss = total_loss / len(dataset_loader)
        avg_metrics = self._average_metrics(all_metrics)
        return avg_loss, avg_metrics

    def _average_metrics(self, metrics_list):
        summary = {}
        keys = metrics_list[0].keys()
        for k in keys:
            summary[k] = {}
            for metric_name in metrics_list[0][k]:
                summary[k][metric_name] = sum(d[k][metric_name] for d in metrics_list) / len(metrics_list)
        return summary

    def _print_metrics(self, metrics, indent_level=1):
        indent = "\t" * indent_level
        total_iou = []
        total_dice = []

        for cls, m in metrics.items():
            print(f"{indent}Class {cls}:")
            for key, val in m.items():
                print(f"{indent}\t{key}: {val:.4f}")

            total_iou.append(m["IoU"])
            total_dice.append(m["Dice"])

        mean_iou = sum(total_iou) / len(total_iou)
        mean_dice = sum(total_dice) / len(total_dice)
        print(f"{indent}Mean IoU: {mean_iou:.4f}")
        print(f"{indent}Mean Dice: {mean_dice:.4f}")

        return mean_iou, mean_dice