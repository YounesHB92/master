import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_metrics, train_loader, val_loader=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_metrics = loss_metrics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_metrics = self._run_one_epoch(self.train_loader, mode='train')
            print(f"\tTrain Loss: {train_loss:.4f}")
            self._print_metrics(train_metrics, indent_level=2)

            if self.val_loader:
                val_loss, val_metrics = self._run_one_epoch(self.val_loader, mode='val')
                print(f"\tVal Loss: {val_loss:.4f}")
                self._print_metrics(val_metrics, indent_level=2)

    def _run_one_epoch(self, loader, mode="train"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        is_train = mode == "train"
        total_loss = 0.0
        all_metrics = []

        loop = tqdm(loader, desc=f"{mode.capitalize()} Loop", leave=False)

        for images, masks in loop:
            images, masks = images.to(self.device), masks.to(self.device)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
                loss = self.loss_metrics.compute_loss(outputs, masks)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                metrics = self.loss_metrics.compute_metrics(outputs.detach(), masks.detach())
                all_metrics.append(metrics)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
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

        # Print overall mIoU and mDice
        mean_iou = sum(total_iou) / len(total_iou)
        mean_dice = sum(total_dice) / len(total_dice)
        print(f"{indent}Mean IoU: {mean_iou:.4f}")
        print(f"{indent}Mean Dice: {mean_dice:.4f}")
