import torch
import os
import csv
from tqdm import tqdm
from src.utils import print_indented, load_env_variables, logs
_ = load_env_variables()

class CnnTrainer:
    def __init__(self, model, train_dataset, val_dataset, loss_and_metrics, config_name,
                 epochs=20, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_dataset.loader
        self.val_loader = val_dataset.loader
        self.class_names = train_dataset.class_names
        self.loss_and_metrics = loss_and_metrics
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.config_name = config_name

        self.best_f1 = -1
        self.checkpoint_path = os.path.join(os.getenv("CHECKPOINTS_DIR"), "cnn", f"{self.config_name}.pt")

        self.log_path = os.path.join(os.getenv("LOGS_DIR"), "cnn", f"{self.config_name}.csv")
        logs.init_cnn_logfile(log_path=self.log_path, class_names=self.class_names)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            total_samples = 0
            all_preds, all_targets = [], []

            loop = tqdm(self.train_loader, desc=f"Training {epoch+1}/{self.epochs}", leave=False)
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_and_metrics.compute_loss(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
                all_preds.append(outputs.detach())
                all_targets.append(labels.detach())

                loop.set_postfix({"loss": f"{loss.item():.4f}"})

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            train_metrics = self.loss_and_metrics.compute_metrics(all_preds, all_targets)
            avg_train = self.loss_and_metrics.compute_average_metrics(train_metrics)
            avg_loss = epoch_loss / total_samples

            print(f"\nTrain Loss: {avg_loss:.4f} | Avg F1: {avg_train['avg_F1-Score']:.4f} | Accuracy: {avg_train['avg_Accuracy']:.4f}")
            self.print_metrics(train_metrics)

            val_loss, avg_val, val_metrics = self.evaluate(epoch)

            # Save logs
            row = [
                epoch + 1,
                avg_loss,
                val_loss,
                avg_train["avg_F1-Score"],
                avg_val["avg_F1-Score"],
                avg_train["avg_Accuracy"],
                avg_val["avg_Accuracy"]
            ]

            # Add per-class metrics to row
            for cls in self.class_names:
                idx = self.train_loader.dataset.class_to_idx[cls]
                row.append(train_metrics[idx]["F1-Score"])
                row.append(train_metrics[idx]["Accuracy"])

            for cls in self.class_names:
                idx = self.train_loader.dataset.class_to_idx[cls]
                row.append(val_metrics[idx]["F1-Score"])
                row.append(val_metrics[idx]["Accuracy"])
            logs.append_cnn_logfile(self.log_path, row)

            # Save best model
            if avg_val["avg_F1-Score"] > self.best_f1:
                self.best_f1 = avg_val["avg_F1-Score"]
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"\n✅ Best model saved at {self.checkpoint_path} (F1={self.best_f1:.4f})")

    def evaluate(self, epoch):
        self.model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Validating {epoch+1}/{self.epochs}", leave=False)
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_and_metrics.compute_loss(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                total_samples += batch_size
                all_preds.append(outputs)
                all_targets.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_metrics = self.loss_and_metrics.compute_metrics(all_preds, all_targets)
        avg_val = self.loss_and_metrics.compute_average_metrics(val_metrics)
        avg_loss = val_loss / total_samples

        print(f"\nValidation Loss: {avg_loss:.4f} | Avg F1: {avg_val['avg_F1-Score']:.4f} | Accuracy: {avg_val['avg_Accuracy']:.4f}")
        self.print_metrics(val_metrics)

        return avg_loss, avg_val, val_metrics

    def print_metrics(self, metrics_dict):
        print("-" * 50)
        for cls, metrics in metrics_dict.items():
            label = list(self.train_loader.dataset.class_to_idx.keys())[cls]
            print_indented(f"Class -> {label}:", level=0)
            print_indented(f"F1-Score: {metrics['F1-Score']}", level=1)
            print_indented(f"Accuracy: {metrics['Accuracy']}", level=1)
        print("-" * 50)
