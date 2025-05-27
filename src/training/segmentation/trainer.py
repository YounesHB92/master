import torch
import os
from src.utils import load_env_variables
from src.training.segmentation import EpochRunner
_ = load_env_variables()
import csv

class SegmentationTrainer(EpochRunner):
    def __init__(self, model, loss_metrics, device, train_loader, optimizer, val_loader, epochs, config_name):
        super().__init__(model, loss_metrics, device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.epochs = epochs
        self.config_name = config_name

        self.best_score = -1  # For tracking best val Mean IoU
        self.checkpoint_dir = os.path.join(os.getenv("CHECKPOINTS_DIR"), self.config_name + ".pt")

        # Create CSV log file
        self.log_dir = os.path.join(os.getenv("LOGS_DIR"), self.config_name + ".csv")
        with open(self.log_dir, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_mean_iou", "val_mean_iou", "train_mean_dice", "val_mean_dice"])

    def train(self):

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_loss, train_metrics = self._run_one_epoch(self.train_loader, train=True, optimizer=self.optimizer)
            print(f"\tTrain Loss: {train_loss:.4f}")
            train_miou, train_mdice = self._print_metrics(train_metrics, indent_level=2)

            val_loss, val_metrics = self._run_one_epoch(self.val_loader, train=False)
            print(f"\tVal Loss: {val_loss:.4f}")
            val_miou, val_mdice = self._print_metrics(val_metrics, indent_level=2)

            self.save_logs(epoch, train_loss, val_loss, train_miou, val_miou, train_mdice, val_mdice)

            # Save best model
            if val_miou > self.best_score:
                self.best_score = val_miou
                torch.save(self.model.state_dict(), self.checkpoint_dir)
                print(f"\n✅ Saved best model at {self.checkpoint_dir} with mIoU={val_miou:.4f}")

    def save_logs(self, epoch, train_loss, val_loss, train_miou, val_miou, train_mdice, val_mdice):
        with open(self.log_dir, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_miou, val_miou, train_mdice, val_mdice])
