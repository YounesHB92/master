import torch
from tqdm import tqdm

from src.utils import print_indented


class CnnTrainer:
    """
    CNN trainer for crack classification.

    Args:
        model (nn.Module): CNN model.
        train_dataset (Dataset): Training set.
        val_dataset (Dataset): Validation set.
        loss_and_metrics (object): Loss and metrics object.
        epochs (int): Epochs to train.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'
    """

    def __init__(self, model, train_dataset, val_dataset, loss_and_metrics, epochs=20, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.loss_and_metrics = loss_and_metrics
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            all_preds, all_targets = [], []

            loop = tqdm(self.train_loader, desc=f"Training, epoch: {epoch+1}/{self.epochs}", leave=False)

            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_and_metrics.compute_loss(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                all_preds.append(outputs.detach())
                all_targets.append(labels.detach())

                loop.set_postfix({"loss": f"{loss.item():.4f}"})

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            print(f"\n\nTrain Loss: {epoch_loss:.4f}")
            train_metrics = self.loss_and_metrics.compute_metrics(all_preds, all_targets)
            avg_metrics = self.loss_and_metrics.compute_average_metrics(train_metrics)
            print(f"Train F1-Score: {avg_metrics['avg_F1-Score']:.4f}")
            self.print_metrics(train_metrics)

            self.evaluate(epoch=epoch)

    def evaluate(self, epoch):
        self.model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Validating, epoch: {epoch+1}/{self.epochs}", leave=False)
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_and_metrics.compute_loss(outputs, labels)

                val_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        print(f"\nValidation Loss: {val_loss:.4f}")
        val_metrics = self.loss_and_metrics.compute_metrics(all_preds, all_targets)
        avg_metrics = self.loss_and_metrics.compute_average_metrics(val_metrics)
        print(f"Validation F1-Score: {avg_metrics['avg_F1-Score']:.4f}")
        self.print_metrics(val_metrics)

    def print_metrics(self, metrics_dict):
        print("-"*50)
        for cls, metrics in metrics_dict.items():
            label = list(self.train_loader.dataset.class_to_idx.keys())[cls]
            print_indented(f"Class -> {label}:", level=0)
            print_indented(f"F1-Score: {metrics['F1-Score']}", level=1)
        print("-"*50)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
