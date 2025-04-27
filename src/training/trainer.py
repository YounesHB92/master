import os
import pickle
from datetime import datetime

import torch
from tqdm import tqdm
import pytz
from dotenv import load_dotenv
load_dotenv()



class Trainer(object):
    def __init__(self, model, train_loader, val_loader, augment, loss_function, optimizer, scheduler, device, output_dir,
                 num_epochs, details):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augment = augment
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.details = details
        self.create_output_dir()

    def create_output_dir(self):
        self.results = {}
        self.results["date"] = datetime.now(tz=pytz.timezone("Australia/Brisbane")).strftime("%Y%m%d-%H%M%S")
        self.results["details"] = self.details
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.sub_dir = os.path.join(self.output_dir, self.model.encoder_name)
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir, exist_ok=True)

    def run_training(self):
        for name_ in ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]:
            self.results[name_] = []
        for epoch in range(self.num_epochs):
            self.results["epoch"].append(epoch)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track training loss and accuracy
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = 100. * correct / total
            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            print(f"Train Loss: {train_loss / len(self.train_loader):.4f}, Train Acc: {train_acc:.2f}%")

            # Validation phase
            self.model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)

                    # Track validation loss and accuracy
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            val_acc = 100. * correct / total
            self.results["val_loss"].append(val_loss)
            self.results["val_acc"].append(val_acc)
            print(f"Val Loss: {val_loss / len(self.val_loader):.4f}, Val Acc: {val_acc:.2f}%")

            self.save_model(epoch)

            # Step the scheduler
            self.scheduler.step()

        return self

    def save_model(self, epoch):
        model_name = f"encoder_{self.model.encoder_name}_epoch_{epoch}.pth"
        model_path = os.path.join(self.sub_dir, model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        results_name = f"encoder_{self.model.encoder_name}_results_epoch_{epoch}.pickle"
        results_path = os.path.join(self.sub_dir, results_name)
        with open(results_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {results_path}")

