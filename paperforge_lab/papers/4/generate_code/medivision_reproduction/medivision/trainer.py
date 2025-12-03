import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model import MediVision
from .metrics import calculate_metrics
import numpy as np
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to train on (CPU/GPU).
            config (dict): Configuration dictionary containing training parameters.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=config["patience"], verbose=True)

        # Training metrics
        self.best_val_accuracy = 0.0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        epoch_loss /= len(self.train_loader)
        epoch_accuracy = 100.0 * correct / total

        self.train_loss_history.append(epoch_loss)
        self.train_accuracy_history.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def validate_epoch(self):
        """Validate the model for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                # Update metrics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        epoch_loss /= len(self.val_loader)
        epoch_accuracy = 100.0 * correct / total

        self.val_loss_history.append(epoch_loss)
        self.val_accuracy_history.appendepoch_accuracy)

        return epoch_loss, epoch_accuracy

    def train(self, num_epochs):
        """Train the model for a specified number of epochs."""
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Update learning rate scheduler
            self.scheduler.step(val_accuracy)

            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), os.path.join(self.config["save_dir"], "best_model.pth"))

        return {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history,
            "train_accuracy": self.train_accuracy_history,
            "val_accuracy": self.val_accuracy_history,
        }

    def evaluate(self, test_loader):
        """Evaluate the model on the test set."""
        self.model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        metrics = calculate_metrics(all_targets, all_predictions)
        return metrics