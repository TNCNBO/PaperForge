import argparse
import os
import torch
from torch.utils.data import DataLoader

from medivision.model import MediVision
from medivision.trainer import Trainer
from medivision.preprocessing import MedicalImagePreprocessor
from medivision.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MediVision model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    parser.add_argument('--save_dir', type=str, default='./results/models', help='Directory to save trained models')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize model
    model = MediVision(num_classes=args.num_classes).to(args.device)

    # Initialize data preprocessing
    preprocessor = MedicalImagePreprocessor()

    # TODO: Implement dataset loading and splitting
    # train_dataset = ...
    # val_dataset = ...
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        config={
            'lr': args.lr,
            'epochs': args.epochs,
            'save_dir': args.save_dir,
            'num_classes': args.num_classes
        }
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()