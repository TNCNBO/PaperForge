import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from medivision.model import MediVision
from medivision.metrics import calculate_metrics
from medivision.preprocessing import MedicalImagePreprocessor

class Evaluator:
    """
    Handles the evaluation pipeline for MediVision model.
    """
    def __init__(self, model, test_loader, device, config):
        """
        Initialize the Evaluator.

        Args:
            model (MediVision): Trained MediVision model.
            test_loader (DataLoader): DataLoader for test dataset.
            device (torch.device): Device to run evaluation on.
            config (dict): Configuration dictionary.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.config = config

    def evaluate(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
        return metrics

def main():
    """
    Main function to run evaluation.
    """
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "model_path": "results/models/best_model.pth",
        "input_size": 224,
        "num_classes": 10
    }

    # Load model
    model = MediVision(input_channels=3, num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(config["model_path"]))

    # Load test dataset
    # Replace with actual test dataset loader
    test_loader = None

    # Initialize evaluator
    evaluator = Evaluator(model, test_loader, device, config)
    metrics = evaluator.evaluate()

    print(f"Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()