import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import ttest_rel, friedmanchisquare
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from medivision.model import MediVision
from medivision.metrics import calculate_metrics
from medivision.preprocessing import MedicalImagePreprocessor

class Benchmark:
    """
    Handles benchmarking of MediVision against baseline models.
    """
    def __init__(self, test_loader, device, config):
        """
        Initialize the Benchmark class.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            device (torch.device): Device to run evaluations on.
            config (dict): Configuration dictionary.
        """
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.baseline_models = {
            'VGG16': models.vgg16(pretrained=True),
            'VGG19': models.vgg19(pretrained=True),
            'ResNet50': models.resnet50(pretrained=True),
            'EfficientNet': models.efficientnet_b0(pretrained=True),
            'DenseNet': models.densenet121(pretrained=True),
            'Hybrid CNN-LSTM': None  # Placeholder for custom implementation
        }
        self.preprocessor = MedicalImagePreprocessor(input_size=config['input_size'])

    def evaluate_baseline(self, model_name):
        """
        Evaluate a baseline model on the test dataset.

        Args:
            model_name (str): Name of the baseline model.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        model = self.baseline_models[model_name].to(self.device)
        model.eval()
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = self.preprocessor(images).to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        return calculate_metrics(y_true, y_pred)

    def run_benchmark(self):
        """
        Run benchmarking against all baseline models.

        Returns:
            dict: Dictionary of results for each model.
        """
        results = {}
        for model_name in self.baseline_models:
            if model_name == 'Hybrid CNN-LSTM':
                # Skip placeholder
                continue
            results[model_name] = self.evaluate_baseline(model_name)
        return results

    def perform_statistical_tests(self, medi_results, baseline_results):
        """
        Perform statistical tests comparing MediVision to baselines.

        Args:
            medi_results (dict): MediVision evaluation results.
            baseline_results (dict): Baseline models evaluation results.

        Returns:
            dict: Dictionary of statistical test results.
        """
        test_results = {}
        for model_name in baseline_results:
            # Paired t-test
            t_stat, p_value = ttest_rel(
                medi_results['accuracy'],
                baseline_results[model_name]['accuracy']
            )
            test_results[model_name] = {
                't_stat': t_stat,
                'p_value': p_value
            }
        
        # Friedman test
        accuracy_scores = [medi_results['accuracy']] + \
                          [baseline_results[model_name]['accuracy'] for model_name in baseline_results]
        friedman_stat, friedman_p = friedmanchisquare(*accuracy_scores)
        test_results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p
        }
        
        return test_results

    def generate_report(self, medi_results, baseline_results, test_results):
        """
        Generate a benchmarking report.

        Args:
            medi_results (dict): MediVision evaluation results.
            baseline_results (dict): Baseline models evaluation results.
            test_results (dict): Statistical test results.

        Returns:
            pd.DataFrame: DataFrame summarizing the results.
        """
        report = []
        for model_name in baseline_results:
            report.append({
                'Model': model_name,
                'Accuracy': baseline_results[model_name]['accuracy'],
                'Precision': baseline_results[model_name]['precision'],
                'Recall': baseline_results[model_name]['recall'],
                'F1-Score': baseline_results[model_name]['f1_score'],
                't-statistic': test_results[model_name]['t_stat'],
                'p-value': test_results[model_name]['p_value']
            })
        
        report.append({
            'Model': 'MediVision',
            'Accuracy': medi_results['accuracy'],
            'Precision': medi_results['precision'],
            'Recall': medi_results['recall'],
            'F1-Score': medi_results['f1_score'],
            't-statistic': None,
            'p-value': None
        })
        
        return pd.DataFrame(report)


def main():
    """
    Example usage of the Benchmark class.
    """
    # Example configuration
    config = {
        'input_size': 224,
        'num_classes': 10
    }
    
    # Example test loader (replace with actual DataLoader)
    test_loader = None
    
    # Initialize benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = Benchmark(test_loader, device, config)
    
    # Run benchmarking
    baseline_results = benchmark.run_benchmark()
    
    # Evaluate MediVision
    medi_model = MediVision(input_channels=3, num_classes=config['num_classes']).to(device)
    medi_results = benchmark.evaluate_baseline('MediVision')
    
    # Perform statistical tests
    test_results = benchmark.perform_statistical_tests(medi_results, baseline_results)
    
    # Generate report
    report = benchmark.generate_report(medi_results, baseline_results, test_results)
    print(report)
    
    # Save report to CSV
    report.to_csv('benchmark_results.csv', index=False)


if __name__ == '__main__':
    main()