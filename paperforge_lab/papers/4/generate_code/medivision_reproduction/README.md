# MediVision Reproduction

This repository contains a reproduction of the hybrid CNN-LSTM-Attention model architecture described in the paper:

**Title**: "A Hybrid Convolutional Neural Network–Long Short-Term Memory (CNN–LSTM)–Attention Model Architecture for Precise Medical Image Analysis and Disease Diagnosis"

**Core Contribution**: MediVision - a hybrid CNN-LSTM-Attention model with skip connections and Grad-CAM interpretability for medical image classification across 10 diverse datasets.

## File Structure

The repository follows the following structure:

```
medivision_reproduction/
│
├── medivision/
│   ├── __init__.py
│   ├── model.py                   # Main MediVision architecture
│   ├── cnn_unit.py                # CNN feature extractor
│   ├── lstm_attention.py          # LSTM + Attention mechanism
│   ├── preprocessing.py           # Data augmentation and normalization
│   ├── trainer.py                 # Training pipeline
│   ├── grad_cam.py                # Interpretability
│   └── metrics.py                 # Evaluation metrics
│
├── scripts/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Model evaluation
│   ├── visualize.py               # Grad-CAM visualization
│   └── benchmark.py               # Baseline comparisons
│
├── config/
│   ├── model_config.yaml          # Architecture parameters
│   └── dataset_config.yaml        # Dataset-specific settings
│
├── data/                          # 10 medical datasets
│   ├── alzheimer/
│   ├── breast_ultrasound/
│   ├── blood_cell/
│   └── ... (8 more)
│
├── results/                       # Output directory
│   ├── models/
│   ├── plots/
│   └── metrics/
│
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

To install the required dependencies, run:

```bash
pip install torch torchvision albumentations opencv-python scikit-learn numpy pandas matplotlib seaborn pillow scikit-image
```

## Usage

### Training

To train the MediVision model, use the following command:

```bash
python scripts/train.py --config config/model_config.yaml --dataset_config config/dataset_config.yaml
```

### Evaluation

To evaluate the trained model, use:

```bash
python scripts/evaluate.py --model_path results/models/best_model.pth --dataset_config config/dataset_config.yaml
```

### Visualization

To generate Grad-CAM visualizations, use:

```bash
python scripts/visualize.py --model_path results/models/best_model.pth --image_path data/alzheimer/sample.jpg --target_layer cnn_unit.layer4
```

### Benchmarking

To compare MediVision against baseline models, use:

```bash
python scripts/benchmark.py --dataset_config config/dataset_config.yaml
```

## Results

The repository includes scripts to reproduce the results reported in the paper:

- Accuracy, Precision, Recall, and F1-Score metrics
- Confusion matrices
- Grad-CAM visualizations
- Statistical significance tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
