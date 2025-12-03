"""
MediVision - Hybrid CNN-LSTM-Attention Model for Medical Image Analysis

This package implements the MediVision architecture described in:
"A Hybrid Convolutional Neural Network–Long Short-Term Memory (CNN–LSTM)–Attention Model Architecture
for Precise Medical Image Analysis and Disease Diagnosis"

Modules:
- cnn_unit: CNN feature extraction layers
- lstm_attention: LSTM with attention mechanism
- model: Main MediVision architecture
- preprocessing: Data augmentation and normalization
- trainer: Training pipeline
- metrics: Evaluation metrics
- grad_cam: Interpretability via Grad-CAM
"""

from .cnn_unit import CNNFeatureExtractor
from .lstm_attention import LSTMAttention
from .model import MediVision
from .preprocessing import MedicalImagePreprocessor
from .trainer import Trainer
from .metrics import calculate_metrics
from .grad_cam import GradCAM

__all__ = [
    'CNNFeatureExtractor',
    'LSTMAttention',
    'MediVision',
    'MedicalImagePreprocessor',
    'Trainer',
    'calculate_metrics',
    'GradCAM'
]