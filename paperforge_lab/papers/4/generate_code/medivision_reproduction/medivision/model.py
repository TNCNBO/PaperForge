import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_unit import CNNFeatureExtractor
from .lstm_attention import LSTMAttention

class MediVision(nn.Module):
    """
    Hybrid CNN-LSTM-Attention model for medical image classification.
    Integrates CNN feature extraction, LSTM sequential processing, attention mechanism, and skip connections.
    """
    def __init__(self, input_channels=3, num_classes=10):
        super(MediVision, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(input_channels=input_channels)
        
        # Flattening and Reshaping Unit
        self.flatten = nn.Flatten()
        
        # LSTM Sequential Processor with Attention
        self.lstm_attention = LSTMAttention(input_dim=512, hidden_dim=256, dropout=0.2)
        
        # Skip Connection
        self.skip_connection = nn.Linear(512 + 256, 512)  # Combines CNN and LSTM outputs
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # CNN Feature Extraction
        cnn_features = self.cnn(x)
        
        # Flatten and Reshape for LSTM
        batch_size = x.size(0)
        flattened = self.flatten(cnn_features)
        reshaped = flattened.view(batch_size, 1, -1)
        
        # LSTM with Attention
        context_vector, _ = self.lstm_attention(reshaped)
        
        # Skip Connection: Combine CNN and LSTM outputs
        combined = torch.cat([cnn_features.view(batch_size, -1), context_vector], dim=1)
        skip_output = self.skip_connection(combined)
        
        # Classification
        output = self.classifier(skip_output)
        
        return output