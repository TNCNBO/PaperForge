import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttention(nn.Module):
    """
    LSTM with Attention Mechanism for processing sequential features.
    
    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden state (default: 256).
        dropout (float): Dropout rate (default: 0.2).
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Attention Mechanism
        self.attention_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.attention_linear2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Forward pass for LSTM with Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dim).
            torch.Tensor: Attention weights of shape (batch_size, seq_len).
        """
        # LSTM Processing
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Attention Mechanism
        attention_scores = F.tanh(self.attention_linear1(lstm_out))  # Shape: (batch_size, seq_len, hidden_dim)
        attention_scores = self.attention_linear2(attention_scores).squeeze(-1)  # Shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_len)
        
        # Weighted Sum
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)  # Shape: (batch_size, hidden_dim)
        
        return context_vector, attention_weights