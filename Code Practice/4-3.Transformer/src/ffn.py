import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    位置前馈网络
    标准 Transformer 结构: Linear -> ReLU -> Linear
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        return self.w2(self.dropout(F.relu(self.w1(x))))