# models/embeddings.py
import math
import torch
import torch.nn as nn

class TokenAndPositionalEmbedding(nn.Module):
    """Kết hợp Token Embedding và Positional Encoding cho ZenithVN"""
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Tạo ma trận Positional Encoding cố định
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_seq_len, d_model)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # Cộng token embedding với positional encoding tương ứng với chiều dài chuỗi
        embeddings = self.token_embedding(x) + self.pe[:, :seq_len, :]
        return self.dropout(embeddings)