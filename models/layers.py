# models/layers.py
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Mạng chuyển tiếp phi tuyến tính (FFN)"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerDecoderBlock(nn.Module):
    """Một khối Decoder hoàn chỉnh gồm Attention + FFN + Residual + LayerNorm"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, eps=1e-5):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention với Pre-LayerNorm và Residual Connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        
        # Feed-forward với Pre-LayerNorm và Residual Connection
        x = x + self.ffn(self.norm2(x))
        return x