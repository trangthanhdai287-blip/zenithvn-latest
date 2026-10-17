# models/transformer.py
import torch
import torch.nn as nn
from models.embeddings import TokenAndPositionalEmbedding
from models.layers import TransformerDecoderBlock

class ZenithVNModel(nn.Module):
    """Kiến trúc mô hình ngôn ngữ chính của ZenithVN"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding lớp
        self.embedding = TokenAndPositionalEmbedding(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT
        )
        
        # Danh sách các khối Decoder chồng lên nhau
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=config.D_MODEL,
                n_heads=config.N_HEADS,
                d_ff=config.D_FF,
                dropout=config.DROPOUT,
                eps=config.EPS
            ) for _ in range(config.N_LAYERS)
        ])
        
        # Lớp chuẩn hóa cuối cùng và đầu ra dự đoán từ (LM Head)
        self.norm = nn.LayerNorm(config.D_MODEL, eps=config.EPS)
        self.lm_head = nn.Linear(config.D_MODEL, config.VOCAB_SIZE, bias=False)
        
        # Chia sẻ trọng số giữa embedding và lm_head để tiết kiệm tham số
        self.lm_head.weight = self.embedding.token_embedding.weight

    def _generate_causal_mask(self, sz, device):
        # Tạo mặt nạ tam giác trên để ngăn không cho nhìn thấy token tương lai
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        device = idx.device
        
        # Tạo mask nhân quả
        mask = self._generate_causal_mask(seq_len, device)
        
        # Lấy embedding
        x = self.embedding(idx)
        
        # Cho qua các lớp Decoder
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits