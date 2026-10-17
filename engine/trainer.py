# engine/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class Trainer:
    """Quản lý quá trình huấn luyện mô hình ZenithVN"""
    def __init__(self, model, dataloader, config, train_config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.train_config = train_config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.LEARNING_RATE,
            weight_decay=train_config.WEIGHT_DECAY
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        print(f"🚀 Bắt đầu huấn luyện ZenithVN trên thiết bị: {self.device}")
        
        for epoch in range(self.train_config.EPOCHS):
            total_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.train_config.EPOCHS}")
            
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                
                # 1. Forward pass
                logits = self.model(x)
                
                # 2. Tính Loss (đưa về dạng 2D cho CrossEntropyLoss)
                loss = self.criterion(logits.view(-1, self.config.VOCAB_SIZE), y.view(-1))
                
                # 3. Backward pass & tối ưu hóa
                self.optimizer.zero_grad()
                loss.backward()
                
                # Cắt gradient tránh bùng nổ
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.GRAD_CLIP)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(self.dataloader)
            print(f"✅ Hoàn thành Epoch {epoch+1} | Loss trung bình: {avg_loss:.4f}")