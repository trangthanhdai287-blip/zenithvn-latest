# data/dataset.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Dataset tùy chỉnh để nạp dữ liệu văn bản cho ZenithVN"""
    def __init__(self, text_tokens, max_seq_len):
        self.tokens = text_tokens
        self.max_seq_len = max_seq_len

    def __len__(self):
        # Số lượng mẫu có thể tạo ra từ chuỗi token
        return max(0, len(self.tokens) - self.max_seq_len)

    def __getitem__(self, idx):
        # Cắt một đoạn chuỗi có độ dài bằng max_seq_len + 1
        chunk = self.tokens[idx : idx + self.max_seq_len + 1]
        
        # Input (x) và Target (y dịch sang phải 1 token để dự đoán từ tiếp theo)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y