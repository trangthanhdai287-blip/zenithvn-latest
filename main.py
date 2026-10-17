# main.py
import torch
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.paths_config import PathsConfig
from models.transformer import ZenithVNModel
from data.dataset import TextDataset
from engine.trainer import Trainer

def main():
    # 1. Tạo các thư mục cần thiết
    PathsConfig.create_dirs()
    
    print(f"🧠 Khởi tạo mô hình ngôn ngữ: {ModelConfig.MODEL_NAME}")
    
    # 2. Khởi tạo mô hình từ cấu hình
    model = ZenithVNModel(ModelConfig)
    
    # 3. Tạo dữ liệu giả lập để kiểm tra (Hoặc bạn thay bằng đọc file raw_data.txt thực tế)
    dummy_tokens = [i % ModelConfig.VOCAB_SIZE for i in range(10000)]
    dataset = TextDataset(dummy_tokens, ModelConfig.MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True)
    
    # 4. Khởi tạo Trainer và bắt đầu train
    trainer = Trainer(model, dataloader, ModelConfig, TrainConfig)
    trainer.train()
    
    # 5. Lưu trọng số sau khi train xong
    save_path = f"{ModelConfig.MODEL_NAME}_checkpoint.pt"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Đã lưu trọng số mô hình vào: {save_path}")

if __name__ == "__main__":
    main()