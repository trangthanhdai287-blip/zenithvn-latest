# configs/train_config.py

class TrainConfig:
    """Cấu hình huấn luyện cho ZenithVN"""
    BATCH_SIZE = 8           # Kích thước batch mỗi lần đưa vào GPU/CPU
    LEARNING_RATE = 3e-4     # Tốc độ học ban đầu (Learning rate)
    EPOCHS = 5               # Số vòng lặp huấn luyện toàn bộ dữ liệu
    WARMUP_STEPS = 1000      # Số bước làm ấm learning rate
    WEIGHT_DECAY = 0.01      # Hệ số phạt chống overfitting
    GRAD_CLIP = 1.0          # Ngưỡng cắt gradient tránh bùng nổ gradient
    SAVE_EVERY_STEPS = 500   # Lưu checkpoint sau mỗi 500 bước