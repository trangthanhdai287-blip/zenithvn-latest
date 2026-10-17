# configs/model_config.py

class ModelConfig:
    """Cấu hình kiến trúc mô hình ZenithVN"""
    MODEL_NAME = "zenithvn"
    
    # Các thông số kích thước mạng
    VOCAB_SIZE = 32000     # Kích thước từ điển (Vocabulary size)
    D_MODEL = 512          # Chiều vector ẩn (Hidden dimension / Embedding size)
    N_LAYERS = 6           # Số lớp Transformer Decoder block
    N_HEADS = 8            # Số đầu chú ý (Multi-Head Attention heads)
    D_FF = 2048            # Chiều ẩn của lớp Feed-Forward (thường là 4 * D_MODEL)
    MAX_SEQ_LEN = 1024     # Độ dài chuỗi tối đa (Context window)
    DROPOUT = 0.1          # Tỷ lệ dropout chống quá khớp (overfitting)
    
    # Thông số số học
    EPS = 1e-5             # Hằng số tránh chia cho 0 trong LayerNorm