# configs/paths_config.py
import os

class PathsConfig:
    """Quản lý đường dẫn thư mục và file trong dự án ZenithVN"""
    
    # Thư mục gốc của dự án
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Các thư mục con
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    ENGINE_DIR = os.path.join(BASE_DIR, "engine")
    UTILS_DIR = os.path.join(BASE_DIR, "utils")
    
    # Đường dẫn các file cụ thể
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.txt")
    TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.model")
    GGUF_OUTPUT_PATH = os.path.join(BASE_DIR, "zenithvn.gguf")
    MODELFILE_PATH = os.path.join(BASE_DIR, "Modelfile")

    @classmethod
    def create_dirs(cls):
        """Tự động tạo các thư mục cần thiết nếu chưa tồn tại"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)