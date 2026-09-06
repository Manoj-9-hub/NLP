import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "T5Summarizer REST API"
    API_V1_STR: str = "/api"
    
    # Model Configurations
    # Allow over-riding via environment variable, fallback to t5-small
    MODEL_CHECKPOINT: str = os.getenv("MODEL_CHECKPOINT", "t5-small")
    
    # Device setup
    # CUDA is supported if available, fallback to cpu
    DEVICE: str = os.getenv("DEVICE", "")  # auto-detected if empty
    
    # Document configurations
    # Max file size: 10MB
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "txt"}
    UPLOAD_DIR: str = "uploads"
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "models/history.db")
    
    # Fine-Tuning Saves Dir
    FINE_TUNE_DIR: str = "models/fine_tuned"

    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FINE_TUNE_DIR, exist_ok=True)
