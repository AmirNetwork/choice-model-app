import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24))
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URI", f"sqlite:///{os.path.join(BASE_DIR, 'users.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage", "uploads")
    LOG_FOLDER = os.path.join(BASE_DIR, "storage", "logs")
    COMPLETED_MODELS_FOLDER = os.path.join(BASE_DIR, "storage", "completed_models")
    REFERENCES_FOLDER = os.path.join(BASE_DIR, "storage", "references")
    TRAINING_DATA_FOLDER = os.path.join(BASE_DIR, "storage", "training_data")
    TRAINED_REFERENCES_FOLDER = os.path.join(BASE_DIR, "storage", "trained_references")
    CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    GPTNEO_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
    ALLOWED_EXTENSIONS = {"csv", "xlsx"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit
    LLM_TIMEOUT_SECS = 180
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"