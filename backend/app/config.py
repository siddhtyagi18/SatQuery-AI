import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "SatQuery-AI Backend"
    APP_VERSION: str = "0.2.0-phase2"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"

    DATABASE_URL: str = "sqlite:///./satquery.db"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 512

    ALLOWED_EXTENSIONS: str = ".tif,.tiff,.png,.jpg,.jpeg"

    STORAGE_BACKEND: str = "local"

    FIREBASE_ENABLED: bool = False
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None
    FIREBASE_SERVICE_ACCOUNT_JSON: Optional[str] = None
    FIREBASE_COLLECTION_ANALYSES: str = "analyses"

    VQA_MODE: str = "auto"
    VQA_MODEL_ID: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    VQA_DEVICE: str = "cpu"
    VQA_PRECISION: str = "fp32"
    VQA_MAX_NEW_TOKENS: int = 512
    VQA_TEMPERATURE: float = 0.2
    VQA_INFERENCE_TIMEOUT_SEC: int = 180
    VQA_CACHE_DIR: Optional[str] = None
    VQA_HF_TOKEN: Optional[str] = None

    INFERENCE_REQUEST_TIMEOUT_SEC: int = 240

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()]

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def vqa_cache_dir_path(self) -> Optional[Path]:
        if not self.VQA_CACHE_DIR:
            return None
        p = Path(self.VQA_CACHE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
