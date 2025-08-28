from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    _BASE_DIR = Path(__file__).resolve().parent.parent

    # Models
    MODELS_BASE_PATH: str = str(_BASE_DIR / "models")
    MODEL_VERSION: str | None = "auto"

    # ==== Inferencia / Ensamble (usados por service.py) ====
    # "best_auc" | "default" | "soft_voting"
    SELECT_STRATEGY: str = "soft_voting"
    # si el request trae use_model, ¿hago ensamble de todos?
    ENSEMBLE_ON_SELECTION: bool = True
    # "vote" | "proba" | "hybrid"
    ENSEMBLE_RULE: str = "hybrid"

    # Security / API
    DEBUG: bool = True  # pon False en prod → oculta /docs y /redoc
    API_KEY: str = "dev-change-me"
    METRICS_KEY: str = "dev-metrics-key"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]

    # Server hardening
    MAX_BODY_BYTES: int = 256 * 1024  # 256 KB
    RATE_LIMIT_PER_MINUTE: int = 60   # simple limiter

    PORT: int = 8000

settings = Settings()
