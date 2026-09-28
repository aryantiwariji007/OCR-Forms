# config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MaintServe (Qwen3-VL via vLLM) — used as the verification stage
    MAINTSERVE_BASE_URL: str = "http://69.19.136.22:8000/api/v1"
    MAINTSERVE_API_KEY: str = ""
    MAINTSERVE_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
    MAINTSERVE_MAX_TOKENS: int = 2048
    MAINTSERVE_TEMPERATURE: float = 0.1

    # Anthropic (Claude) — comparison VLM for gauge-reading accuracy testing
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-5"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
