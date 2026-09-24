"""Configuration management using Pydantic Settings"""
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database
    database_url: str = Field(default="postgresql://aiv:aiv@localhost:5432/aiv")
    
    # API Keys
    gemini_api_key: str = Field(default="")
    brave_api_key: str = Field(default="")
    
    # Rate limits (requests per minute)
    gemini_rpm: int = Field(default=60)
    brave_rpm: int = Field(default=20)
    
    # Cache
    cache_dir: Path = Field(default=Path("data/cache"))
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Model settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()