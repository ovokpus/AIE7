import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    app_name: str = "PostAssist"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    
    # Tavily Configuration  
    tavily_api_key: str = ""
    
    # Redis Configuration (for task queue)
    redis_url: str = "redis://localhost:6379"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # File Storage
    working_directory: str = "./content/data"
    max_file_size: int = 10_000_000  # 10MB
    
    # LangGraph Configuration
    recursion_limit: int = 50
    max_tokens: int = 4000
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def validate_api_keys():
    """Validate that required API keys are present."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")


# Set environment variables for LangChain
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["TAVILY_API_KEY"] = settings.tavily_api_key 