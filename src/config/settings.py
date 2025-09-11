"""
Platform Configuration Settings

Central configuration for the Market & Customer Intelligence Platform.
Handles environment variables, API keys, and service configurations.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class PlatformSettings(BaseSettings):
    """Main platform configuration settings."""
    
    # Core Platform
    platform_name: str = "Market & Customer Intelligence Platform"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # LLM & AI Services
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # LangChain & Monitoring
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="market-intelligence", env="LANGCHAIN_PROJECT")
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    
    # Weaviate Vector Database
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # MCP Servers Configuration
    mcp_social_media_endpoint: str = Field(default="ws://localhost:3001", env="MCP_SOCIAL_MEDIA_ENDPOINT")
    mcp_review_platforms_endpoint: str = Field(default="ws://localhost:3002", env="MCP_REVIEW_PLATFORMS_ENDPOINT")
    mcp_competitive_intel_endpoint: str = Field(default="ws://localhost:3003", env="MCP_COMPETITIVE_INTEL_ENDPOINT")
    
    # Data Source APIs
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    g2_api_key: Optional[str] = Field(default=None, env="G2_API_KEY")
    google_places_api_key: Optional[str] = Field(default=None, env="GOOGLE_PLACES_API_KEY")
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-west-2", env="AWS_REGION")
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    
    # Processing Configuration
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Sentiment Analysis
    sentiment_batch_size: int = Field(default=32, env="SENTIMENT_BATCH_SIZE")
    sentiment_threshold: float = Field(default=0.7, env="SENTIMENT_THRESHOLD")
    
    # Trend Detection
    trend_detection_window_days: int = Field(default=30, env="TREND_DETECTION_WINDOW_DAYS")
    trend_significance_threshold: float = Field(default=0.05, env="TREND_SIGNIFICANCE_THRESHOLD")
    
    # Caching & Performance
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = PlatformSettings()


def get_settings() -> PlatformSettings:
    """Get platform settings instance."""
    return settings