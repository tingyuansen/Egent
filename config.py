"""
Egent Configuration
===================

Centralized configuration for the Egent EW measurement pipeline.
Supports both Azure OpenAI and direct OpenAI API backends.

Environment Variables:
    OPENAI_API_KEY: OpenAI API key (for OpenAI backend)
    AZURE_API_KEY or AZURE: Azure OpenAI API key (for Azure backend)
    EGENT_BACKEND: 'openai' or 'azure' (default: 'openai')
    EGENT_MODEL: Model to use (default: 'gpt-5' or 'gpt-5-mini')

Usage:
    from config import get_config
    cfg = get_config()
    print(cfg.model_id)
    print(cfg.api_key)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from dotenv import load_dotenv

# Load environment from ~/.env
load_dotenv(Path.home() / '.env')


@dataclass
class EgentConfig:
    """Configuration for the Egent pipeline."""
    
    # API Backend: 'openai' or 'azure'
    backend: Literal['openai', 'azure'] = 'azure'
    
    # Model configuration
    model_id: str = 'gpt-5'
    model_mini_id: str = 'gpt-5-mini'
    
    # API endpoints
    openai_base_url: str = 'https://api.openai.com/v1'
    azure_endpoint: str = 'https://astromlab.openai.azure.com'
    api_version: str = '2024-12-01-preview'
    
    # Rate limiting
    max_retries: int = 5
    base_delay: float = 2.0
    base_delay_mini: float = 0.5  # Faster for mini models
    
    # Worker configuration
    default_workers: int = 15
    default_workers_mini: int = 20
    
    # Quality thresholds
    good_rms_threshold: float = 1.5
    
    # Output directories
    output_dir: Optional[Path] = None
    output_dir_mini: Optional[Path] = None
    
    # API keys (populated from environment)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    azure_api_key: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load API keys from environment and set defaults."""
        # Load API keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI')
        self.azure_api_key = os.getenv('AZURE_API_KEY') or os.getenv('AZURE')
        
        # Check backend from environment
        env_backend = os.getenv('EGENT_BACKEND', '').lower()
        if env_backend in ('openai', 'azure'):
            self.backend = env_backend
        elif self.openai_api_key and not self.azure_api_key:
            self.backend = 'openai'
        elif self.azure_api_key and not self.openai_api_key:
            self.backend = 'azure'
        
        # Allow environment override for model
        env_model = os.getenv('EGENT_MODEL')
        if env_model:
            self.model_id = env_model
        
        # Set default output directories
        if self.output_dir is None:
            self.output_dir = Path.home() / 'Egent_output'
        if self.output_dir_mini is None:
            self.output_dir_mini = Path.home() / 'Egent_output_mini'
    
    @property
    def api_key(self) -> str:
        """Get the API key for the current backend."""
        if self.backend == 'openai':
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in ~/.env or environment."
                )
            return self.openai_api_key
        else:
            if not self.azure_api_key:
                raise ValueError(
                    "Azure API key not found. Set AZURE_API_KEY or AZURE in ~/.env or environment."
                )
            return self.azure_api_key
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the current backend."""
        if self.backend == 'openai':
            return self.openai_base_url
        return self.azure_base_url
    
    def get_workers(self, use_mini: bool = False) -> int:
        """Get the recommended worker count."""
        return self.default_workers_mini if use_mini else self.default_workers
    
    def get_base_delay(self, use_mini: bool = False) -> float:
        """Get the base delay for rate limiting."""
        return self.base_delay_mini if use_mini else self.base_delay
    
    def get_model(self, use_mini: bool = False) -> str:
        """Get the model ID."""
        return self.model_mini_id if use_mini else self.model_id
    
    def get_output_dir(self, use_mini: bool = False) -> Path:
        """Get the output directory."""
        return self.output_dir_mini if use_mini else self.output_dir


# Global config instance (lazy-loaded)
_config: Optional[EgentConfig] = None


def get_config() -> EgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = EgentConfig()
    return _config


def set_config(**kwargs) -> EgentConfig:
    """Create a new configuration with custom settings."""
    global _config
    _config = EgentConfig(**kwargs)
    return _config

