"""
Egent Configuration
===================

Configuration for the Egent EW measurement pipeline.

Backends:
    - 'openai': OpenAI API (requires API key, default: gpt-5-mini)
    - 'local': Local MLX-VLM (no API key, requires Apple Silicon)

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (for openai backend)
    EGENT_BACKEND: 'openai' or 'local' (default: 'openai')
    EGENT_MODEL: Model to use

Usage:
    from config import get_config
    cfg = get_config()
    print(cfg.backend, cfg.model_id)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / '.env')
except ImportError:
    pass


@dataclass
class EgentConfig:
    """Configuration for the Egent pipeline."""
    
    # Backend: 'openai' or 'local'
    backend: Literal['openai', 'local'] = 'openai'
    
    # Model configuration
    # - OpenAI: 'gpt-5-mini' (default), 'gpt-4o', etc.
    # - Local: 'lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit' (default)
    model_id: str = None
    
    # Rate limiting (for OpenAI)
    max_retries: int = 5
    base_delay: float = 0.5
    
    # Worker configuration for parallel processing
    # Local models should use 1 worker (no parallelism)
    default_workers: int = None
    
    # Quality thresholds
    good_rms_threshold: float = 1.5
    
    # Output directory
    output_dir: Optional[Path] = None
    
    # API key (populated from environment, only for openai backend)
    api_key: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load settings from environment."""
        # Check backend from environment
        env_backend = os.getenv('EGENT_BACKEND', '').lower()
        if env_backend in ('openai', 'local'):
            self.backend = env_backend
        
        # Set default model based on backend
        if self.model_id is None:
            if self.backend == 'openai':
                self.model_id = 'gpt-5-mini'
            else:
                self.model_id = 'lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit'
        
        # Allow environment override for model
        env_model = os.getenv('EGENT_MODEL')
        if env_model:
            self.model_id = env_model
        
        # Set default workers based on backend
        if self.default_workers is None:
            self.default_workers = 10 if self.backend == 'openai' else 1
        
        # Load API key for OpenAI backend
        if self.backend == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Set default output directory
        if self.output_dir is None:
            self.output_dir = Path.home() / 'Egent_output'
    
    def validate(self):
        """Validate configuration."""
        if self.backend == 'openai' and not self.api_key:
            raise ValueError(
                "OpenAI API key not found.\n"
                "Set OPENAI_API_KEY in your environment or ~/.env file:\n"
                "  export OPENAI_API_KEY='your-key-here'\n\n"
                "Or use local backend (no API key required):\n"
                "  export EGENT_BACKEND='local'"
            )
        return True


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
