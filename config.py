"""
Egent Configuration
===================

Configuration for the Egent EW measurement pipeline.
Uses OpenAI API with GPT-5-mini as default.

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    EGENT_MODEL: Model to use (default: 'gpt-5-mini')

Usage:
    from config import get_config
    cfg = get_config()
    print(cfg.model_id)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / '.env')
except ImportError:
    pass


@dataclass
class EgentConfig:
    """Configuration for the Egent pipeline."""
    
    # Model configuration (GPT-5-mini is faster and cheaper)
    model_id: str = 'gpt-5-mini'
    
    # Rate limiting
    max_retries: int = 5
    base_delay: float = 0.5
    
    # Worker configuration for parallel processing
    default_workers: int = 10
    
    # Quality thresholds
    good_rms_threshold: float = 1.5
    
    # Output directory
    output_dir: Optional[Path] = None
    
    # API key (populated from environment)
    api_key: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load API key from environment."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Allow environment override for model
        env_model = os.getenv('EGENT_MODEL')
        if env_model:
            self.model_id = env_model
        
        # Set default output directory
        if self.output_dir is None:
            self.output_dir = Path.home() / 'Egent_output'
    
    def validate(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found.\n"
                "Set OPENAI_API_KEY in your environment or ~/.env file:\n"
                "  export OPENAI_API_KEY='your-key-here'"
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
