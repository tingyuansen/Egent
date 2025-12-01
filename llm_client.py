#!/usr/bin/env python3
"""
Unified LLM Client with Retry Logic
====================================

Handles OpenAI and Azure OpenAI API calls with exponential backoff.
"""

import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI


class LLMClient:
    """Unified LLM client for OpenAI and Azure with retry logic."""
    
    def __init__(self, use_mini: bool = False):
        """Initialize client with configuration."""
        from config import get_config
        config = get_config()
        self.use_mini = use_mini
        self.backend = config.backend
        self.model_id = config.get_model(use_mini)
        
        if self.backend == 'azure':
            # Use OpenAI client with Azure endpoint (simpler, works with both)
            self.client = OpenAI(
                base_url=config.azure_endpoint + "/openai/v1",
                api_key=config.api_key
            )
        else:  # openai
            self.client = OpenAI(api_key=config.api_key)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: str = None,
        timeout: int = 90,
        max_retries: int = 8,
        initial_delay: float = 2.0,
    ) -> Any:
        """
        Call LLM with exponential backoff retry logic.
        
        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            system_prompt: Optional system prompt to prepend
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            
        Returns:
            OpenAI response object
        """
        # Prepend system message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    'model': self.model_id,
                    'messages': messages,
                    'timeout': timeout,
                }
                if tools:
                    kwargs['tools'] = tools
                # Note: temperature/max_tokens not supported by some models
                
                response = self.client.chat.completions.create(**kwargs)
                return response
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's retryable
                is_retryable = any(x in error_str for x in [
                    'rate', 'limit', 'timeout', '429', '503', '500', 'overloaded', 'retry',
                    'connection', 'reset', 'refused'
                ])
                
                if not is_retryable or attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter, longer waits for rate limits
                base_wait = initial_delay * (2 ** attempt)
                # Rate limits often need 60+ seconds
                if '429' in error_str or 'rate' in error_str:
                    base_wait = max(base_wait, 30)  # At least 30s for rate limits
                wait_time = base_wait + (time.time() % 5)  # Add jitter
                
                print(f"  ⚠️  LLM error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"  ⏳ Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        raise last_error
    
    def chat_with_vision(
        self,
        text_prompt: str,
        image_base64: str = None,
        image_path: str = None,
        timeout: int = 120,
        max_tokens: int = 1000,
        max_retries: int = 8,
        initial_delay: float = 2.0,
    ) -> str:
        """
        Call LLM with vision capability.
        
        Args:
            text_prompt: Text prompt
            image_base64: Base64-encoded image (optional)
            image_path: Path to image file (optional)
            timeout: Request timeout
            max_tokens: Maximum tokens in response
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            
        Returns:
            Response content as string
        """
        # Load and encode image if path provided
        if image_path and not image_base64:
            img_path = Path(image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            with open(img_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine image format
            ext = img_path.suffix.lower()
            mime_type = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext[1:], 'image/png')
        else:
            mime_type = 'image/png'
        
        # Build message with image
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_base64}', 'detail': 'high'}}
            ]
        }]
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    'model': self.model_id,
                    'messages': messages,
                    'max_completion_tokens': max_tokens,
                    'timeout': timeout,
                }
                # Note: temperature not supported by some models
                
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                is_retryable = any(x in error_str for x in [
                    'rate', 'limit', 'timeout', '429', '503', '500', 'overloaded', 'retry',
                    'connection', 'reset', 'refused'
                ])
                
                if not is_retryable or attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with longer waits for rate limits
                base_wait = initial_delay * (2 ** attempt)
                if '429' in error_str or 'rate' in error_str:
                    base_wait = max(base_wait, 30)
                wait_time = base_wait + (time.time() % 5)
                
                print(f"  ⚠️  Vision LLM error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"  ⏳ Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        raise last_error
