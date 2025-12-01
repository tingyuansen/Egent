#!/usr/bin/env python3
"""
LLM Client with Retry Logic
============================

Handles OpenAI API calls with exponential backoff for rate limits.
"""

import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI


class LLMClient:
    """OpenAI LLM client with retry logic for EW measurement."""
    
    def __init__(self):
        """Initialize client with configuration."""
        from config import get_config
        config = get_config()
        config.validate()
        
        self.model_id = config.model_id
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
            tools: Optional tool definitions for function calling
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
                
                response = self.client.chat.completions.create(**kwargs)
                return response
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's retryable
                is_retryable = any(x in error_str for x in [
                    'rate', 'limit', 'timeout', '429', '503', '500', 
                    'overloaded', 'retry', 'connection', 'reset', 'refused'
                ])
                
                if not is_retryable or attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                base_wait = initial_delay * (2 ** attempt)
                # Rate limits often need longer waits
                if '429' in error_str or 'rate' in error_str:
                    base_wait = max(base_wait, 30)
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
    ) -> str:
        """
        Call LLM with vision capability for plot inspection.
        
        Args:
            text_prompt: Text prompt
            image_base64: Base64-encoded image (optional)
            image_path: Path to image file (optional)
            timeout: Request timeout
            max_tokens: Maximum tokens in response
            
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
            
            ext = img_path.suffix.lower()
            mime_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
        else:
            mime_type = 'image/png'
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_base64}', 'detail': 'high'}}
            ]
        }]
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content
