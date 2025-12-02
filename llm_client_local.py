#!/usr/bin/env python3
"""
Local LLM Client using MLX-VLM
==============================

Runs Qwen3-VL locally on Apple Silicon (M1/M2/M3/M4) using MLX.
No API key required - completely offline after model download.

Supported models:
- lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit (recommended, ~5GB RAM)
- mlx-community/Qwen3-VL-2B-Instruct-4bit (smaller, ~2GB RAM)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set MLX environment before importing
os.environ.setdefault('MLX_METAL_PREALLOCATE', 'false')


class LocalLLMClient:
    """Local MLX-VLM client for EW measurement with Qwen3-VL."""
    
    # Default model (4-bit for memory efficiency on 16GB Macs)
    DEFAULT_MODEL = 'lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit'
    
    def __init__(self, model_id: str = None):
        """
        Initialize local LLM client.
        
        Args:
            model_id: HuggingFace model ID for MLX-VLM model.
                     Defaults to Qwen3-VL-8B-Instruct-MLX-4bit
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self._model = None
        self._processor = None
        self._loaded = False
        
    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self._loaded:
            return
            
        print(f"🔄 Loading local model: {self.model_id}")
        print("   (First run will download ~4GB from HuggingFace)")
        
        try:
            from mlx_vlm import load
            self._model, self._processor = load(self.model_id)
            self._loaded = True
            print("✅ Model loaded successfully!")
        except ImportError:
            raise ImportError(
                "mlx-vlm not installed. Install with:\n"
                "  pip install mlx-vlm\n"
                "Note: Requires macOS with Apple Silicon (M1/M2/M3/M4)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def chat_with_vision(
        self,
        text_prompt: str,
        image_base64: str = None,
        image_path: str = None,
        timeout: int = 120,
        max_tokens: int = 1000,
    ) -> str:
        """
        Call local VLM with vision capability for plot inspection.
        
        Args:
            text_prompt: Text prompt
            image_base64: Base64-encoded image (optional, not recommended)
            image_path: Path to image file (recommended)
            timeout: Not used for local models
            max_tokens: Maximum tokens in response
            
        Returns:
            Response content as string
        """
        self._ensure_loaded()
        
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        
        # Build message content
        content = []
        
        # Add image (prefer file path for better compatibility)
        if image_path:
            content.append({"type": "image", "image": str(image_path)})
        elif image_base64:
            # MLX-VLM has issues with base64, save to temp file
            import tempfile
            import base64
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(base64.b64decode(image_base64))
                temp_path = f.name
            content.append({"type": "image", "image": temp_path})
        
        # Add text
        content.append({"type": "text", "text": text_prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Format prompt for model
        formatted = apply_chat_template(
            self._processor, 
            config=self._model.config, 
            prompt=messages
        )
        
        # Generate response
        output = generate(
            self._model, 
            self._processor, 
            formatted, 
            max_tokens=max_tokens,
            verbose=False
        )
        
        # Clean up temp file if created
        if image_base64 and not image_path:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return output.text
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: str = None,
        timeout: int = 90,
        max_retries: int = 1,  # No retries needed for local
        initial_delay: float = 0,
    ) -> Any:
        """
        Chat with local LLM (with function calling support).
        
        Note: Function calling is emulated - Qwen3-VL doesn't have native
        tool use, so we parse the response for tool calls.
        
        Args:
            messages: List of message dicts
            tools: Tool definitions (for prompt construction)
            system_prompt: System prompt
            timeout: Not used
            max_retries: Not used
            initial_delay: Not used
            
        Returns:
            Response object mimicking OpenAI format
        """
        self._ensure_loaded()
        
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        
        # Build the prompt
        full_messages = []
        
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        
        # Add tool descriptions to system prompt if tools provided
        if tools:
            tool_desc = self._format_tools_for_prompt(tools)
            if full_messages and full_messages[0]["role"] == "system":
                full_messages[0]["content"] += "\n\n" + tool_desc
            else:
                full_messages.insert(0, {"role": "system", "content": tool_desc})
        
        full_messages.extend(messages)
        
        # Check for images in messages
        has_image = False
        processed_messages = []
        for msg in full_messages:
            if isinstance(msg.get("content"), list):
                # Message with image
                content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        # Convert OpenAI format to MLX format
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            # Base64 image - save to temp file
                            import base64
                            import tempfile
                            b64_data = url.split(",")[1]
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                                f.write(base64.b64decode(b64_data))
                                content.append({"type": "image", "image": f.name})
                        else:
                            content.append({"type": "image", "image": url})
                        has_image = True
                    elif item.get("type") == "text":
                        content.append(item)
                    else:
                        content.append(item)
                processed_messages.append({"role": msg["role"], "content": content})
            else:
                processed_messages.append(msg)
        
        # Format for model
        formatted = apply_chat_template(
            self._processor,
            config=self._model.config,
            prompt=processed_messages
        )
        
        # Generate
        output = generate(
            self._model,
            self._processor,
            formatted,
            max_tokens=2000,
            verbose=False
        )
        
        # Parse response for tool calls
        response_text = output.text
        tool_calls = self._parse_tool_calls(response_text, tools) if tools else None
        
        # Return OpenAI-compatible response object
        return LocalResponse(
            content=response_text,
            tool_calls=tool_calls,
            model=self.model_id,
            usage={
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.generation_tokens,
                "total_tokens": output.total_tokens
            }
        )
    
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools for prompt injection."""
        lines = ["You have access to the following tools:\n"]
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                lines.append(f"**{func['name']}**: {func.get('description', '')}")
                if func.get("parameters", {}).get("properties"):
                    lines.append("  Parameters:")
                    for name, prop in func["parameters"]["properties"].items():
                        lines.append(f"    - {name}: {prop.get('description', prop.get('type', 'any'))}")
                lines.append("")
        
        lines.append("\nTo use a tool, respond with a JSON block like:")
        lines.append('```json')
        lines.append('{"tool": "tool_name", "arguments": {"param1": "value1"}}')
        lines.append('```')
        lines.append("\nYou can call multiple tools by including multiple JSON blocks.")
        
        return "\n".join(lines)
    
    def _parse_tool_calls(self, text: str, tools: List[Dict]) -> Optional[List]:
        """Parse tool calls from response text."""
        import re
        
        # Look for JSON blocks
        pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try inline JSON
            pattern = r'\{["\']tool["\']:\s*["\'][^"\']+["\'][^}]+\}'
            matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        tool_names = {t["function"]["name"] for t in tools if t.get("type") == "function"}
        tool_calls = []
        
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                tool_name = data.get("tool") or data.get("name") or data.get("function")
                if tool_name in tool_names:
                    tool_calls.append(LocalToolCall(
                        id=f"call_{i}",
                        name=tool_name,
                        arguments=json.dumps(data.get("arguments", data.get("params", {})))
                    ))
            except json.JSONDecodeError:
                continue
        
        return tool_calls if tool_calls else None


class LocalResponse:
    """Mimics OpenAI response structure."""
    
    def __init__(self, content: str, tool_calls: list = None, model: str = "", usage: dict = None):
        self.choices = [LocalChoice(content, tool_calls)]
        self.model = model
        self.usage = usage or {}


class LocalChoice:
    """Mimics OpenAI choice structure."""
    
    def __init__(self, content: str, tool_calls: list = None):
        self.message = LocalMessage(content, tool_calls)
        self.finish_reason = "tool_calls" if tool_calls else "stop"


class LocalMessage:
    """Mimics OpenAI message structure."""
    
    def __init__(self, content: str, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class LocalToolCall:
    """Mimics OpenAI tool call structure."""
    
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = LocalFunction(name, arguments)


class LocalFunction:
    """Mimics OpenAI function structure."""
    
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


# Quick test
if __name__ == "__main__":
    print("Testing LocalLLMClient...")
    client = LocalLLMClient()
    
    # Test text-only
    response = client.chat([{"role": "user", "content": "What is 2+2?"}])
    print(f"Response: {response.choices[0].message.content[:200]}...")

