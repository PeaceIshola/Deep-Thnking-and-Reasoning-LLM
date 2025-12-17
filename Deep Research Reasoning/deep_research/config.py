"""
Configuration module for Deep Research System.
Provides centralized client setup and model configurations.
"""

from openai import OpenAI
from typing import Optional


class Config:
    """Configuration for LLM clients and models."""
    
    # Default models
    DEFAULT_BASE_MODEL = "llama3.2:3b"
    DEFAULT_REASONING_MODEL = "deepseek-r1:8b"
    
    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"
    
    @staticmethod
    def get_ollama_client() -> OpenAI:
        """
        Get an OpenAI-compatible client for Ollama.
        
        Returns:
            OpenAI: Client configured for Ollama
        """
        return OpenAI(
            api_key=Config.OLLAMA_API_KEY,
            base_url=Config.OLLAMA_BASE_URL
        )
    
    @staticmethod
    def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
        """
        Get an OpenAI client (for use with actual OpenAI API).
        
        Args:
            api_key: OpenAI API key (optional, uses env var if not provided)
            
        Returns:
            OpenAI: Client configured for OpenAI API
        """
        if api_key:
            return OpenAI(api_key=api_key)
        return OpenAI()  # Uses OPENAI_API_KEY env var


def get_client(use_openai: bool = False, api_key: Optional[str] = None) -> OpenAI:
    """
    Convenience function to get the appropriate client.
    
    Args:
        use_openai: If True, use OpenAI API; otherwise use Ollama
        api_key: API key for OpenAI (if using OpenAI)
        
    Returns:
        OpenAI: Configured client
    """
    if use_openai:
        return Config.get_openai_client(api_key)
    return Config.get_ollama_client()
