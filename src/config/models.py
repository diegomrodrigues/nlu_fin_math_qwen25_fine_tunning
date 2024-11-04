from dataclasses import dataclass
from enum import Enum

@dataclass
class ModelConfig:
    """Base configuration for AI models."""
    name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    batch_size: int = 1000
    provider: 'ModelProvider' = 'ModelProvider.ANTHROPIC'

class ModelProvider(Enum):
    """Supported AI model providers."""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

# Define available models configurations
MODELS = {
    # Anthropic models
    "claude-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        max_tokens=4096,
        temperature=0
    ),
    "claude-haiku": ModelConfig(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        max_tokens=4096,
        temperature=0
    ),
    "claude-opus": ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC
    )
} 