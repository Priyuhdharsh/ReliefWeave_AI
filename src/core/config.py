from pathlib import Path
import yaml
from pydantic import BaseModel
from typing import Literal
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ProviderModelConfig(BaseModel):
    model: str


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "gemini"]
    openai: ProviderModelConfig
    gemini: ProviderModelConfig


class AppConfig(BaseModel):
    embedding: EmbeddingConfig

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return AppConfig.model_validate(config)