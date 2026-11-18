import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_env_variable(variable_name: str) -> str:
    value = os.getenv(variable_name)

    if not value:
        raise ValueError(
            f"Environment variable '{variable_name}' is missing"
        )

    return value