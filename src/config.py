import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml and inject secrets from .env."""
    load_dotenv()

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    token_var = cfg["apify"]["token_env_var"]
    token = os.environ.get(token_var)
    if not token:
        raise EnvironmentError(
            f"Environment variable {token_var} is not set. "
            f"Copy .env.example to .env and fill in your Apify token."
        )
    cfg["apify"]["token"] = token

    return cfg
