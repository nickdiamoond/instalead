import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Defaults for ``pipeline.step1`` when keys are missing from YAML
# (used by ``scripts/pipeline.py``, ``ApifyWrapper``, and dev scripts).
PIPELINE_STEP1_DEFAULT_POSTS_MAX_AGE_DAYS = 7
PIPELINE_STEP1_DEFAULT_MIN_COMMENTS_PER_POST = 10


def _step1_section(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("pipeline") or {}).get("step1") or {}


def step1_posts_max_age_days(cfg: dict[str, Any]) -> int:
    """Resolve ``pipeline.step1.posts_max_age_days`` with module default."""
    raw = _step1_section(cfg).get(
        "posts_max_age_days", PIPELINE_STEP1_DEFAULT_POSTS_MAX_AGE_DAYS
    )
    return int(raw)


def step1_min_comments_per_post(cfg: dict[str, Any]) -> int:
    """Resolve ``pipeline.step1.min_comments_per_post`` with module default."""
    raw = _step1_section(cfg).get(
        "min_comments_per_post", PIPELINE_STEP1_DEFAULT_MIN_COMMENTS_PER_POST
    )
    return int(raw)


def step1_cookie_search_section(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cookie-keyword actor tuning: ``pipeline.step1.cookie_search`` (preferred).

    Falls back to legacy ``search.cookie_search`` when step1 omits the block.
    """
    s1 = _step1_section(cfg)
    return (s1.get("cookie_search") or (cfg.get("search") or {}).get("cookie_search") or {})


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
