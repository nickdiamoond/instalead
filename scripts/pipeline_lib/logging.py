"""Shared structlog handle for the daily pipeline."""

from src.logger import get_logger

log = get_logger("pipeline")
