def _cfg_prompt_terminal_confirmation(value, default: bool = True) -> bool:
    """Parse ``pipeline.prompt_terminal_confirmation`` (bool / int / str)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("false", "no", "n", "0", "off"):
            return False
        if s in ("true", "yes", "y", "1", "on"):
            return True
        return default
    return default
