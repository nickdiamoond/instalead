"""Step 1 discovery mode parsing and Telegram summary helpers."""

from __future__ import annotations

from scripts.pipeline_lib.defaults import DEFAULT_STEP1_DISCOVERY_MODE

VALID_STEP1_DISCOVERY_MODES = frozenset(
    {"realtors", "hashtags", "cookie_keywords"},
)


def parse_step1_discovery_modes(raw: object) -> list[str]:
    """Normalize ``pipeline.step1.discovery_mode`` (str or list) to ordered unique modes."""
    if raw is None:
        items: list[object] = [DEFAULT_STEP1_DISCOVERY_MODE]
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]

    modes: list[str] = []
    seen: set[str] = set()
    for item in items:
        mode = str(item).strip().lower()
        if not mode or mode in seen:
            continue
        seen.add(mode)
        modes.append(mode)

    if not modes:
        return [DEFAULT_STEP1_DISCOVERY_MODE.strip().lower()]
    return modes


def format_step1_discovery_modes_label(modes: list[str]) -> str:
    """Compact label for banners and logs, e.g. ``realtors + hashtags``."""
    if not modes:
        return DEFAULT_STEP1_DISCOVERY_MODE
    return " + ".join(modes)


def build_step1_searched_summary(source_counts: dict[str, int]) -> str:
    """Human-readable 'searched N …' line for Step 1 Telegram / logs."""
    parts: list[str] = []
    if source_counts.get("realtors"):
        n = source_counts["realtors"]
        parts.append(f"{n} realtor(s)")
    if source_counts.get("hashtags"):
        n = source_counts["hashtags"]
        parts.append(f"{n} hashtag(s)")
    if source_counts.get("cookie_keywords"):
        n = source_counts["cookie_keywords"]
        parts.append(f"{n} keyword(s)")
    if not parts:
        return "No discovery sources ran."
    if len(parts) == 1:
        return f"Searched {parts[0]}."
    return "Searched " + ", ".join(parts) + "."


def discovery_modes_include_realtors(modes: list[str] | str) -> bool:
    """True when ``realtors`` is among the modes (Apify ``onlyPostsNewerThan`` applies)."""
    if isinstance(modes, str):
        parsed = [m.strip().lower() for m in modes.split(",") if m.strip()]
    else:
        parsed = [m.strip().lower() for m in modes]
    return "realtors" in parsed
