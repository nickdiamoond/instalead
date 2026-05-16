def _banner(title: str, char: str = "=") -> None:
    """Print a wide stdout banner — survives the structlog stderr scroll
    on Windows PowerShell, so per-step status remains readable after the
    run finishes."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def _format_eta(seconds: float) -> str:
    """Render a duration as ``Xh Ym`` / ``Ym Zs`` for the cost banner."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.0f}s"


def _realtor_usernames_from_cfg(cfg: dict) -> list[str]:
    """Instagram usernames for Step 1 ``discovery_mode=realtors``.

    Reads ``search.realtor_accounts`` from config (same contract as
    ``search.hashtags`` for the hashtag path): non-strings skipped,
    stripped, empties dropped, duplicates removed with order preserved.
    """
    raw = list((cfg.get("search") or {}).get("realtor_accounts") or [])
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        u = x.strip()
        if u:
            out.append(u)
    return list(dict.fromkeys(out))
