import argparse


def _parse_cli_args() -> argparse.Namespace:
    """Pipeline-level CLI flags. Kept minimal -- the daily run uses
    no flags; flags exist for ad-hoc Step 5 / Step 6 runs."""
    parser = argparse.ArgumentParser(
        description="Daily lead collection pipeline (Apify + DeepSeek + Sherlock)."
    )
    parser.add_argument(
        "--skip-sherlock",
        action="store_true",
        help="Skip Step 5 (Sherlock contact resolution). "
             "Useful when only Steps 1-4 are needed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override Step 5 worker count when pipeline.step5.sequential "
             "is false. Default: probe /v1/health and use pool.idle "
             "(fallback 3).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm Step 5's cost prompt (skip the y/n input). "
             "Use only when running unattended. Same effect as "
             "pipeline.prompt_terminal_confirmation: false for Step 5 only "
             "(--yes does not skip Step 3 or the post-issues Enter pause).",
    )
    parser.add_argument(
        "--keep-photos",
        action="store_true",
        help="Skip Step 6 (cleanup of spent face assets). Use for "
             "debugging / forensic sessions where you need avatars "
             "and face photos to stay on disk after Sherlock.",
    )
    return parser.parse_args()
