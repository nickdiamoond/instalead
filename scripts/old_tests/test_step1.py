"""Step 1 (realtors only) — Apify post-scraper smoke test, no DB.

Fetches posts from ``search.realtor_accounts`` via the same actor/input
as ``scripts/pipeline.py`` Step 1 (``discovery_mode=realtors``), then
prints raw Apify JSON, then each item as post URL - post date after the
same client-side ``filter_items_within_max_age`` used for hashtags/keys
in ``pipeline.py`` Step 1.

Usage:
    python scripts/test_step1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.config import load_config, step1_posts_max_age_days
from src.ig_media_payload import filter_items_within_max_age
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_step1")

DEFAULT_POST_SCRAPER_RESULTS_LIMIT = 20


def _banner(title: str, char: str = "=") -> None:
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def _realtor_usernames_from_cfg(cfg: dict) -> list[str]:
    raw = list((cfg.get("search") or {}).get("realtor_accounts") or [])
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        u = x.strip()
        if u:
            out.append(u)
    return list(dict.fromkeys(out))


def _post_link(item: dict) -> str:
    url = (item.get("url") or "").strip()
    if url:
        return url
    shortcode = (item.get("shortCode") or "").strip()
    if shortcode:
        return f"https://www.instagram.com/p/{shortcode}/"
    return "—"


def _post_date(item: dict) -> str:
    ts = item.get("timestamp")
    if ts is None or ts == "":
        return "—"
    return str(ts)


def _print_raw_apify(items: list[dict], run_detail: dict) -> None:
    print("--- Raw Apify response ---")
    print(json.dumps(
        {"run": run_detail, "dataset_items": items},
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    print("--- End raw response ---\n")


def _print_posts(items: list[dict]) -> None:
    print("--- Posts (url - date) ---")
    for item in items:
        print(f"{_post_link(item)} - {_post_date(item)}")


def run_realtor_fetch(
    cfg: dict,
    *,
    apify: ApifyClient,
    pipeline: PipelineLogger,
) -> list[tuple[str, str]]:
    """Fetch realtor posts from Apify; return ``(step, hint)`` issues."""
    issues: list[tuple[str, str]] = []

    s1_cfg = (cfg.get("pipeline") or {}).get("step1") or {}
    posts_max_age_days = step1_posts_max_age_days(cfg)
    post_scraper_results_limit = int(
        s1_cfg.get("post_scraper_results_limit", DEFAULT_POST_SCRAPER_RESULTS_LIMIT)
    )

    _banner(f"STEP 1: Fetch posts (last {posts_max_age_days} days) [realtors]")
    realtors = _realtor_usernames_from_cfg(cfg)
    if not realtors:
        log.error("step1_no_realtor_accounts")
        print("FAILED: search.realtor_accounts is empty in config.")
        issues.append(("Step 1", "search.realtor_accounts empty"))
        return issues

    print(f"  Realtors:       {len(realtors)}")
    log.info(
        "step1_fetch_posts",
        discovery_mode="realtors",
        realtors=len(realtors),
        max_age_days=posts_max_age_days,
        post_scraper_results_limit=post_scraper_results_limit,
    )

    run = apify.actor("apify/instagram-post-scraper").call(
        run_input={
            "username": realtors,
            "resultsLimit": post_scraper_results_limit,
            "onlyPostsNewerThan": f"{posts_max_age_days} days",
            "dataDetailLevel": "basicData",
            "proxy": {"useApifyProxy": True},
        }
    )
    detail = apify.run(run["id"]).get() or {}
    items_raw = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
    items_filtered, age_stats = filter_items_within_max_age(
        items_raw, posts_max_age_days
    )
    cost_usd = float(detail.get("usageTotalUsd") or 0)

    pipeline.log_run(
        actor_id="apify/instagram-post-scraper",
        run_id=run["id"],
        status=run["status"],
        input_params={
            "realtors": len(realtors),
            "resultsLimit": post_scraper_results_limit,
            "max_age_days": posts_max_age_days,
        },
        items_count=len(items_raw),
        cost_usd=cost_usd,
        duration_ms=detail.get("stats", {}).get("durationMillis"),
    )

    log.info(
        "step1_done",
        fetched=len(items_raw),
        after_age_filter=len(items_filtered),
        age_stats=age_stats,
        cost=cost_usd,
    )
    print(
        f"\n  Fetched: {len(items_raw)}  "
        f"after client age filter (≤{posts_max_age_days}d): {len(items_filtered)}  "
        f"dropped_too_old={age_stats['dropped_too_old']}  "
        f"kept_missing_timestamp={age_stats['kept_missing_timestamp']}  "
        f"cost=${cost_usd:.4f}\n"
    )
    _print_raw_apify(items_raw, run_detail=detail)
    _print_posts(items_filtered)

    if not items_raw:
        issues.append(("Step 1", "post-scraper returned 0 items"))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch realtor posts via Apify (no DB); print URL - date."
    )
    parser.parse_args()

    load_dotenv()
    cfg = load_config()
    apify = ApifyClient(cfg["apify"]["token"])
    log_dir = (cfg.get("logging") or {}).get("pipeline_log_dir", "logs")
    pipeline = PipelineLogger(log_dir, "test_step1")

    log.info("test_step1_start")
    issues = run_realtor_fetch(cfg, apify=apify, pipeline=pipeline)

    if issues:
        print("\nIssues:")
        for step, hint in issues:
            print(f"  [{step}] {hint}")
        return 1

    ps = pipeline.summary()
    print(f"\nApify session cost: ${ps.get('total_cost_usd') or 0:.4f}")
    print(f"Pipeline log: {pipeline.file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
