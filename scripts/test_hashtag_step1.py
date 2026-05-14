"""Hashtag-based post discovery (mirrors ``pipeline.step1`` when ``discovery_mode=hashtags``).

Actor: ``apify/instagram-hashtag-scraper`` — documented in CLAUDE.md (Apify Actors table) and
``docs/apify_api_schemas.md`` (hashtag scraper section). Returns real posts/reels, not hashtag page URLs.

Relevance scoring (relevant / irrelevant / unknown) is Step 2 (DeepSeek); this script only
exercises hashtag discovery + the same ``min_comments`` gate as ``pipeline.step1``.

The hashtag actor does **not** accept ``onlyPostsNewerThan`` (unlike ``instagram-post-scraper``).
We apply the same window as Step 1 by filtering each item's ``timestamp`` **after** the run.

Lighter smoke test (posts only, verbose structlog per item): ``scripts/test_apify_search.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apify_client_wrapper import DEFAULT_APIFY_WRAPPER_LIMITS, ApifyWrapper
from src.config import load_config
from src.db import LeadDB
from src.ig_media_payload import filter_items_within_max_age, is_reel_payload
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("hashtag_step1")

# What instagram-hashtag-scraper returns per item (see docs/apify_api_schemas.md).
_ACTOR_OUTPUT_BLURB_RU = """
  Актор отдаёт не «голый текст», а объект медиа как у ленты/тега:
  • идентификация: shortCode, url, id, inputUrl
  • автор: ownerUsername, ownerFullName, ownerId
  • текст и теги: caption, hashtags[], mentions[]
  • счётчики и время: commentsCount, likesCount, timestamp (+ videoViewCount для видео)
  • медиа: type, productType (clips = Reel), displayUrl, images[], иногда musicInfo
  • комментарии: firstComment, latestComments[] (только хвост, не вся лента комментов)
  • гео: locationName, locationId (если есть)
"""

# Aligns with ``scripts/pipeline.py`` / ``pipeline.step1.posts_max_age_days`` default.
_DEFAULT_POSTS_MAX_AGE_DAYS = 7


def _min_comments_from_config(cfg: dict) -> int:
    pipe = cfg.get("pipeline") or {}
    s1 = pipe.get("step1") or {}
    return int(s1.get("min_comments_per_post", 10))


def _posts_max_age_days_from_config(cfg: dict) -> int:
    pipe = cfg.get("pipeline") or {}
    s1 = pipe.get("step1") or {}
    return int(s1.get("posts_max_age_days", _DEFAULT_POSTS_MAX_AGE_DAYS))


def _normalize_item(
    p: dict,
    *,
    source_label: str,
    matched_hashtags: list[str],
) -> dict:
    return {
        "url": p.get("url"),
        "shortcode": p.get("shortCode"),
        "content_type": "reel" if is_reel_payload(p) else "post",
        "owner_username": p.get("ownerUsername"),
        "caption": p.get("caption"),
        "comments_count": p.get("commentsCount") or 0,
        "likes_count": p.get("likesCount") or 0,
        "views_count": p.get("videoViewCount") or 0,
        "timestamp": p.get("timestamp"),
        "location_name": p.get("locationName"),
        "hashtags": p.get("hashtags") or [],
        "discovery_source": source_label,
        "config_hashtags_query": matched_hashtags,
    }


def _truncate(text: str | None, max_len: int) -> str:
    if not text:
        return "—"
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _fmt_list(items: list[str] | None, max_len: int) -> str:
    if not items:
        return "—"
    s = ", ".join(items)
    return _truncate(s, max_len)


def _print_actor_run_items(
    run_title: str,
    items: list[dict],
    *,
    min_comments: int,
    max_age_days: int = 0,
    age_stats: dict[str, int] | None = None,
) -> None:
    """Print one block per dataset item (RU labels). List is usually **after** age filter."""
    bar = "-" * 62
    print(f"\n{bar}\n  {run_title}\n  Всего items: {len(items)}\n{bar}")
    if max_age_days > 0:
        co = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        print(
            f"  Учёт даты: не старше {max_age_days} дн. "
            f"(отсечка UTC {co.strftime('%Y-%m-%d %H:%M:%S')}); "
            "без поля timestamp — оставляем."
        )
        if age_stats:
            print(
                f"  До отсева: {age_stats['fetched']} шт.; "
                f"снято как старые: {age_stats['dropped_too_old']}; "
                f"без timestamp: {age_stats['kept_missing_timestamp']}"
            )
        print(bar)
    if not items:
        print("  (пусто)\n")
        return

    for i, p in enumerate(items, 1):
        comments = p.get("commentsCount") or 0
        passes = comments >= min_comments
        kind = "reel" if is_reel_payload(p) else "post"
        latest = p.get("latestComments") or []
        latest_preview = ""
        if latest:
            bits = []
            for c in latest[:3]:
                u = c.get("ownerUsername") or "?"
                t = _truncate(c.get("text"), 40)
                bits.append(f"@{u}: {t}")
            latest_preview = " | ".join(bits)

        print(f"\n  --- item {i}/{len(items)} ({kind}) ---")
        print(f"  Ссылка:           {p.get('url') or '—'}")
        print(f"  shortCode:        {p.get('shortCode') or '—'}")
        print(f"  inputUrl (тег):  {p.get('inputUrl') or '—'}")
        print(f"  Автор:            @{p.get('ownerUsername') or '?'} ({p.get('ownerFullName') or '—'})  id={p.get('ownerId') or '—'}")
        print(f"  Тип:              type={p.get('type') or '—'}  productType={p.get('productType') or '—'}")
        print(f"  Комментарии:      {comments}  (порог {min_comments}: {'да' if passes else 'нет'})")
        print(f"  Лайки / просмотры: {p.get('likesCount') or 0} / {p.get('videoViewCount') or 0}")
        print(f"  Время:            {p.get('timestamp') or '—'}")
        print(f"  Локация:          {p.get('locationName') or '—'}")
        print(f"  Подпись:          {_truncate(p.get('caption'), 400)}")
        print(f"  Хештеги в посте:  {_fmt_list(p.get('hashtags'), 200)}")
        fc = p.get("firstComment")
        if fc:
            print(f"  firstComment:     {_truncate(str(fc), 120)}")
        print(f"  latestComments:   {len(latest)} шт.  {latest_preview or '—'}")
        print(f"  displayUrl:       {_truncate(p.get('displayUrl'), 90)}")

    print(f"\n{bar}\n")


def _print_human_report(
    *,
    hashtags: list[str],
    limit: int,
    min_comments: int,
    max_age_days: int,
    posts_only: bool,
    posts_raw: list[dict],
    reels_raw: list[dict],
    posts_age_stats: dict[str, int] | None,
    reels_age_stats: dict[str, int] | None,
    normalized: list[dict],
    qualifying: list[dict],
    out_path: Path,
    pipeline_log: Path,
    cost_usd: float | None,
    sample_keys_from: dict | None,
) -> None:
    line = "=" * 62
    print(f"\n{line}")
    print("  instagram-hashtag-scraper — итог прогона (test_hashtag_step1)")
    print(line)
    print(_ACTOR_OUTPUT_BLURB_RU)
    if sample_keys_from:
        keys = ", ".join(sorted(sample_keys_from.keys()))
        print(f"  Пример полей одного item (ключи): {keys[:500]}{'…' if len(keys) > 500 else ''}")
    print(line)
    print("  Параметры")
    print(f"    Хештеги ({len(hashtags)}): {', '.join('#' + h for h in hashtags)}")
    print(f"    resultsLimit на тип выдачи: {limit}")
    print(f"    Порог комментариев (как pipeline.step1): >= {min_comments}")
    if max_age_days > 0:
        print(
            f"    Окно по дате (как pipeline.step1, по timestamp после рана): "
            f"≤ {max_age_days} дн."
        )
    else:
        print("    Окно по дате: выключено (--no-age-filter)")
    print(f"    Режим: {'только posts' if posts_only else 'posts + reels (2 рана)'}")
    print(line)
    print("  Статистика")
    if max_age_days > 0 and posts_age_stats:
        print(
            f"    Posts: с актора {posts_age_stats['fetched']} → после даты {len(posts_raw)} "
            f"(снято {posts_age_stats['dropped_too_old']})"
        )
    else:
        print(f"    Posts после даты/без фильтра: {len(posts_raw)}")
    if not posts_only and max_age_days > 0 and reels_age_stats:
        print(
            f"    Reels: с актора {reels_age_stats['fetched']} → после даты {len(reels_raw)} "
            f"(снято {reels_age_stats['dropped_too_old']})"
        )
    elif not posts_only:
        print(f"    Reels после даты/без фильтра: {len(reels_raw)}")
    elif posts_only:
        print("    Reels: не запускали (--posts-only)")
    print(f"    Итого в выборке: posts={len(posts_raw)}, reels={len(reels_raw)}, всего={len(normalized)}")
    print(f"    Проходят порог по комментариям: {len(qualifying)}")
    print(f"    Apify (эта сессия скрипта): ${cost_usd or 0:.4f}")
    print(line)
    print("  Файлы")
    print(f"    JSON: {out_path.resolve()}")
    print(f"    Лог пайплайна: {pipeline_log.resolve()}")
    print(line)
    print("  (Детальный разбор каждого item — в блоках выше после каждого рана актора.)")
    print(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch posts/reels via instagram-hashtag-scraper.")
    parser.add_argument(
        "--posts-only",
        action="store_true",
        help="Skip reels run (only resultsType=posts).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override per-hashtag resultsLimit for this run "
            f"(default: {DEFAULT_APIFY_WRAPPER_LIMITS['results_limit']})."
        ),
    )
    parser.add_argument(
        "--no-age-filter",
        action="store_true",
        help="Keep all items from the actor (no max-age cut on timestamp).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=None,
        metavar="N",
        help="Override pipeline.step1.posts_max_age_days for this run.",
    )
    args = parser.parse_args()

    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "test_hashtag_step1")
    apify = ApifyWrapper(cfg, db, pipeline)

    hashtags = cfg["search"]["hashtags"]
    limit = args.limit if args.limit is not None else apify.limits["results_limit"]
    min_comments = _min_comments_from_config(cfg)
    if args.no_age_filter:
        max_age_days = 0
    elif args.max_age_days is not None:
        max_age_days = int(args.max_age_days)
    else:
        max_age_days = _posts_max_age_days_from_config(cfg)
    if max_age_days < 0:
        max_age_days = 0

    log.info(
        "hashtag_run_start",
        hashtags=hashtags,
        results_limit=limit,
        min_comments=min_comments,
        max_age_days=max_age_days,
    )

    posts_fetched = apify.search_by_hashtag(hashtags, results_type="posts", limit=limit)
    posts_raw, posts_age_stats = filter_items_within_max_age(posts_fetched, max_age_days)
    _print_actor_run_items(
        "Ран 1: resultsType=posts (instagram-hashtag-scraper), после фильтра по дате",
        posts_raw,
        min_comments=min_comments,
        max_age_days=max_age_days,
        age_stats=posts_age_stats if max_age_days > 0 else None,
    )

    reels_fetched: list[dict] = []
    reels_raw: list[dict] = []
    reels_age_stats: dict[str, int] | None = None
    if not args.posts_only:
        reels_fetched = apify.search_by_hashtag(hashtags, results_type="reels", limit=limit)
        reels_raw, reels_age_stats = filter_items_within_max_age(reels_fetched, max_age_days)
        _print_actor_run_items(
            "Ран 2: resultsType=reels (instagram-hashtag-scraper), после фильтра по дате",
            reels_raw,
            min_comments=min_comments,
            max_age_days=max_age_days,
            age_stats=reels_age_stats if max_age_days > 0 else None,
        )

    combined = [("hashtag_posts", posts_raw), ("hashtag_reels", reels_raw)]
    normalized: list[dict] = []
    for label, items in combined:
        for p in items:
            normalized.append(
                _normalize_item(p, source_label=label, matched_hashtags=hashtags)
            )

    qualifying = [x for x in normalized if x["comments_count"] >= min_comments]

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = data_dir / f"hashtag_step1_{ts}.json"
    payload = {
        "written_at": ts,
        "hashtags": hashtags,
        "results_limit_per_type": limit,
        "min_comments_gate": min_comments,
        "age_filter": {
            "max_age_days": max_age_days,
            "note": (
                "instagram-hashtag-scraper has no onlyPostsNewerThan; "
                "filter applied client-side on each item timestamp (UTC)."
            ),
            "posts": posts_age_stats,
            "reels": reels_age_stats,
        },
        "counts": {
            "posts_after_age": len(posts_raw),
            "reels_after_age": len(reels_raw),
            "combined": len(normalized),
            "passing_min_comments": len(qualifying),
        },
        "items_passing_gate": qualifying,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ps = pipeline.summary()
    log.info(
        "hashtag_run_done",
        out_path=str(out_path),
        **payload["counts"],
        apify_cost_usd=ps.get("total_cost_usd"),
    )

    sample_src = (
        posts_fetched[0]
        if posts_fetched
        else (reels_fetched[0] if reels_fetched else None)
    )
    _print_human_report(
        hashtags=hashtags,
        limit=limit,
        min_comments=min_comments,
        max_age_days=max_age_days,
        posts_only=args.posts_only,
        posts_raw=posts_raw,
        reels_raw=reels_raw,
        posts_age_stats=posts_age_stats,
        reels_age_stats=reels_age_stats,
        normalized=normalized,
        qualifying=qualifying,
        out_path=out_path,
        pipeline_log=Path(pipeline.file_path),
        cost_usd=ps.get("total_cost_usd"),
        sample_keys_from=sample_src,
    )


if __name__ == "__main__":
    main()
