from apify_client import ApifyClient

from src.comment_normalizer import normalize_apidojo_api
from src.pipeline_logger import PipelineLogger
from src.telegram_notifier import PipelineTelegramNotifier

from scripts.pipeline_lib.logging import log


def _run_apify_actor(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    actor_id: str,
    run_input: dict,
    *,
    log_input: dict | None = None,
    tg_notifier: PipelineTelegramNotifier | None = None,
    apify_step: str | None = None,
) -> tuple[list[dict], float, dict]:
    """Run an Apify actor and return ``(items, cost_usd, run_meta)``.

    Centralizes the boilerplate of ``actor.call`` -> ``run.get`` ->
    ``dataset.iterate_items`` -> ``pipeline.log_run`` so Step 3's
    primary/fallback split doesn't duplicate it. ``log_input`` is what
    gets persisted to the pipeline JSON log -- usually a sanitized
    summary like ``{"urls_count": N}`` rather than the full URL list.
    """
    run = apify.actor(actor_id).call(run_input=run_input)
    detail = apify.run(run["id"]).get() or {}
    cost = detail.get("usageTotalUsd") or 0.0
    items: list[dict] = []
    dataset_id = run.get("defaultDatasetId")
    if dataset_id:
        items = list(apify.dataset(dataset_id).iterate_items())
    pipeline.log_run(
        actor_id=actor_id,
        run_id=run["id"],
        status=run["status"],
        input_params=log_input or run_input,
        items_count=len(items),
        cost_usd=cost,
        duration_ms=detail.get("stats", {}).get("durationMillis"),
    )
    if tg_notifier is not None and apify_step:
        tg_notifier.maybe_notify_apify_run_failure(
            run, actor_id=actor_id, step=apify_step
        )
    return items, cost, run


def _fetch_comments_with_fallback(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    urls: list[str],
    *,
    primary_actor: str,
    fallback_actor: str,
    louisdeconinck_cap_per_post: int,
    apidojo_cap_per_post: int,
    tg_notifier: PipelineTelegramNotifier | None = None,
) -> tuple[list[dict], float, str, dict]:
    """Pull comments for ``urls`` with primary -> apidojo-api fallback.

    Returns ``(items, total_cost, source, debug)`` where:

    * ``items`` is a list of louisdeconinck-shaped dicts (the apidojo-api
      branch normalizes via
      :func:`src.comment_normalizer.normalize_apidojo_api` so the
      caller's dedup / save loop is actor-agnostic).
    * ``total_cost`` is primary + fallback ``usageTotalUsd`` summed.
    * ``source`` is one of ``"primary"`` / ``"fallback"`` /
      ``"both-empty"`` -- the caller uses ``"both-empty"`` to leave
      ``processed_posts.last_scanned_at`` untouched so the queue keeps
      retrying instead of silently freezing (the same guard the script
      had before the fallback was added).
    * ``debug`` carries metadata each branch may want to surface in
      banners / issues -- ``primary_run_id``, ``primary_cost``,
      ``primary_items``, plus ``fallback_*`` if the fallback fired.

    Both Apify runs are logged separately via ``pipeline.log_run`` so
    the per-actor cost split stays explicit in ``logs/pipeline_*.json``.

    The actor ids and the per-post comment cap are passed in (rather
    than read from module-level constants) so ``main()`` can override
    them from ``config.yaml`` (``pipeline.step3.*``) without touching
    this function.
    """
    primary_items, primary_cost, primary_run = _run_apify_actor(
        apify,
        pipeline,
        primary_actor,
        run_input={
            "urls": urls,
            "proxy": {"useApifyProxy": True},
            "resultsLimit": louisdeconinck_cap_per_post,
            "maxComments": louisdeconinck_cap_per_post,
        },
        log_input={
            "urls_count": len(urls),
            "results_limit": louisdeconinck_cap_per_post,
        },
        tg_notifier=tg_notifier,
        apify_step="Step 3 (comments primary)",
    )
    debug = {
        "primary_actor": primary_actor,
        "primary_run_id": primary_run["id"],
        "primary_status": primary_run["status"],
        "primary_items": len(primary_items),
        "primary_cost": primary_cost,
    }

    if primary_items:
        return primary_items, primary_cost, "primary", debug

    log.warning(
        "step3_primary_empty_falling_back",
        actor=primary_actor,
        fallback=fallback_actor,
        urls=len(urls),
        run_id=primary_run["id"],
        primary_cost=primary_cost,
        msg="primary returned 0 items, retrying via fallback",
    )

    apidojo_max_items = apidojo_cap_per_post * max(len(urls), 1)
    fb_raw, fb_cost, fb_run = _run_apify_actor(
        apify,
        pipeline,
        fallback_actor,
        run_input={
            "startUrls": urls,
            "proxy": {"useApifyProxy": True},
            "maxItems": apidojo_max_items,
        },
        log_input={
            "startUrls_count": len(urls),
            "max_items": apidojo_max_items,
            "apidojo_cap_per_post": apidojo_cap_per_post,
            "fallback": True,
        },
        tg_notifier=tg_notifier,
        apify_step="Step 3 (comments fallback)",
    )
    debug.update(
        {
            "fallback_actor": fallback_actor,
            "fallback_run_id": fb_run["id"],
            "fallback_status": fb_run["status"],
            "fallback_raw_items": len(fb_raw),
            "fallback_cost": fb_cost,
        }
    )

    fb_items = [
        normalized
        for normalized in (normalize_apidojo_api(it) for it in fb_raw)
        if normalized is not None
    ]
    debug["fallback_normalized_items"] = len(fb_items)
    total_cost = primary_cost + fb_cost

    if not fb_items:
        log.error(
            "step3_fallback_also_empty",
            primary=primary_actor,
            fallback=fallback_actor,
            primary_run_id=primary_run["id"],
            fallback_run_id=fb_run["id"],
            total_cost=total_cost,
        )
        return [], total_cost, "both-empty", debug

    log.info(
        "step3_fallback_recovered",
        actor=fallback_actor,
        raw=len(fb_raw),
        normalized=len(fb_items),
        primary_cost=primary_cost,
        fallback_cost=fb_cost,
    )
    return fb_items, total_cost, "fallback", debug
