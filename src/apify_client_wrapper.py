"""Thin wrapper around ApifyClient with structlog logging and pipeline JSON tracking."""

import json

from apify_client import ApifyClient

from src.db import LeadDB
from src.logger import get_logger
from src.pipeline_logger import PipelineLogger

log = get_logger("apify")


class ApifyWrapper:
    """Runs Apify actors with automatic logging, cost tracking, and dedup."""

    def __init__(self, config: dict, db: LeadDB, pipeline: PipelineLogger):
        self.config = config
        self.token = config["apify"]["token"]
        self.client = ApifyClient(self.token)
        self.db = db
        self.pipeline = pipeline

        self.proxy = (
            {"useApifyProxy": True}
            if config["apify"]["proxy"]["use_apify_proxy"]
            else {}
        )
        self.actors = config["apify"]["actors"]
        self.limits = config["apify"]["test_limits"]

    def run_actor(
        self,
        actor_id: str,
        run_input: dict,
        *,
        max_items: int | None = None,
        sample_count: int = 3,
    ) -> list[dict]:
        """Run an actor and return dataset items. Logs everything."""
        run_input.setdefault("proxy", self.proxy)

        log.info("actor_starting", actor=actor_id, input_keys=list(run_input.keys()))

        run = self.client.actor(actor_id).call(run_input=run_input)
        run_id = run["id"]
        status = run["status"]

        log.info("actor_finished", actor=actor_id, run_id=run_id, status=status)

        items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

        if max_items and len(items) > max_items:
            items = items[:max_items]

        # Get cost info
        run_detail = self.client.run(run_id).get()
        cost_usd = run_detail.get("usageTotalUsd")
        duration_ms = run_detail.get("stats", {}).get("durationMillis")

        log.info(
            "run_stats",
            actor=actor_id,
            run_id=run_id,
            items=len(items),
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        # Log to DB
        self.db.log_apify_run(
            run_id=run_id,
            actor_id=actor_id,
            status=status,
            items_count=len(items),
            cost_usd=cost_usd,
            input_summary=json.dumps(
                {k: v for k, v in run_input.items() if k != "proxy"},
                ensure_ascii=False,
                default=str,
            ),
        )

        # Log to pipeline JSON
        samples = items[:sample_count] if items else []
        self.pipeline.log_run(
            actor_id=actor_id,
            run_id=run_id,
            status=status,
            input_params=run_input,
            items_count=len(items),
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            dataset_id=run.get("defaultDatasetId"),
            sample_items=samples,
        )

        return items

    # --- Convenience methods ---

    def search_by_hashtag(
        self, hashtags: list[str], results_type: str = "posts", limit: int | None = None
    ) -> list[dict]:
        """Get posts OR reels for hashtags via dedicated hashtag scraper."""
        return self.run_actor(
            self.actors["hashtag"],
            {
                "hashtags": hashtags,
                "resultsType": results_type,
                "resultsLimit": limit or self.limits["results_limit"],
            },
        )

    def search_by_hashtag_all(
        self, hashtags: list[str], limit: int | None = None
    ) -> list[dict]:
        """Get BOTH posts and reels for hashtags (two API calls)."""
        posts = self.search_by_hashtag(hashtags, results_type="posts", limit=limit)
        reels = self.search_by_hashtag(hashtags, results_type="reels", limit=limit)
        return posts + reels

    def search_users(
        self, query: str, limit: int | None = None
    ) -> list[dict]:
        """Search for Instagram user accounts by keyword."""
        return self.run_actor(
            self.actors["universal"],
            {
                "search": query,
                "searchType": "user",
                "searchLimit": limit or self.limits["search_limit"],
                "resultsType": "details",
            },
        )

    def get_account_posts(
        self, username: str, limit: int | None = None
    ) -> list[dict]:
        """Get recent posts/reels for a specific Instagram account."""
        return self.get_accounts_posts_batch([username], limit)

    def get_accounts_posts_batch(
        self, usernames: list[str], limit: int | None = None, max_age_days: int | None = None
    ) -> list[dict]:
        """Get recent posts/reels for multiple accounts in one API call."""
        age = max_age_days or self.config["filters"]["max_post_age_days"]
        return self.run_actor(
            self.actors["posts"],
            {
                "username": usernames,
                "resultsLimit": limit or self.limits["results_limit"],
                "onlyPostsNewerThan": f"{age} days",
                "dataDetailLevel": "basicData",
            },
        )

    def get_comments(
        self, post_url: str, limit: int | None = None
    ) -> list[dict]:
        """Get comments for a specific post/reel."""
        return self.run_actor(
            self.actors["comments"],
            {
                "directUrls": [post_url],
                "resultsLimit": limit or self.limits["comments_limit"],
            },
        )

    def get_profile(self, username: str) -> list[dict]:
        """Get profile info for a username."""
        return self.run_actor(
            self.actors["profile"],
            {"usernames": [username]},
        )

    def get_profiles_batch(self, usernames: list[str]) -> list[dict]:
        """Get profile info for multiple usernames in one call."""
        return self.run_actor(
            self.actors["profile"],
            {"usernames": usernames},
        )
