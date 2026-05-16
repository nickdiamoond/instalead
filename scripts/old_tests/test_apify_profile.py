"""Test script: get Instagram profile info for given usernames.

Pass usernames as arguments or it will use a hardcoded example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger
from src.apify_client_wrapper import ApifyWrapper

setup_logging()
log = get_logger("test_profile")


def main():
    usernames = sys.argv[1:] if len(sys.argv) > 1 else []

    if not usernames:
        log.error(
            "no_usernames",
            msg="Pass Instagram usernames as arguments: "
                "python scripts/test_apify_profile.py user1 user2",
        )
        return

    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "test_profile")

    apify = ApifyWrapper(cfg, db, pipeline)

    log.info("fetching_profiles", usernames=usernames)
    profiles = apify.get_profiles_batch(usernames)

    for p in profiles:
        log.info(
            "profile",
            username=p.get("username"),
            full_name=p.get("fullName"),
            bio=str(p.get("biography", ""))[:120],
            followers=p.get("followersCount"),
            following=p.get("followsCount"),
            posts=p.get("postsCount"),
            is_private=p.get("isPrivate"),
            is_verified=p.get("isVerified"),
            profile_pic=p.get("profilePicUrlHD") or p.get("profilePicUrl"),
            external_url=p.get("externalUrl"),
        )

        # Log all available keys for the first profile (discover schema)
        if p == profiles[0]:
            log.info("profile_all_keys", keys=sorted(p.keys()))

    summary = pipeline.summary()
    log.info("session_summary", **summary)
    print(f"\nPipeline log saved to: {pipeline.file_path}")


if __name__ == "__main__":
    main()
