"""Find realtor accounts through Instagram's relatedProfiles (snowball method).

Takes seed accounts → gets relatedProfiles → filters by SPB keywords.
Budget-limited: stops when cost exceeds MAX_COST_USD.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apify_client_wrapper import ApifyWrapper
from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("related_realtors")

SEEDS = [
    "trushin_ilya_realty",
    "elizaveta_dianova",
    "alinalvv_realtor",
]

MAX_ROUNDS = 2
MAX_COST_USD = 0.50

# Related profiles only have full_name, no bio.
# Full profiles have both. We check whatever is available.
SPB_KEYWORDS = re.compile(
    r"спб|питер|петербург|санкт|spb|piter|peter|pieter",
    re.IGNORECASE,
)
REALTY_KEYWORDS = re.compile(
    r"недвижимост|риелтор|риэлтор|realt|квартир|новостройк|жильё|жилье|агент|ипотек|estate",
    re.IGNORECASE,
)


def matches_spb(text: str | None) -> bool:
    return bool(text and SPB_KEYWORDS.search(text))


def matches_realty(text: str | None) -> bool:
    return bool(text and REALTY_KEYWORDS.search(text))


def is_strict_related(r: dict) -> bool:
    """Check related profile by full_name: must have BOTH spb AND realty keywords."""
    name = r.get("full_name") or ""
    return matches_spb(name) and matches_realty(name)


def is_loose_related(r: dict) -> bool:
    """Looser check for queuing: spb OR realty in full_name."""
    name = r.get("full_name") or ""
    return matches_spb(name) or (matches_realty(name) and not r.get("is_private"))


def is_relevant_full(p: dict) -> bool:
    """Check full profile by full_name + bio + business_category."""
    text = " ".join(filter(None, [
        p.get("fullName") or p.get("full_name"),
        p.get("biography"),
        p.get("businessCategoryName") or p.get("business_category"),
        p.get("username"),
    ]))
    return matches_spb(text) and matches_realty(text)


def extract_profile_info(p: dict, found_via: str) -> dict:
    return {
        "username": p.get("username"),
        "full_name": p.get("fullName"),
        "biography": p.get("biography"),
        "followers_count": p.get("followersCount"),
        "posts_count": p.get("postsCount"),
        "is_private": p.get("private"),
        "is_verified": p.get("verified"),
        "is_business": p.get("isBusinessAccount"),
        "business_category": p.get("businessCategoryName"),
        "external_url": p.get("externalUrl"),
        "profile_pic_url": p.get("profilePicUrlHD") or p.get("profilePicUrl"),
        "found_via": found_via,
    }


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "related_realtors")
    apify = ApifyWrapper(cfg, db, pipeline)

    relevant_accounts: dict[str, dict] = {}
    candidates: dict[str, dict] = {}  # strict match by name, no full profile yet
    processed = set()

    # Seeds always go in
    queued = list(SEEDS)

    for round_num in range(1, MAX_ROUNDS + 1):
        batch = [u for u in queued if u not in processed]
        if not batch:
            log.info("no_new_accounts", round=round_num)
            break

        # Budget check
        current_cost = pipeline.summary()["total_cost_usd"]
        if current_cost >= MAX_COST_USD:
            log.warning("budget_exceeded", cost=current_cost, limit=MAX_COST_USD)
            break

        log.info("round_start", round=round_num, batch_size=len(batch))
        profiles = apify.get_profiles_batch(batch)
        processed.update(batch)

        next_queue = []
        for p in profiles:
            username = p.get("username")
            if not username:
                continue

            is_seed = username in SEEDS
            relevant = is_seed or is_relevant_full(p)

            log.info(
                "profile",
                username=username,
                relevant=relevant,
                followers=p.get("followersCount"),
                bio=str(p.get("biography", ""))[:80],
            )

            if relevant:
                found_via = "seed" if is_seed else f"related_round_{round_num}"
                relevant_accounts[username] = extract_profile_info(p, found_via)

            # Collect related — pre-filter by full_name
            for r in (p.get("relatedProfiles") or []):
                r_user = r.get("username")
                if not r_user or r_user in processed or r_user in relevant_accounts:
                    continue

                # Strict match (both spb + realty) → save as candidate
                if is_strict_related(r):
                    candidates[r_user] = {
                        "username": r_user,
                        "full_name": r.get("full_name"),
                        "is_private": r.get("is_private"),
                        "profile_pic_url": r.get("profile_pic_url"),
                        "found_via": f"related_of_{username}",
                    }
                    log.info("candidate_strict", username=r_user, name=r.get("full_name"))

                # Loose match → queue for next round
                if is_loose_related(r):
                    next_queue.append(r_user)

        queued = next_queue
        log.info(
            "round_done",
            round=round_num,
            relevant_found=len(relevant_accounts),
            queued_next=len(queued),
            cost=pipeline.summary()["total_cost_usd"],
        )

    # Remove candidates that got full profiles
    for u in relevant_accounts:
        candidates.pop(u, None)

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out_path = Path("data") / f"related_realtors_{ts}.json"
    accounts_list = sorted(
        relevant_accounts.values(),
        key=lambda x: x.get("followers_count") or 0,
        reverse=True,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(accounts_list, f, ensure_ascii=False, indent=2)

    candidates_path = Path("data") / f"realtor_candidates_{ts}.json"
    candidates_list = sorted(candidates.values(), key=lambda x: x.get("full_name") or "")
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates_list, f, ensure_ascii=False, indent=2)

    ps = pipeline.summary()
    print(f"\nFound {len(accounts_list)} verified SPB realtor accounts")
    print(f"Found {len(candidates_list)} candidates (strict name match, no full profile)")
    print(f"Rounds: {min(round_num, MAX_ROUNDS)}, API calls: {ps['total_runs']}")
    print(f"Total cost: ${ps['total_cost_usd']:.4f}")
    print(f"\nVerified:   {out_path}")
    print(f"Candidates: {candidates_path}")


if __name__ == "__main__":
    main()
