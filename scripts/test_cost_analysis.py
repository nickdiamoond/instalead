"""Analyze costs from pipeline JSON logs.

Reads all JSON log files from logs/ and prints a summary table.
No Apify calls — works purely on previously saved data.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging

setup_logging()
log = get_logger("cost_analysis")


def load_all_runs(log_dir: str = "logs") -> list[dict]:
    """Load all run records from all JSON log files."""
    runs = []
    for f in sorted(Path(log_dir).glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            for record in data:
                record["_source_file"] = f.name
            runs.extend(data)
    return runs


def main():
    cfg = load_config()
    log_dir = cfg.get("logging", {}).get("pipeline_log_dir", "logs")

    runs = load_all_runs(log_dir)
    if not runs:
        log.warning("no_logs", msg=f"No JSON logs found in {log_dir}/")
        return

    # Group by actor
    by_actor: dict[str, list[dict]] = {}
    for r in runs:
        actor = r.get("actor_id", "unknown")
        by_actor.setdefault(actor, []).append(r)

    print("\n" + "=" * 90)
    print(f"{'ACTOR':<45} {'RUNS':>5} {'ITEMS':>7} {'COST $':>9} {'$/ITEM':>10}")
    print("=" * 90)

    grand_cost = 0
    grand_items = 0

    for actor, actor_runs in sorted(by_actor.items()):
        total_items = sum(r.get("items_count", 0) for r in actor_runs)
        total_cost = sum(r.get("cost_usd") or 0 for r in actor_runs)
        cost_per_item = total_cost / total_items if total_items else 0

        print(
            f"{actor:<45} {len(actor_runs):>5} {total_items:>7} "
            f"{total_cost:>9.4f} {cost_per_item:>10.6f}"
        )

        grand_cost += total_cost
        grand_items += total_items

    print("-" * 90)
    print(
        f"{'TOTAL':<45} {len(runs):>5} {grand_items:>7} "
        f"{grand_cost:>9.4f} "
        f"{grand_cost / grand_items if grand_items else 0:>10.6f}"
    )
    print("=" * 90)

    # DB stats
    db = LeadDB(cfg["db"]["path"])
    db_stats = db.get_stats()
    print(f"\nDB stats: {db_stats}")

    # Estimate cycle cost
    print("\n--- Cycle cost estimate ---")
    if grand_items and grand_cost:
        avg_per_item = grand_cost / grand_items
        max_accounts = cfg.get("cycle", {}).get("max_accounts_per_cycle", 200)
        est = avg_per_item * max_accounts
        print(f"Avg cost per item:    ${avg_per_item:.6f}")
        print(f"Max accounts/cycle:   {max_accounts}")
        print(f"Estimated cycle cost: ${est:.2f}")


if __name__ == "__main__":
    main()
