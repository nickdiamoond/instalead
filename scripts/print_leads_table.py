"""Print Instagram usernames from lead_accounts, one per line.

Usage:
    python scripts/print_leads_table.py
    python scripts/print_leads_table.py --order username
    python scripts/print_leads_table.py --db data/leads.db > usernames.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.db_table_printer import ensure_utf8_stdout
from src.config import load_config

_ORDER_CHOICES = {
    "discovered": "discovered_at DESC, username",
    "username": "username",
    "followers": "followers_count DESC NULLS LAST, username",
    "profile": "profile_fetched DESC, profile_fetched_at DESC, username",
    "sherlock": "sherlock_processed_at DESC NULLS LAST, username",
    "contact": "contact_found DESC, contact_found_at DESC, username",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="path to leads.db (default: config db.path)")
    parser.add_argument(
        "--order",
        choices=sorted(_ORDER_CHOICES),
        default="username",
        help="sort order (default: alphabetical by username)",
    )
    args = parser.parse_args()

    db_path = args.db
    if not db_path:
        cfg = load_config()
        db_path = cfg.get("db", {}).get("path", "data/leads.db")

    ensure_utf8_stdout()
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            f"SELECT username FROM lead_accounts ORDER BY {_ORDER_CHOICES[args.order]}"
        )
        for (username,) in cur:
            if username:
                print(username)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
