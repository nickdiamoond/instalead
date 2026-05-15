"""Print post URLs from processed_posts, one per line.

Usage:
    python scripts/print_posts_table.py
    python scripts/print_posts_table.py --order comments
    python scripts/print_posts_table.py --db data/leads.db > post_urls.txt
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
    "processed_at": "processed_at DESC, post_url",
    "comments": "comments_count DESC NULLS LAST, post_url",
    "relevance": "relevance, cta_type, comments_count DESC, post_url",
    "owner": "owner_username, processed_at DESC, post_url",
    "post_url": "post_url",
    "post_id": "post_id",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="path to leads.db (default: config db.path)")
    parser.add_argument(
        "--order",
        choices=sorted(_ORDER_CHOICES),
        default="post_url",
        help="sort order (default: alphabetical by post_url)",
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
            f"SELECT post_url FROM processed_posts ORDER BY {_ORDER_CHOICES[args.order]}"
        )
        for (post_url,) in cur:
            if post_url:
                print(post_url)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
