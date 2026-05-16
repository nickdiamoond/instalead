"""One-shot recovery: reset comment-scan state for posts that were marked
scanned but never produced any leads.

Background
----------
``louisdeconinck/instagram-comments-scraper`` started silently returning
0 comments for every URL while still completing successfully (its own
log contains ``fetched 0/null comments`` per post). Step 3 of the
pipeline was unconditionally calling
:py:meth:`LeadDB.mark_post_comments_scanned` after each run, so all
``relevant + cta=comment`` posts ended up with
``last_comments_count = real_comments_count`` despite no leads ever
being saved. ``get_posts_needing_comments`` then keeps them out of the
queue until comments grow another 5%.

This script clears ``last_scanned_at`` and ``last_comments_count`` for
the affected posts so the next pipeline run picks them up again. The
selector intentionally only resets posts that have **zero rows** in
``lead_post_links`` for that shortcode — a successful past scan would
have produced at least one row, so leaving those alone avoids paying
for re-scans of already-harvested posts.

Idempotent: running it twice is a no-op once the queue has been
re-scanned successfully.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SELECT_AFFECTED = """
    SELECT pp.post_id,
           pp.owner_username,
           pp.comments_count,
           pp.last_comments_count,
           pp.last_scanned_at
    FROM processed_posts pp
    WHERE pp.relevance = 'relevant'
      AND pp.cta_type  = 'comment'
      AND pp.last_scanned_at IS NOT NULL
      AND NOT EXISTS (
            SELECT 1 FROM lead_post_links lpl
            WHERE lpl.post_shortcode = pp.post_id
      )
    ORDER BY pp.comments_count DESC
"""

UPDATE_AFFECTED = """
    UPDATE processed_posts
    SET last_scanned_at = NULL,
        last_comments_count = NULL
    WHERE relevance = 'relevant'
      AND cta_type  = 'comment'
      AND last_scanned_at IS NOT NULL
      AND NOT EXISTS (
            SELECT 1 FROM lead_post_links lpl
            WHERE lpl.post_shortcode = processed_posts.post_id
      )
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/leads.db", help="path to leads.db")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write the reset (default is dry-run)",
    )
    parser.add_argument(
        "--limit-preview",
        type=int,
        default=15,
        help="how many candidate rows to print in the preview",
    )
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    candidates = list(con.execute(SELECT_AFFECTED))
    print(f"Posts that would be reset: {len(candidates)}")
    if not candidates:
        print("Nothing to do — DB already clean.")
        return 0

    total_lost = sum((r["comments_count"] or 0) for r in candidates)
    print(f"Sum of comments_count on those posts: {total_lost}")
    print()
    print(f"First {args.limit_preview} (by comments_count desc):")
    for r in candidates[: args.limit_preview]:
        print(
            f"  {r['post_id']:<12} "
            f"@{(r['owner_username'] or '?'):<22} "
            f"comments={r['comments_count']:<6} "
            f"last_scanned={r['last_scanned_at']}"
        )
    if len(candidates) > args.limit_preview:
        print(f"  … and {len(candidates) - args.limit_preview} more")

    if not args.apply:
        print()
        print("Dry-run only. Re-run with --apply to commit the reset.")
        return 0

    con.execute(UPDATE_AFFECTED)
    con.commit()
    print()
    print(f"Reset {len(candidates)} posts. They will reappear in the next "
          f"pipeline Step 3 queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
