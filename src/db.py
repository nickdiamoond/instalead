"""SQLite storage for leads, realtors, posts, and run tracking."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class LeadDB:
    def __init__(self, db_path: str = "data/leads.db"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._persistent_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
        self.init_tables()

    @contextmanager
    def _conn(self):
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_tables(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tracked_realtors (
                    username        TEXT PRIMARY KEY,
                    full_name       TEXT,
                    followers_count INTEGER,
                    found_via       TEXT,
                    added_at        TEXT NOT NULL,
                    is_active       INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS lead_accounts (
                    username            TEXT PRIMARY KEY,
                    user_id             TEXT,
                    full_name           TEXT,
                    biography           TEXT,
                    profile_pic_url     TEXT,
                    profile_pic_url_hd  TEXT,
                    is_private          INTEGER,
                    is_verified         INTEGER,
                    is_business         INTEGER,
                    business_category   TEXT,
                    followers_count     INTEGER,
                    following_count     INTEGER,
                    posts_count         INTEGER,
                    external_url        TEXT,
                    latest_media_urls   TEXT,

                    -- avatar / face detection (Module 2 prep)
                    avatar_path         TEXT,
                    faces_count         INTEGER,

                    -- contacts (filled by bio parsing or Module 2)
                    phone               TEXT,
                    email               TEXT,
                    telegram_username   TEXT,
                    whatsapp            TEXT,

                    -- processing state
                    profile_fetched     INTEGER DEFAULT 0,
                    contact_found       INTEGER DEFAULT 0,
                    discovered_at       TEXT NOT NULL,
                    profile_fetched_at  TEXT,
                    contact_found_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS lead_post_links (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT,
                    username        TEXT NOT NULL,
                    post_url        TEXT NOT NULL,
                    post_shortcode  TEXT,
                    comment_text    TEXT,
                    comment_at      TEXT,
                    UNIQUE(username, post_url)
                );

                CREATE TABLE IF NOT EXISTS processed_posts (
                    post_id             TEXT PRIMARY KEY,
                    post_url            TEXT NOT NULL,
                    shortcode           TEXT,
                    owner_username      TEXT,
                    comments_count      INTEGER,
                    likes_count         INTEGER,
                    views_count         INTEGER,
                    post_type           TEXT,
                    caption             TEXT,
                    relevance           TEXT,
                    has_cta             INTEGER,
                    cta_type            TEXT,
                    timestamp           TEXT,
                    last_comments_count INTEGER,
                    last_scanned_at     TEXT,
                    processed_at        TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS apify_runs (
                    run_id        TEXT PRIMARY KEY,
                    actor_id      TEXT NOT NULL,
                    started_at    TEXT,
                    finished_at   TEXT,
                    status        TEXT,
                    items_count   INTEGER,
                    cost_usd      REAL,
                    input_summary TEXT
                );
            """)
            self._migrate_add_columns(conn)

    def _migrate_add_columns(self, conn: sqlite3.Connection) -> None:
        """Idempotent ALTER TABLE for existing databases.

        SQLite does not support ADD COLUMN IF NOT EXISTS, so we inspect
        PRAGMA table_info and only add missing columns.
        """
        required = {
            "lead_accounts": [
                ("avatar_path", "TEXT"),
                ("faces_count", "INTEGER"),
            ],
        }
        for table, columns in required.items():
            existing = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                    )

    # --- tracked realtors ---

    def add_realtor(self, username: str, **kwargs) -> bool:
        if self.get_realtor(username):
            return False
        kwargs.setdefault("added_at", _now())
        cols = ["username"] + list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        vals = [username] + list(kwargs.values())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO tracked_realtors ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                vals,
            )
        return True

    def get_realtor(self, username: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tracked_realtors WHERE username = ?", (username,)
            ).fetchone()
            return dict(row) if row else None

    def get_active_realtors(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username FROM tracked_realtors WHERE is_active = 1"
            ).fetchall()
            return [r["username"] for r in rows]

    # --- lead accounts ---

    def is_account_known(self, username: str, user_id: str | None = None) -> bool:
        """Check if lead exists by username OR user_id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM lead_accounts WHERE username = ?", (username,)
            ).fetchone()
            if row:
                return True
            if user_id:
                row = conn.execute(
                    "SELECT 1 FROM lead_accounts WHERE user_id = ?", (user_id,)
                ).fetchone()
                return row is not None
            return False

    def add_lead_account(self, username: str, **kwargs) -> bool:
        """Add lead. Returns True if inserted, False if already existed.

        Checks both username and user_id for dedup (username can change).
        If user_id exists with a different username, updates the username.
        """
        user_id = kwargs.get("user_id")

        # Check by user_id first — username might have changed
        if user_id:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT username FROM lead_accounts WHERE user_id = ?", (user_id,)
                ).fetchone()
                if existing:
                    old_username = existing[0]
                    if old_username != username:
                        # Username changed — update it
                        conn.execute(
                            "UPDATE lead_accounts SET username = ? WHERE user_id = ?",
                            (username, user_id),
                        )
                    return False

        if self.is_account_known(username):
            return False

        kwargs.setdefault("discovered_at", _now())
        cols = ["username"] + list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        vals = [username] + list(kwargs.values())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO lead_accounts ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                vals,
            )
        return True

    def update_lead_profile(self, username: str, **kwargs) -> None:
        if not kwargs:
            return
        kwargs["profile_fetched"] = 1
        kwargs["profile_fetched_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [username]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE lead_accounts SET {set_clause} WHERE username = ?",
                vals,
            )

    def update_lead_contacts(self, username: str, **kwargs) -> None:
        if not kwargs:
            return
        kwargs["contact_found"] = 1
        kwargs["contact_found_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [username]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE lead_accounts SET {set_clause} WHERE username = ?",
                vals,
            )

    def add_lead_post_link(self, username: str, post_url: str, user_id: str | None = None, **kwargs) -> None:
        cols = ["username", "post_url"]
        vals = [username, post_url]
        if user_id:
            cols.append("user_id")
            vals.append(user_id)
        cols += list(kwargs.keys())
        vals += list(kwargs.values())
        placeholders = ", ".join(["?"] * len(cols))
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO lead_post_links ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                vals,
            )

    def update_lead_avatar(
        self, username: str, avatar_path: str, faces_count: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE lead_accounts SET avatar_path = ?, faces_count = ? "
                "WHERE username = ?",
                (avatar_path, faces_count, username),
            )

    def get_leads_needing_avatar(self, limit: int = 1000) -> list[dict]:
        """Leads that have profile data but no avatar processed yet."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, profile_pic_url_hd, profile_pic_url "
                "FROM lead_accounts "
                "WHERE profile_fetched = 1 "
                "  AND avatar_path IS NULL "
                "  AND COALESCE(is_private, 0) = 0 "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_leads_with_single_face(self, limit: int = 1000) -> list[dict]:
        """Leads whose avatar has exactly one detected face."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lead_accounts WHERE faces_count = 1 LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_leads_without_profile(self, limit: int = 100, max_age_days: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username FROM lead_accounts "
                "WHERE is_private = 0 AND ("
                "  profile_fetched = 0 "
                "  OR (profile_fetched_at IS NOT NULL "
                "      AND profile_fetched_at < datetime('now', ?)) "
                ") LIMIT ?",
                (f"-{max_age_days} days", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- processed posts ---

    def is_post_processed(self, post_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_posts WHERE post_id = ?", (post_id,)
            ).fetchone()
            return row is not None

    def get_post(self, post_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM processed_posts WHERE post_id = ?", (post_id,)
            ).fetchone()
            return dict(row) if row else None

    def upsert_post(self, post_id: str, **kwargs) -> None:
        """Insert or update a processed post."""
        existing = self.get_post(post_id)
        if existing:
            set_clause = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [post_id]
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE processed_posts SET {set_clause} WHERE post_id = ?",
                    vals,
                )
        else:
            kwargs.setdefault("processed_at", _now())
            kwargs.setdefault("post_url", "")
            cols = ["post_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            vals = [post_id] + list(kwargs.values())
            with self._conn() as conn:
                conn.execute(
                    f"INSERT INTO processed_posts ({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    vals,
                )

    def get_posts_needing_comments(self, min_growth_pct: float = 5.0) -> list[dict]:
        """Get relevant posts that need comment scanning.

        Returns posts where:
        - relevance=relevant AND cta_type=comment
        - AND (never scanned OR comments grew by min_growth_pct% since last scan)
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT post_id, post_url, shortcode, owner_username, "
                "       comments_count, last_comments_count, last_scanned_at "
                "FROM processed_posts "
                "WHERE relevance = 'relevant' AND cta_type = 'comment' "
                "  AND ("
                "    last_scanned_at IS NULL "
                "    OR (comments_count > last_comments_count * (1 + ? / 100.0))"
                "  )",
                (min_growth_pct,),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_post_comments_scanned(self, post_id: str, comments_count: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE processed_posts SET last_scanned_at = ?, last_comments_count = ? "
                "WHERE post_id = ?",
                (_now(), comments_count, post_id),
            )

    # --- apify runs ---

    def log_apify_run(self, run_id: str, actor_id: str, **kwargs) -> None:
        cols = ["run_id", "actor_id"] + list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        vals = [run_id, actor_id] + list(kwargs.values())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO apify_runs ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                vals,
            )

    # --- stats ---

    def get_stats(self) -> dict:
        with self._conn() as conn:
            leads = conn.execute("SELECT COUNT(*) FROM lead_accounts").fetchone()[0]
            leads_with_profile = conn.execute(
                "SELECT COUNT(*) FROM lead_accounts WHERE profile_fetched = 1"
            ).fetchone()[0]
            leads_with_contacts = conn.execute(
                "SELECT COUNT(*) FROM lead_accounts WHERE contact_found = 1"
            ).fetchone()[0]
            leads_with_avatar = conn.execute(
                "SELECT COUNT(*) FROM lead_accounts WHERE avatar_path IS NOT NULL"
            ).fetchone()[0]
            leads_with_single_face = conn.execute(
                "SELECT COUNT(*) FROM lead_accounts WHERE faces_count = 1"
            ).fetchone()[0]
            realtors = conn.execute(
                "SELECT COUNT(*) FROM tracked_realtors WHERE is_active = 1"
            ).fetchone()[0]
            posts = conn.execute("SELECT COUNT(*) FROM processed_posts").fetchone()[0]
            post_links = conn.execute("SELECT COUNT(*) FROM lead_post_links").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM apify_runs").fetchone()[0]
            total_cost = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM apify_runs"
            ).fetchone()[0]
        return {
            "tracked_realtors": realtors,
            "leads_total": leads,
            "leads_with_profile": leads_with_profile,
            "leads_with_contacts": leads_with_contacts,
            "leads_with_avatar": leads_with_avatar,
            "leads_with_single_face": leads_with_single_face,
            "processed_posts": posts,
            "post_links": post_links,
            "apify_runs": runs,
            "total_cost_usd": round(total_cost, 6),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
