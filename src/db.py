"""SQLite storage for leads, realtors, posts, and run tracking."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def lead_disk_photo_usable(
    path: str | None, *, base_dir: Path | None = None
) -> bool:
    """True if ``path`` points at a non-empty local file readable on disk."""
    if not path:
        return False
    try:
        fp = Path(path)
        if base_dir is not None and not fp.is_absolute():
            fp = base_dir / fp
        return fp.is_file() and fp.stat().st_size > 0
    except OSError:
        return False


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

                    -- canonical single-face photo to send to Sherlock bot
                    -- (the avatar itself if faces_count == 1, else a photo
                    -- picked from the last N posts via cluster-leader search).
                    face_photo_path     TEXT,

                    -- contacts (filled by bio parsing or Module 2)
                    phone               TEXT,
                    email               TEXT,
                    telegram_username   TEXT,
                    whatsapp            TEXT,

                    -- Sherlock (Module 2): Step 5 resolves Telegram
                    -- contacts for leads whose bio gave us nothing.
                    -- Found phone / telegram_username are written into
                    -- the existing contact columns above; sherlock_link
                    -- is the URL of the matched profile (vk.com/...,
                    -- t.me/...). sherlock_processed_at gates retries
                    -- (NULL = never tried; selector key for Step 5).
                    -- sherlock_status records the outcome label so we
                    -- can debug coverage without re-running.
                    sherlock_link           TEXT,
                    sherlock_processed_at   TEXT,
                    sherlock_status         TEXT,

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
                    comment_pk      TEXT,
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
                    location            TEXT,
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
                ("face_photo_path", "TEXT"),
                # Module 2 / Step 5 (Sherlock contact resolution).
                # See the CREATE TABLE block above for the meaning;
                # listed here so existing DBs pick them up via ALTER.
                ("sherlock_link", "TEXT"),
                ("sherlock_processed_at", "TEXT"),
                ("sherlock_status", "TEXT"),
            ],
            "lead_post_links": [
                ("comment_pk", "TEXT"),
            ],
            "processed_posts": [
                ("location", "TEXT"),
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

    def clear_lead_avatar_path(self, username: str) -> None:
        """Set ``avatar_path`` to NULL; keep ``faces_count`` (detection already ran)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE lead_accounts SET avatar_path = NULL WHERE username = ?",
                (username,),
            )

    def update_lead_face(self, username: str, face_photo_path: str) -> None:
        """Persist the canonical single-face photo for a lead.

        This is the photo we'll forward to the external Sherlock bot — the
        avatar itself if it has exactly one face, otherwise a photo chosen
        from the last N posts via the face-leader search.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE lead_accounts SET face_photo_path = ? WHERE username = ?",
                (face_photo_path, username),
            )

    # --- Sherlock (Step 5) -----------------------------------------

    def get_leads_for_sherlock(
        self, limit: int = 10000, *, photo_base_dir: Path | None = None
    ) -> list[dict]:
        """Leads that need Sherlock-based contact resolution.

        Selection rules (per Step 5 spec):
          * profile_fetched=1 -- we have at least the bare profile data
            (Step 4 ran for them).
          * phone IS NULL AND telegram_username IS NULL -- the bio
            extractor in Step 4 found no contact. If we already have
            *any* contact from bio, Sherlock is skipped to save bot
            quota and avoid overwriting cheaper / equally-valid data.
          * sherlock_processed_at IS NULL -- we never tried Sherlock
            on this lead before. Step 5 marks every terminal outcome
            (found / no_match / no_face_photo / error) so leads aren't
            silently retried; clear this column manually to re-process.
          * is_private != 1 -- private accounts have no useful avatar
            and no public bio, Sherlock can't help.
          * Canonical ``face_photo_path`` recorded in DB and the file exists
            on disk (non-empty). Step 5 always needs this for photo
            fallback after nick misses; omitting naked leads without a usable
            file avoids enqueueing batches that terminate as ``no_face_photo``.
            Stale paths in DB are skipped until :py:meth:`null_missing_photo_paths`
            clears them. Relative paths use the process CWD unless
            ``photo_base_dir`` is set.

        Also joins the latest ``lead_post_links`` row per username for
        Telegram context (post URL, optional comment permalink).

        Rows are fetched in SQL batches until ``limit`` matching leads are
        collected or the candidate table is exhausted.
        """
        sql = (
            "SELECT "
            "la.username AS username, "
            "la.user_id AS user_id, "
            "la.full_name AS full_name, "
            "la.face_photo_path AS face_photo_path, "
            "lpl.post_url AS context_post_url, "
            "lpl.post_shortcode AS context_post_shortcode, "
            "lpl.comment_pk AS context_comment_pk "
            "FROM lead_accounts la "
            "LEFT JOIN lead_post_links lpl ON lpl.id = ("
            "  SELECT MAX(id) FROM lead_post_links l2 "
            "  WHERE l2.username = la.username"
            ") "
            "WHERE profile_fetched = 1 "
            "  AND phone IS NULL "
            "  AND telegram_username IS NULL "
            "  AND sherlock_processed_at IS NULL "
            "  AND COALESCE(is_private, 0) = 0 "
            "  AND la.face_photo_path IS NOT NULL "
            "LIMIT ? OFFSET ?"
        )
        batch = max(256, min(limit * 4, 4000))
        out: list[dict] = []
        offset = 0
        with self._conn() as conn:
            while len(out) < limit:
                rows = conn.execute(
                    sql, (batch, offset)
                ).fetchall()
                if not rows:
                    break
                offset += len(rows)
                for r in rows:
                    d = dict(r)
                    fph = d.get("face_photo_path")
                    if not lead_disk_photo_usable(
                        None if fph is None else str(fph),
                        base_dir=photo_base_dir,
                    ):
                        continue
                    out.append(d)
                    if len(out) >= limit:
                        break
        return out

    def mark_lead_sherlock(
        self,
        username: str,
        *,
        status: str,
        telegram_username: str | None = None,
        phone: str | None = None,
        sherlock_link: str | None = None,
    ) -> None:
        """Record the outcome of a Sherlock pass for one lead.

        Always sets ``sherlock_processed_at = now`` and
        ``sherlock_status = status`` so the lead is excluded from the
        next ``get_leads_for_sherlock()`` window. When Sherlock did
        find data, the contact fields (``telegram_username`` /
        ``phone``) are filled in *only if currently NULL* -- this
        preserves the invariant from
        :py:meth:`get_leads_for_sherlock` (Sherlock never overwrites
        bio data) and keeps the function safe to call even if some
        other code path raced and filled the column.

        ``status`` is the free-form label produced by Step 5
        (e.g. ``"found_nick"`` / ``"found_photo"`` / ``"no_match"`` /
        ``"no_face_photo"`` / ``"error"``). It's not enum-validated
        here because Step 5 owns the vocabulary.
        """
        sets: list[str] = [
            "sherlock_processed_at = ?",
            "sherlock_status = ?",
        ]
        vals: list = [_now(), status]

        if telegram_username:
            sets.append(
                "telegram_username = COALESCE(telegram_username, ?)"
            )
            vals.append(telegram_username)
        if phone:
            sets.append("phone = COALESCE(phone, ?)")
            vals.append(phone)
        if sherlock_link:
            sets.append("sherlock_link = ?")
            vals.append(sherlock_link)

        # contact_found is the global "did we get any contact?" flag
        # used by ``get_stats`` and downstream consumers. Flip it to 1
        # whenever Sherlock contributed something usable so the stats
        # banner reflects reality without a separate counter.
        if telegram_username or phone:
            sets.append("contact_found = 1")
            sets.append("contact_found_at = COALESCE(contact_found_at, ?)")
            vals.append(_now())

        vals.append(username)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE lead_accounts SET {', '.join(sets)} "
                f"WHERE username = ?",
                vals,
            )

    def get_leads_with_spent_photos(self, limit: int = 10000) -> list[dict]:
        """Leads whose face assets are no longer needed and can be deleted.

        Step 6 of the pipeline calls this to find leads where:
          * Sherlock has finished (``sherlock_processed_at IS NOT NULL``).
          * Outcome was NOT ``error`` -- those keep their photos so a
            retry (after manually clearing ``sherlock_processed_at``)
            doesn't have to re-pay Apify for Step 4.
          * At least one of ``avatar_path`` / ``face_photo_path`` still
            points at a file. The presence-OR clause is also what makes
            this query self-terminating: once Step 6 NULLs both columns,
            the lead is excluded next run.

        Returns the columns Step 6 needs to delete files and clear DB
        state in a single pass:
          ``username``, ``user_id``, ``avatar_path``, ``face_photo_path``.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, avatar_path, face_photo_path "
                "FROM lead_accounts "
                "WHERE sherlock_processed_at IS NOT NULL "
                "  AND COALESCE(sherlock_status, '') != 'error' "
                "  AND (avatar_path IS NOT NULL OR face_photo_path IS NOT NULL) "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_lead_photos_cleaned(self, username: str) -> None:
        """NULL out ``avatar_path`` and ``face_photo_path`` for a lead.

        Called by Step 6 after the on-disk files were unlinked. Keeps
        ``faces_count`` intact -- it's an analytical signal (how many
        faces we saw on this lead's avatar), not a path. Keeps every
        Sherlock column intact -- the lead's terminal status / link /
        timestamp are part of its permanent contact history.

        Note: there's no dedicated ``photos_cleaned_at`` column.
        Cleaned-vs-not is implicit: ``sherlock_processed_at IS NOT NULL
        AND avatar_path IS NULL AND face_photo_path IS NULL`` means the
        lead was cleaned; the four face-detection queries
        (:py:meth:`get_leads_needing_avatar` &c) gate on
        ``sherlock_processed_at IS NULL`` so cleaned leads aren't
        re-fetched by ``backfill_avatars.py`` / dev test scripts.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE lead_accounts "
                "SET avatar_path = NULL, face_photo_path = NULL "
                "WHERE username = ?",
                (username,),
            )

    def null_missing_photo_paths(self) -> dict[str, int]:
        """NULL ``avatar_path`` / ``face_photo_path`` when the file is absent.

        Relative paths are resolved from the process current working directory;
        run from the repository root (where ``data/`` lives). Empty (0-byte)
        files are treated as missing. Does not change ``faces_count`` or
        Sherlock columns.
        """
        leads_changed = 0
        avatar_nulled = 0
        face_nulled = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, avatar_path, face_photo_path "
                "FROM lead_accounts "
                "WHERE avatar_path IS NOT NULL OR face_photo_path IS NOT NULL"
            ).fetchall()
            for row in rows:
                ap = row["avatar_path"]
                fph = row["face_photo_path"]
                drop_ap = ap is not None and not lead_disk_photo_usable(ap)
                drop_f = fph is not None and not lead_disk_photo_usable(fph)
                if not drop_ap and not drop_f:
                    continue
                sets: list[str] = []
                if drop_ap:
                    sets.append("avatar_path = NULL")
                    avatar_nulled += 1
                if drop_f:
                    sets.append("face_photo_path = NULL")
                    face_nulled += 1
                conn.execute(
                    f"UPDATE lead_accounts SET {', '.join(sets)} "
                    "WHERE username = ?",
                    (row["username"],),
                )
                leads_changed += 1
        return {
            "leads_changed": leads_changed,
            "avatar_path_nulled": avatar_nulled,
            "face_photo_path_nulled": face_nulled,
        }

    def get_leads_needing_avatar(self, limit: int = 1000) -> list[dict]:
        """Leads that have profile data but never had a successful avatar pass.

        Requires ``faces_count IS NULL`` so we do not re-queue leads whose
        on-disk avatar was intentionally removed after Step 4 (orphan file
        cleanup) while SCRFD results remain in ``faces_count``.

        Excludes leads Sherlock has already processed (terminal status set).
        Step 6 of the pipeline NULLs out ``avatar_path`` after Sherlock
        finishes (except for ``error`` outcomes), so without the
        ``sherlock_processed_at`` gate ``backfill_avatars.py`` would re-fetch
        every cleaned lead via Apify -- direct money loss. The retry path
        "clear sherlock_processed_at to re-process" is documented on
        :py:meth:`get_leads_for_sherlock` and naturally re-admits the lead
        here once it's cleared.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, profile_pic_url_hd, profile_pic_url "
                "FROM lead_accounts "
                "WHERE profile_fetched = 1 "
                "  AND avatar_path IS NULL "
                "  AND faces_count IS NULL "
                "  AND sherlock_processed_at IS NULL "
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

    def get_leads_needing_face_fallback(self, limit: int = 1000) -> list[dict]:
        """Leads whose avatar face count is not 1 and canonical face is unresolved.

        Candidates for the last-N-posts fallback. Skips private profiles and
        rows without stored ``latest_media_urls`` (we'd have nothing to probe
        anyway — a fresh Apify refetch is needed, which the dev script does).

        Excludes Sherlock-processed leads -- same reasoning as
        :py:meth:`get_leads_needing_avatar`. Step 6's cleanup may have
        wiped ``face_photo_path``; re-running fallback for them
        without re-running Step 5 is a money-burn loop.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, profile_pic_url_hd, profile_pic_url, "
                "       faces_count, latest_media_urls "
                "FROM lead_accounts "
                "WHERE profile_fetched = 1 "
                "  AND faces_count IS NOT NULL AND faces_count != 1 "
                "  AND face_photo_path IS NULL "
                "  AND sherlock_processed_at IS NULL "
                "  AND COALESCE(is_private, 0) = 0 "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_leads_with_non_single_face(self, limit: int = 10000) -> list[dict]:
        """Leads that were face-detected but not exactly one face (0 or >1).

        Useful for re-running detection with tweaked parameters or a new
        model without touching leads that already look clean.

        Skips Sherlock-processed leads: their face assets may have been
        wiped by Step 6, and re-detection on them would either find
        nothing (cleaned) or duplicate work that no longer feeds Step 5.
        Clear ``sherlock_processed_at`` manually to re-admit a lead.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, profile_pic_url_hd, "
                "       profile_pic_url, faces_count "
                "FROM lead_accounts "
                "WHERE faces_count IS NOT NULL AND faces_count != 1 "
                "  AND sherlock_processed_at IS NULL "
                "  AND COALESCE(is_private, 0) = 0 "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_face_detection_candidates(self, limit: int = 10000) -> list[dict]:
        """Every lead eligible for face detection, regardless of its
        current ``faces_count`` / ``avatar_path`` state.

        Unlike :py:meth:`get_leads_needing_avatar` (which skips already
        processed leads) and :py:meth:`get_leads_with_non_single_face`
        (which skips clean single-face leads), this one returns the
        full superset. Useful for wholesale re-detection after swapping
        models — e.g. the MediaPipe → SCRFD migration — where every
        stored ``faces_count`` should be overwritten with the new
        detector's verdict.

        Excludes private accounts (their avatars aren't accessible)
        AND Sherlock-processed leads. The latter is for the same
        reason as :py:meth:`get_leads_needing_avatar`: Step 6 may
        have wiped their avatars, so wholesale re-detection would
        re-trigger Apify downloads on leads we deliberately cleaned.
        Clear ``sherlock_processed_at`` first if you really want to
        include them.

        ``faces_count`` and ``avatar_path`` are returned so callers can
        log before/after diffs.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username, user_id, profile_pic_url_hd, "
                "       profile_pic_url, faces_count, avatar_path "
                "FROM lead_accounts "
                "WHERE profile_fetched = 1 "
                "  AND sherlock_processed_at IS NULL "
                "  AND COALESCE(is_private, 0) = 0 "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_leads_without_profile(self, limit: int = 100) -> list[dict]:
        """Leads that have never had their profile scraped.

        Once a profile is fetched we never re-poll it — contact info
        (phone / telegram / email in bio) doesn't meaningfully change
        for most people, and re-scraping costs Apify money per lead.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT username FROM lead_accounts "
                "WHERE is_private = 0 AND profile_fetched = 0 "
                "LIMIT ?",
                (limit,),
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
            leads_with_face_photo = conn.execute(
                "SELECT COUNT(*) FROM lead_accounts WHERE face_photo_path IS NOT NULL"
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
            "leads_with_face_photo": leads_with_face_photo,
            "processed_posts": posts,
            "post_links": post_links,
            "apify_runs": runs,
            "total_cost_usd": round(total_cost, 6),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
