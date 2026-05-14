"""Unit tests for LeadDB — runs against in-memory SQLite, no Apify needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.db import LeadDB


@pytest.fixture
def db():
    return LeadDB(":memory:")


def test_init_tables(db):
    stats = db.get_stats()
    assert stats["processed_posts"] == 0
    assert stats["leads_total"] == 0
    assert stats["apify_runs"] == 0


def test_post_dedup(db):
    assert not db.is_post_processed("abc123")
    db.upsert_post("abc123", post_url="https://instagram.com/p/abc123/")
    assert db.is_post_processed("abc123")

    # Inserting again should not raise
    db.upsert_post("abc123", post_url="https://instagram.com/p/abc123/")
    assert db.get_stats()["processed_posts"] == 1


def test_lead_account_dedup(db):
    assert not db.is_account_known("user1")

    added = db.add_lead_account("user1", full_name="Test User")
    assert added is True
    assert db.is_account_known("user1")

    added_again = db.add_lead_account("user1", full_name="Test User")
    assert added_again is False
    assert db.get_stats()["leads_total"] == 1


def test_multiple_leads(db):
    db.add_lead_account("user1")
    db.add_lead_account("user2")
    db.add_lead_account("user3")
    assert db.get_stats()["leads_total"] == 3


def test_apify_run_logging(db):
    db.log_apify_run(
        run_id="run1",
        actor_id="apify/instagram-scraper",
        status="SUCCEEDED",
        items_count=10,
        cost_usd=0.05,
    )
    stats = db.get_stats()
    assert stats["apify_runs"] == 1
    assert stats["total_cost_usd"] == 0.05


def test_apify_run_upsert(db):
    db.log_apify_run(run_id="run1", actor_id="test", cost_usd=0.01)
    db.log_apify_run(run_id="run1", actor_id="test", cost_usd=0.02)
    assert db.get_stats()["apify_runs"] == 1
    assert db.get_stats()["total_cost_usd"] == 0.02


def test_clear_lead_avatar_path_keeps_faces_count(db):
    db.add_lead_account("u_avatar", user_id="1")
    db.update_lead_profile("u_avatar", full_name="A")
    db.update_lead_avatar("u_avatar", "data/avatars/x.jpg", 2)
    db.clear_lead_avatar_path("u_avatar")
    with db._conn() as conn:
        row = conn.execute(
            "SELECT avatar_path, faces_count FROM lead_accounts WHERE username = ?",
            ("u_avatar",),
        ).fetchone()
    assert row["avatar_path"] is None
    assert row["faces_count"] == 2


def test_get_leads_needing_avatar_excludes_detection_without_file(db):
    """Orphan Step 4 cleanup: faces_count set, avatar_path NULL — no Apify re-queue."""
    db.add_lead_account("orphan", user_id="99")
    db.update_lead_profile("orphan", full_name="O")
    db.update_lead_avatar("orphan", "/gone.jpg", 3)
    db.clear_lead_avatar_path("orphan")
    assert db.get_leads_needing_avatar(limit=10) == []


def test_get_leads_needing_avatar_includes_never_detected(db):
    db.add_lead_account("fresh", user_id="100")
    db.update_lead_profile("fresh", full_name="F")
    rows = db.get_leads_needing_avatar(limit=10)
    assert len(rows) == 1
    assert rows[0]["username"] == "fresh"
