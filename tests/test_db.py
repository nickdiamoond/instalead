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
    db.mark_post_processed("abc123", post_url="https://instagram.com/p/abc123/")
    assert db.is_post_processed("abc123")

    # Inserting again should not raise
    db.mark_post_processed("abc123", post_url="https://instagram.com/p/abc123/")
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
