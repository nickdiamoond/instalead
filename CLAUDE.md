# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Instagram lead checker for real estate buyers (SPB focus). The system collects Instagram accounts of people interested in buying property (based on their comments on realtor reels/posts), then finds their contact information via Telegram.

**Two main modules:**
- **Module 1 (Instagram Collector):** Finds potential lead Instagram accounts by monitoring realtor accounts, collecting their posts/reels, scoring relevance via AI, and extracting commenters as leads.
- **Module 2 (Contact Finder):** *(future)* Resolves Instagram accounts to phone numbers/Telegram contacts using Telegram SearchGlobalRequest and a face-recognition bot ("Sherlock bot").

## Tech Stack

- Python 3.11+
- Apify API — Instagram data via multiple actors (see below)
- DeepSeek API (OpenAI-compatible) — relevance scoring of post captions
- Nexara API (`/audio/transcriptions`) — Whisper-style transcription of
 Reels audio when the caption is missing or DeepSeek-on-caption returns
 `unknown`
- structlog — logging
- SQLite — deduplication, state, lead storage (`data/leads.db`)
- Pipeline JSON logs — every API call logged to `logs/` for cost analysis
- InsightFace + onnxruntime — SCRFD face detection + ArcFace 512-d
 embeddings for same-person search (single detector across avatars and
 post photos)

Future (not yet implemented):
- Telethon — Telegram client (SearchGlobalRequest)
- Aiogram — Telegram bot for notifications
- replicate.com — avatar upscaling

## Apify Actors Used

| Actor | Purpose | Price |
|---|---|---|
| `apify/instagram-profile-scraper` | Profile info, relatedProfiles, latestPosts | ~$0.0023/profile |
| `apify/instagram-post-scraper` | Posts/reels from accounts (batch, date filter) | ~$0.0017/post |
| `apify/instagram-hashtag-scraper` | Posts/reels by hashtag | ~$0.0023/post |
| `louisdeconinck/instagram-comments-scraper` | Comments for posts (best pagination) | ~$0.50/1K comments |

**Important:** `louisdeconinck` is the preferred comment scraper — other actors (`apidojo/*`, `apify/*`) have severe pagination/dedup issues. `media_id` from comments loses precision (JS float64) — use shortcode fuzzy matching (±1000 tolerance).

## Pipeline Architecture

Daily pipeline (`scripts/pipeline.py`):

```
Step 1: Fetch posts from tracked_realtors (batch, last 7 days)
        Actor: instagram-post-scraper
        Skip posts already in DB, update comments_count for existing
        Filter: commentsCount >= 10

Step 2: Score new posts via DeepSeek (with Nexara video fallback)
 Only posts with relevance=NULL
 First pass: run RELEVANCE_PROMPT on caption.
 Fallback: if caption is empty OR DeepSeek returned `is_real_estate=null`,
 AND the post has a fresh `videoUrl` from Step 1's in-memory pass,
 download the video, transcribe via Nexara, then re-run
 RELEVANCE_PROMPT on the transcript alone (caption is not combined).
 IG video URLs are signed and expire in ~1-2 days, so transcription
 only fires for posts fetched in the *current* run — `relevance IS NULL`
 leftovers from older runs stay "unknown" until a fresh fetch.
 Output: relevant / irrelevant / unknown + CTA type

Step 3: Fetch comments (with cost confirmation prompt)
        Posts where: relevant + CTA=comment + (never scanned OR comments grew 5%+)
        Actor: louisdeconinck/instagram-comments-scraper
        Dedup leads by user_id (not username — usernames can change)

Step 4: Fetch profiles for new leads (batches of 50)
        Extract contacts from bio (phone, telegram, whatsapp, email)
        Save latest_media_urls for future face recognition
        Download avatar -> data/avatars/<user_id>.jpg
        Run SCRFD face detection -> faces_count
        If faces_count == 1: avatar becomes face_photo_path
        If faces_count != 1: fall back to last N post photos (face leader)
        Actor: instagram-profile-scraper
```

**Avatar face detection note:** Instagram CDN URLs are signed and expire
in ~1-2 days, so avatars are downloaded immediately during Step 4.
SCRFD (from InsightFace's ``buffalo_s`` bundle) counts faces locally on
CPU. The ``min_det_score`` threshold is tuned to 0.7 by default to
reject background / false-positive faces common on Instagram
full-body / studio shots; override via ``face_detection.min_det_score``
in ``config.yaml``.

**Face leader fallback (Step 4 extension):** when the avatar has 0 or
\>1 faces, the pipeline probes the last N posts from the same Apify
response (no extra cost). For each post the carousel cover (or
`displayUrl` of photo posts; videos skipped) is downloaded to
`data/lead_photos/<user_id>/`. The same SCRFD + ArcFace pass both counts
faces and produces the 512-d embedding — photos with exactly one face
(above ``min_det_score``) are greedy-clustered by cosine similarity. If
the largest cluster covers at least M photos, the best-scoring member
is promoted to `lead_accounts.face_photo_path` — the single canonical
photo we later forward to the external Sherlock Telegram bot (which
does the actual cross-profile matching itself). Otherwise the lead is
skipped. Embeddings are used internally for clustering and discarded
afterwards. All knobs live under `face_fallback:` in `config.yaml`.
Downloaded post photos are removed except the chosen one (configurable).

## Database Schema (SQLite)

**`tracked_realtors`** — monitored realtor accounts (source of posts)
- `username` PK, `full_name`, `followers_count`, `found_via`, `is_active`

**`processed_posts`** — all posts with 10+ comments
- `post_id` PK (shortcode), `post_url`, `owner_username`, `comments_count`
- `relevance` (relevant/irrelevant/unknown), `cta_type` (comment/direct/none)
- `last_comments_count`, `last_scanned_at` — for 5% growth detection

**`lead_accounts`** — collected leads (commenters)
- `username` PK, `user_id` (numeric, permanent), profile data
- `phone`, `email`, `telegram_username`, `whatsapp` — contacts
- `profile_fetched` (0/1), `contact_found` (0/1) — processing state
- `latest_media_urls` — JSON array of photo/video URLs from posts
- `avatar_path` — local path to downloaded avatar (`data/avatars/<user_id>.jpg`)
- `faces_count` — number of faces detected by SCRFD above `min_det_score` (NULL = not processed)
- `face_photo_path` — canonical single-face photo sent to the Sherlock bot (avatar if single-face, else post-fallback winner)

**`lead_post_links`** — which lead commented on which post
- `username`, `user_id`, `post_url`, `post_shortcode`, `comment_text`

**`apify_runs`** — cost tracking for every API call

## Development Commands

```bash
# Virtual environment
python -m venv .venv
.venv/Scripts/activate     # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (Linux/Mac — straight from PyPI)
pip install -r requirements.txt

# Install dependencies (Windows + Python 3.12) — PyPI ships insightface
# only as sdist on Windows, which needs MSVC Build Tools. Easier path:
# install the prebuilt community wheel first, then the rest.
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run the daily pipeline
python scripts/pipeline.py

# Individual test scripts (for exploration/debugging)
python scripts/test_related_realtors.py    # Find realtor accounts via relatedProfiles
python scripts/test_batch_posts.py         # Fetch posts from realtors
python scripts/test_score_posts.py         # Score posts via DeepSeek
python scripts/test_fetch_comments.py      # Fetch comments for relevant posts
python scripts/test_fetch_profiles.py      # Fetch lead profiles + extract contacts
python scripts/test_cost_analysis.py       # Analyze costs from pipeline logs

# Backfill avatars + face detection for existing leads
python scripts/backfill_avatars.py              # refetch profiles (Apify $$)
python scripts/backfill_avatars.py --no-refetch # try stale URLs only (most 403)
python scripts/backfill_avatars.py --limit 100  # cap leads processed

# Face matching smoke test (dev, uses facetest/ folder)
python scripts/test_face_matcher.py
python scripts/test_face_matcher.py --threshold 0.45

# Face leader fallback test (last-N posts for leads with faces_count != 1)
python scripts/test_face_leader.py
python scripts/test_face_leader.py --limit 20
python scripts/test_face_leader.py --keep-photos
```

## Configuration

- `config.yaml` — search parameters, Apify actor IDs, limits, filters
- `.env` — secrets: `APIFY_API_TOKEN`, `DEEPSEEK_API_KEY`, `NEXARA_API_KEY`
- Realtor accounts stored in DB table `tracked_realtors` (not config)

## Key Source Files

- `src/db.py` — SQLite DB with all tables, dedup logic, lead lifecycle methods
- `src/apify_client_wrapper.py` — Apify wrapper with logging and cost tracking
- `src/pipeline_logger.py` — JSON pipeline logs (every API call → `logs/*.json`)
- `src/contact_extractor.py` — regex extraction of phone/telegram/whatsapp/email from bio
- `src/avatar_downloader.py` — download avatar URL → `data/avatars/<user_id>.jpg`
- `src/transcriber.py` — `NexaraTranscriber`: downloads IG videoUrl to a
 temp file and POSTs it to Nexara `/audio/transcriptions`; degrades
 gracefully when `NEXARA_API_KEY` is missing (returns `None`, pipeline
 falls back to legacy `relevance="unknown"`)
- `src/face_embedder.py` — InsightFace SCRFD + ArcFace wrapper: exposes
  both `count_faces()` (avatars) and `embed_faces()` (post-photo
  clustering), with a shared `min_det_score` threshold
- `src/face_matcher.py` — pure-Python greedy clustering by cosine similarity
- `src/face_leader.py` — last-N-photos leader resolution (SCRFD single-pass
  filter + ArcFace + cluster)
- `src/logger.py` — structlog configuration
- `src/config.py` — config.yaml + .env loader
- `docs/apify_api_schemas.md` — detailed API schemas for all actors
- `models/` — vendored ML weights (InsightFace `buffalo_s` only);
  committed to the repo so Ubuntu deploys don't re-download ~155 MB on
  first use. See `models/README.md` for layout and Git LFS tips.
- `facetest/` — dev-only sandbox for `scripts/test_face_matcher.py`

## Architecture Principles

- **Cost awareness:** Apify requests cost money. Always deduplicate — check DB before making API calls. Track and log costs per cycle.
- **Dedup by user_id:** Instagram usernames can change. Always check `user_id` (numeric pk) for deduplication, not just username.
- **Budget controls:** Pipeline shows estimated cost before expensive operations and asks for confirmation.
- **Incremental:** Each pipeline run only processes new/changed data. Safe to run repeatedly.
- **5% comment growth threshold:** Don't re-scan comments on a post unless comment count grew by at least 5% since last scan.

## Language

The project spec and communication are in Russian. Code, comments, variable names, and logs should be in English.
