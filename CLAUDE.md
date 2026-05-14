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
| `crawlerbros/instagram-keyword-search-scraper` | Step 1 keyword search (cookie session; `search.cookie_search_keywords`) | Apify usage USD |
| `louisdeconinck/instagram-comments-scraper` | Comments for posts (Step 3 primary) | ~$1/1K comments |
| `apidojo/instagram-comments-scraper-api` | Comments for posts (Step 3 fallback) | $0.0075/post + $0.0005/comment (15 free per post) |

**Comment scraper preference:** `louisdeconinck` is the **primary** because its
snake_case Instagram-raw schema (`user.full_name`, `is_private`, `created_at_utc`,
`media_id`) maps 1:1 to `lead_accounts` columns and to `apify/instagram-profile-scraper`
(Step 4) — no field remapping needed downstream. `media_id` from louisdeconinck
loses precision (JS float64), so the pipeline matches it back to `processed_posts.shortcode`
via `shortcode_to_id()` with a ±1000 tolerance window.

**louisdeconinck input contract (mandatory fields):** the actor silently
returns `0 items` with `status=SUCCEEDED` if its `run_input` is missing
either of these — bisected via `scripts/test_comment_scrapers.py`:

* `proxy: { useApifyProxy: true }` — without it, requests go from raw
  Apify infra IPs and Instagram blocks them within seconds.
* `resultsLimit` + `maxComments` — the actor refuses to commit to a run
  without a per-post comment cap. The pipeline passes
  `pipeline.step3.louisdeconinck_comments_cap_per_post` from
  `config.yaml` (default `10_000` via `DEFAULT_LOUISDECONINCK_COMMENTS_CAP_PER_POST`).
  Actor returns only comments that actually exist on the post, so a
  higher cap does NOT raise our bill -- it just protects against
  truncating the tail on a viral post. Current peak in DB is `~2200`
  (avg `~130`), so the default gives ~5x headroom without any cost
  downside.

Both fields are applied **only on the primary call** in
`_fetch_comments_with_fallback`. The fallback (`apidojo-api`) does not
need them and intentionally keeps an uncapped shape.

`apidojo/instagram-comments-scraper-api` is the **fallback** that fires when the
primary returns 0 items per URL with `status=SUCCEEDED` (the historical failure
mode -- now rare since we honor the input contract above, but kept as a safety
net). Its camelCase output (`message`, `createdAt`, `userId`, `user.fullName`,
...) is remapped to louisdeconinck's shape via
`src.comment_normalizer.normalize_apidojo_api` before saving, so the rest of
Step 3 is actor-agnostic. Apidojo also exposes `postId` directly (= shortcode),
so the synthesized `media_id` is exact and the fuzzy match is a no-op for
fallback items. Other historical alternatives (`apify/instagram-comment-scraper`,
`apidojo/instagram-comments-scraper` w/o `-api` suffix) had pagination / dedup
issues — see `scripts/test_comment_scrapers.py` for the side-by-side comparison
harness used to vet them. The harness includes a `HARDCODE_OVERRIDES` dict at
the top of the file with named bisection recipes (`mimic-pipeline`, per-rail
toggles for `proxy` / `input_limit` / SDK `max_items` / SDK `timeout_secs`,
batch-vs-per-url) so future regressions can be reproduced without editing the
test code -- just swap the dict.

## Pipeline Architecture

Daily pipeline (`scripts/pipeline.py`):

```
Step 1: Discover posts (config: pipeline.step1.discovery_mode)
        Mode "realtors" (default): search.realtor_accounts in config.yaml →
        instagram-post-scraper batch, onlyPostsNewerThan = pipeline.step1.posts_max_age_days,
        resultsLimit = pipeline.step1.post_scraper_results_limit (pipeline code default if omitted in yaml).
        Mode "hashtags": search.hashtags → two runs of
        apify/instagram-hashtag-scraper (resultsType posts + reels),
        resultsLimit = pipeline.step1.hashtag_results_limit (fallback:
        post_scraper_results_limit). Hashtag actor has no onlyPostsNewerThan;
        same max-age window applied client-side on item timestamps (UTC).
        Merge posts+reels datasets by shortCode (prefer row with valid videoUrl).
        Mode "cookie_keywords": search.cookie_search_keywords → single run of
        crawlerbros/instagram-keyword-search-scraper (Instagram cookies from .env;
        see search.cookie_search). Rows are normalized to the same Apify-shaped dicts
        as hashtags, deduped by shortCode, then the same client-side max-age filter
        (pipeline.step1.posts_max_age_days) as hashtags. Reels need a valid CDN videoUrl
        (from media_urls) like the hashtag path.
        Skip posts already in DB, update comments_count for existing.
        Filter: commentsCount >= pipeline.step1.min_comments_per_post.
        Video/Reel items (type Video or productType clips) require a valid
        HTTPS Instagram/Facebook CDN videoUrl — otherwise skipped (not upserted).

Step 2: Score new posts via DeepSeek (caption + transcript combined)
 Only posts with relevance=NULL.
 If the post has a fresh `videoUrl` from Step 1's in-memory pass,
 always download the video and transcribe it via Nexara (no
 caption-based gating). The pipeline then concatenates the two
 strings -- caption first, transcript second, separated by a
 blank line -- and runs RELEVANCE_PROMPT on the combined payload
 in a single DeepSeek call.
 IG video URLs are signed and expire in ~1-2 days, so transcription
 only fires for posts fetched in the *current* run. Older
 `relevance IS NULL` leftovers fall back to caption-only scoring
 on subsequent runs (or "unknown" if the caption is too short).
 Output: relevant / irrelevant / unknown + CTA type

Step 3: Fetch comments (with cost confirmation prompt)
        Posts where: relevant + CTA=comment + (never scanned OR
        comments grew >= pipeline.step3.comments_growth_pct % since
        last scan; default 5%).
        Primary actor:  louisdeconinck/instagram-comments-scraper
            -- run_input MUST include proxy: useApifyProxy AND
               resultsLimit/maxComments=pipeline.step3.louisdeconinck_comments_cap_per_post
               (default 10_000). Either field missing -> actor returns
               0 items with status=SUCCEEDED. See "louisdeconinck input
               contract" above for the bisection.
        Fallback actor: apidojo/instagram-comments-scraper-api
            -- triggers automatically when the primary returns 0 items
               for the entire batch (with status=SUCCEEDED). Apidojo's
               camelCase output is normalized to louisdeconinck's shape
               via src.comment_normalizer.normalize_apidojo_api, so the
               dedup / save loop stays agnostic to the source actor.
            -- the fallback is kept as a safety net even though the
               primary's known failure mode is now blocked by the
               mandatory input fields above.
            -- if BOTH primary and fallback return empty, posts stay
               unscanned in the queue (don't mark last_scanned_at) so
               the next pipeline run retries them.
        Actor IDs are configurable via apify.actors.comments_primary /
        apify.actors.comments_fallback in config.yaml.
        Dedup leads by user_id (not username -- usernames can change)

Step 4: Fetch profiles for new leads (batches of 50)
        Extract contacts from bio (phone, telegram, whatsapp, email)
        Save latest_media_urls for future face recognition
        Download avatar -> data/avatars/<user_id>.jpg
        Run SCRFD face detection -> faces_count
        If faces_count == 1: avatar becomes face_photo_path
        If faces_count != 1: fall back to last N post photos (face leader)
        Actor: instagram-profile-scraper

Step 5: Resolve Telegram contacts via Sherlock (parallel)
        For "naked" leads (profile fetched, bio gave no phone/telegram).
        Stage 1: nick search (cheap, ~30s) -- POST /v1/search/nick.
        Stage 2: photo search (slow, ~135s) if face_photo_path exists --
                 POST /v1/search/photo. Skipped under --skip-sherlock or
                 if SHERLOCK_API_KEY missing.
        Sets sherlock_processed_at on every terminal outcome (found_nick,
        found_photo, no_match, no_face_photo, error) so leads aren't
        silently retried -- clear the column manually to re-process.
        Worker pool defaults to /v1/health pool.idle (override via
        --workers or sherlock.concurrency.workers).

Step 6: Cleanup spent face assets
        Wipes avatar_path + face_photo_path files from disk for leads
        where Sherlock has reached a non-error terminal outcome
        (sherlock_processed_at IS NOT NULL AND sherlock_status != 'error').
        NULLs those columns in DB. error-status leads keep their photos
        so a manual retry doesn't have to re-pay Apify for Step 4.
        Runs even with --skip-sherlock to drain the backlog of leads
        Sherlock'd in prior runs. Suppress with --keep-photos.
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
The chosen one (and the avatar) are then themselves wiped by Step 6
once Sherlock has finished with the lead -- see "Disk hygiene" below.

**Disk hygiene (Step 6):** face assets (`avatar_path` and
`face_photo_path`) live on disk only for the window
`Step 4 -> Step 5 + Step 6`. Once Sherlock has produced a terminal
outcome, the files are unlinked and the path columns NULL'd
(``faces_count`` is preserved as an analytical signal). Leads with
`sherlock_status='error'` are exempt: their photos stay so a manual
retry (clear `sherlock_processed_at`) doesn't require re-paying
Apify for Step 4. The four face-detection helper queries on
``LeadDB`` (``get_leads_needing_avatar``, ``get_leads_needing_face_fallback``,
``get_leads_with_non_single_face``, ``get_all_face_detection_candidates``)
gate on ``sherlock_processed_at IS NULL`` so cleaned leads aren't
re-fetched by ``backfill_avatars.py`` / dev test scripts. Runs every
pipeline invocation (including ``--skip-sherlock``) to drain the
backlog; suppress with ``--keep-photos``. There is no
``photos_cleaned_at`` column -- "cleaned" is implicit:
``sherlock_processed_at IS NOT NULL AND avatar_path IS NULL AND
face_photo_path IS NULL``.

## Database Schema (SQLite)

**`tracked_realtors`** — reserved SQLite table (same columns as before: `username` PK,
`full_name`, `followers_count`, `found_via`, `added_at`, `is_active`). The daily
pipeline does **not** read it for Step 1; monitored usernames live in
`config.yaml` under `search.realtor_accounts` when `discovery_mode` is `realtors`.
The table remains for future features or manual experiments.

**`processed_posts`** — all posts with 10+ comments
- `post_id` PK (shortcode), `post_url`, `owner_username`, `comments_count`
- `relevance` (relevant/irrelevant/unknown), `cta_type` (comment/direct/none)
- `last_comments_count`, `last_scanned_at` — for 5% growth detection

**`lead_accounts`** — collected leads (commenters)
- `username` PK, `user_id` (numeric, permanent), profile data
- `phone`, `email`, `telegram_username`, `whatsapp` — contacts
- `profile_fetched` (0/1), `contact_found` (0/1) — processing state
- `latest_media_urls` — JSON array of photo/video URLs from posts
- `avatar_path` — local path to downloaded avatar (`data/avatars/<user_id>.jpg`); **NULLed by Step 6 after Sherlock for non-error leads**
- `faces_count` — number of faces detected by SCRFD above `min_det_score` (NULL = not processed); preserved across Step 6 cleanup as an analytical signal
- `face_photo_path` — canonical single-face photo sent to the Sherlock bot (avatar if single-face, else post-fallback winner); **NULLed by Step 6 after Sherlock for non-error leads**
- `sherlock_processed_at`, `sherlock_status`, `sherlock_link` — Step 5 outcome. `sherlock_processed_at IS NOT NULL` gates Step 6 cleanup AND excludes the lead from face-detection helper queries (so backfill scripts don't re-fetch via Apify after cleanup).

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
python scripts/test_comment_scrapers.py    # Side-by-side compare comment scrapers (interactive picker)
python scripts/reset_failed_scans.py --apply  # Recover posts marked scanned but with 0 leads

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
  - `pipeline.stepN.*` — per-step tuning knobs (post age, min comments,
    Step 1 `discovery_mode` (`realtors` | `hashtags` | `cookie_keywords`),
    `hashtag_results_limit`, `search.realtor_accounts`, `search.cookie_search_keywords`, growth threshold,
    batch sizes, Sherlock cap). Every value falls back to a `DEFAULT_*`
    constant in `scripts/pipeline.py` if the key is missing, so a fresh /
    partial config still boots.
- `.env` — secrets: `APIFY_API_TOKEN`, `DEEPSEEK_API_KEY`, `NEXARA_API_KEY`;
  optional `INSTAGRAM_SESSION_COOKIE` (or env from `search.cookie_search.session_cookie_env_var`)
  when using Step 1 `discovery_mode=cookie_keywords`
- Step 1 realtor usernames: `search.realtor_accounts` in `config.yaml` when
  `discovery_mode` is `realtors` (not the `tracked_realtors` table)

## Key Source Files

- `src/db.py` — SQLite DB with all tables, dedup logic, lead lifecycle methods
- `src/apify_client_wrapper.py` — Apify wrapper with logging and cost tracking;
  dev defaults in `DEFAULT_APIFY_WRAPPER_LIMITS` when `limit=` is omitted (optional
  `apify.test_limits` in yaml overrides the same keys for legacy configs)
- `src/ig_media_payload.py` — Apify Instagram media helpers shared by Step 1 and
  `scripts/test_hashtag_step1.py`: reel detection, video URL extract/validate,
  hashtag age filter on timestamps, merge posts+reels by shortCode
- `src/instagram_cookie_search.py` — Instagram session cookie normalization for
  Apify and mapping of crawlerbros keyword-search dataset rows to Apify-shaped
  items (shared with `scripts/test_cookie_keyword_search.py` and Step 1
  `discovery_mode=cookie_keywords`)
- `src/pipeline_logger.py` — JSON pipeline logs (every API call → `logs/*.json`)
- `src/contact_extractor.py` — regex extraction of phone/telegram/whatsapp/email from bio
- `src/comment_normalizer.py` — `normalize_apidojo_api()` remaps the
  apidojo-api fallback's camelCase output (`message`, `createdAt`,
  `userId`, `user.fullName`, ...) to louisdeconinck's snake_case shape
  so Step 3's dedup / save loop is actor-agnostic. Also synthesizes
  `media_id` from `postId` so the existing shortcode fuzzy lookup
  matches without code changes.
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
- **Disk hygiene:** face photos (avatars + post-photo leaders) live on disk only between Step 4 and Step 6. After Sherlock finishes a lead with a non-error terminal status, Step 6 unlinks its files and NULLs the path columns. Face-detection helper queries gate on `sherlock_processed_at IS NULL` so cleaned leads aren't re-fetched via Apify.

## Language

The project spec and communication are in Russian. Code, comments, variable names, and logs should be in English.
