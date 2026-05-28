# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Instagram lead checker for real estate buyers (SPB focus). The system collects Instagram accounts of people interested in buying property (based on their comments on realtor reels/posts), then finds their contact information via Telegram.

**Two main modules:**
- **Module 1 (Instagram Collector):** Finds potential lead Instagram accounts by monitoring realtor accounts (or hashtags / cookie keyword search), collecting their posts/reels, scoring relevance via AI, and extracting commenters as leads.
- **Module 2 (Contact Finder):** Resolves Instagram accounts to Telegram contacts via the external **Sherlock API** (Step 5: nick search, then photo search when `face_photo_path` exists). Bio contacts from Step 4 are tried first. **Telethon** (`SearchGlobalRequest`) is still future — not used in the daily pipeline.

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
- Lingua — Step 2 language gate (non-Russian captions → `irrelevant` without DeepSeek)
- Sherlock API — Step 5 contact resolution (`src/sherlock_client.py`)
- Aiogram — Telegram bot notifications and Step 2/3 inline confirmations
  (`src/telegram_notifier.py`, `src/telegram_inline_confirm.py`)

Future (not yet implemented):
- Telethon — Telegram client (SearchGlobalRequest)
- replicate.com — avatar upscaling

## Apify Actors Used

| Actor | Purpose | Price |
|---|---|---|
| `apify/instagram-hashtag-scraper` | Step 1 posts/reels by hashtag (`discovery_mode=hashtags`) | ~$0.0023/post |
| `apify/instagram-post-scraper` | Step 1 posts/reels from realtor accounts (`discovery_mode=realtors`) | ~$0.0017/post (basicData) |
| `crawlerbros/instagram-keyword-search-scraper` | Step 1 keyword search (`discovery_mode=cookie_keywords`; `search.cookie_search_keywords`) | Apify usage USD |
| `louisdeconinck/instagram-comments-scraper` | Comments for posts (Step 3 primary) | ~$1/1K comments |
| `apidojo/instagram-comments-scraper-api` | Comments for posts (Step 3 fallback) | $0.0075/post + $0.0005/comment (15 free per post) |
| `apify/instagram-profile-scraper` | Profile info, relatedProfiles, latestPosts (Step 4) | ~$0.0026/profile |

Actor IDs are overridable via `config.yaml` → `apify.actors.*`. Legacy `apify/instagram-comment-scraper` is for dev/`ApifyWrapper` only — not the daily pipeline. See `docs/apify_api_schemas.md` for request/response shapes.

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
`scripts/pipeline_lib/apify_runner.py` → `_fetch_comments_with_fallback`.

**apidojo-api fallback input** (same module): `startUrls` (post URLs), `proxy:
{ useApifyProxy: true }`, and `maxItems = pipeline.step3.apidojo_comments_cap_per_post
× len(urls)` (run-wide cap, default multiplier from
`DEFAULT_APIFY_COMMENTS_CAP_PER_POST` in `scripts/pipeline_lib/defaults.py`). It does
**not** use `resultsLimit` / `maxComments`.

`apidojo/instagram-comments-scraper-api` is the **fallback** that fires when the
primary returns **0 items for the entire batch** with `status=SUCCEEDED` (the
historical failure mode -- now rare since we honor the primary input contract above,
but kept as a safety net). Its camelCase output (`message`, `createdAt`, `userId`,
`user.fullName`, ...) is remapped to louisdeconinck's shape via
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

Daily pipeline entrypoint: `scripts/pipeline.py` (orchestration). Step logic is
split into `scripts/pipeline_lib/` (`defaults.py`, `apify_runner.py`, `scoring.py`,
`step4_faces.py`, `step5_sherlock.py`, `step6_cleanup.py`, …).

```
Step 1: Discover posts (config: pipeline.step1.discovery_mode)
        Mode "realtors" (default): search.realtor_accounts →
        apify/instagram-post-scraper: username[], resultsLimit,
        onlyPostsNewerThan = "{posts_max_age_days} days", dataDetailLevel basicData,
        proxy useApifyProxy. Plus client-side max-age filter on timestamp (UTC)
        via src.ig_media_payload.filter_items_within_max_age (same helper as other modes).
        Mode "hashtags": search.hashtags → two runs of apify/instagram-hashtag-scraper
        (resultsType posts + reels), resultsLimit = hashtag_results_limit (fallback:
        post_scraper_results_limit), proxy useApifyProxy. Hashtag actor has no
        onlyPostsNewerThan — age filter client-side. Merge posts+reels by shortCode
        (prefer row with valid videoUrl).
        Mode "cookie_keywords": search.cookie_search_keywords →
        crawlerbros/instagram-keyword-search-scraper (cookies from env;
        pipeline.step1.cookie_search: size_per_keyword, session_cookie_env_var,
        session_name). Normalize via src.instagram_cookie_search, dedupe by shortCode,
        client-side age filter. Reels need valid CDN videoUrl from media_urls.
        All modes: skip/update existing posts in DB; commentsCount >= min_comments;
        reels require valid HTTPS Instagram/Facebook CDN videoUrl or not upserted.

Step 2: Lingua language gate + DeepSeek scoring (caption + transcript)
        Only posts with relevance=NULL.
        Non-Russian text (Lingua) → irrelevant without a DeepSeek call.
        If the post has a fresh videoUrl from Step 1's in-memory map, transcribe
        via Nexara; concatenate caption + transcript; single DeepSeek RELEVANCE_PROMPT.
        IG video URLs expire in ~1-2 days — transcription only for posts from the
        current run; older NULL leftovers are caption-only (or unknown if empty).
        Posts scored is_real_estate=true may get an optional Telegram inline human
        confirm (aiogram) when the bot is configured.
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
            -- when primary returns 0 items for the entire batch (SUCCEEDED).
               Input: startUrls, proxy, maxItems = apidojo_comments_cap_per_post × N URLs.
               Normalized via src.comment_normalizer.normalize_apidojo_api.
            -- if BOTH return empty, posts stay unscanned (no last_scanned_at update).
        Implemented in scripts/pipeline_lib/apify_runner.py.
        Actor IDs: apify.actors.comments_primary / comments_fallback.
        Cost confirm when pipeline.prompt_terminal_confirmation is true.
        Dedup leads by user_id (not username -- usernames can change)

Step 4: Fetch profiles for new leads (batches of profile_batch_size, default 50;
        up to pipeline.step4.batch_limit leads per run)
        Actor: apify/instagram-profile-scraper — input { usernames: [...] } only.
        Extract contacts from bio (phone, telegram, whatsapp, email)
        Save latest_media_urls for future face recognition
        Download avatar -> data/avatars/<user_id>.jpg
        Run SCRFD face detection -> faces_count
        If faces_count == 1: avatar becomes face_photo_path
        If faces_count != 1: fall back to last N post photos (face leader)

Step 5: Resolve Telegram contacts via Sherlock (parallel)
        For "naked" leads (profile fetched, bio gave no phone/telegram).
        Stage 1: nick search (cheap, ~30s) -- POST /v1/search/nick; hits
                 are logged to Telegram report_chat only (not saved to DB).
        Stage 2: photo search (slow, ~135s) always runs when face_photo_path
                 exists on disk -- POST /v1/search/photo; only this stage
                 writes sherlock_status / phone / link. Skipped under
                 --skip-sherlock or if SHERLOCK_API_KEY missing.
        Sets sherlock_processed_at on every terminal photo outcome
        (found_photo, no_match, no_face_photo, error) so leads aren't
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

**`processed_posts`** — posts that passed Step 1 (`comments_count >= min_comments_per_post`)
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
- `username`, `user_id`, `post_url`, `post_shortcode`, `comment_text`, `comment_pk`, `comment_at`

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
python scripts/pipeline.py --skip-sherlock   # Steps 1-4 + 6 only
python scripts/pipeline.py --keep-photos     # skip Step 6 face cleanup
python scripts/pipeline.py -y                # auto-confirm Step 5 cost prompt only

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

- `config.yaml` — search parameters, Apify actor IDs, and per-step limits under `pipeline.stepN.*` (Step 1 post age and min-comments fallbacks live in `src.config` when those keys are omitted).
  - `pipeline.stepN.*` — per-step tuning knobs (post age, min comments,
    Step 1 `discovery_mode` (`realtors` | `hashtags` | `cookie_keywords`),
    `hashtag_results_limit`, `pipeline.step1.cookie_search.*`, comment caps,
    batch sizes, Sherlock cap, `prompt_terminal_confirmation`). Missing keys fall
    back to safe defaults (`pipeline.step1` post age / min comments via `src.config`;
    other steps use `DEFAULT_*` in `scripts/pipeline_lib/defaults.py`), so a fresh /
    partial config still boots.
- `.env` — secrets: `APIFY_API_TOKEN`, `DEEPSEEK_API_KEY`, `NEXARA_API_KEY`,
  `SHERLOCK_API_KEY`; optional `TELEGRAM_BOT_TOKEN` for notifications/confirmations;
  optional `INSTAGRAM_SESSION_COOKIE` (or env from
  `pipeline.step1.cookie_search.session_cookie_env_var`) for `discovery_mode=cookie_keywords`
- Step 1 realtor usernames: `search.realtor_accounts` in `config.yaml` when
  `discovery_mode` is `realtors` (not the `tracked_realtors` table)

## Key Source Files

- `scripts/pipeline.py` — daily pipeline orchestration (Steps 1–6)
- `scripts/pipeline_lib/defaults.py` — `DEFAULT_*` fallbacks when config keys are missing
- `scripts/pipeline_lib/apify_runner.py` — `_fetch_comments_with_fallback` (Step 3 Apify)
- `scripts/pipeline_lib/scoring.py` — Step 2 Lingua gate, DeepSeek scoring, human confirm
- `scripts/pipeline_lib/step4_faces.py` — avatar face area, post-photo leader helpers
- `scripts/pipeline_lib/step5_sherlock.py` — Step 5 Sherlock worker pool
- `scripts/pipeline_lib/step6_cleanup.py` — Step 6 face asset cleanup
- `src/db.py` — SQLite DB with all tables, dedup logic, lead lifecycle methods
- `src/sherlock_client.py` — Sherlock HTTP client (Step 5)
- `src/telegram_notifier.py` — pipeline Telegram notifications (aiogram)
- `src/telegram_inline_confirm.py` — inline yes/no confirm for Step 2/3
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
- `docs/apify_api_schemas.md` — Apify actors used by the pipeline (inputs/outputs)
- `models/` — vendored ML weights (InsightFace `buffalo_s` only);
  committed to the repo so Ubuntu deploys don't re-download ~155 MB on
  first use. See `models/README.md` for layout and Git LFS tips.
- `facetest/` — dev-only sandbox for `scripts/test_face_matcher.py`

## Architecture Principles

- **Cost awareness:** Apify requests cost money. Always deduplicate — check DB before making API calls. Track and log costs per cycle.
- **Dedup by user_id:** Instagram usernames can change. Always check `user_id` (numeric pk) for deduplication, not just username.
- **Budget controls:** Step 3 (and Step 5) show estimated cost and ask for confirmation when `pipeline.prompt_terminal_confirmation` is true; set false for cron/unattended runs.
- **Incremental:** Each pipeline run only processes new/changed data. Safe to run repeatedly.
- **5% comment growth threshold:** Don't re-scan comments on a post unless comment count grew by at least 5% since last scan.
- **Disk hygiene:** face photos (avatars + post-photo leaders) live on disk only between Step 4 and Step 6. After Sherlock finishes a lead with a non-error terminal status, Step 6 unlinks its files and NULLs the path columns. Face-detection helper queries gate on `sherlock_processed_at IS NULL` so cleaned leads aren't re-fetched via Apify.

## Language

The project spec and communication are in Russian. Code, comments, variable names, and logs should be in English.
