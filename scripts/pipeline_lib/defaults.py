# Per-step tuning knobs. These are *defaults*; ``main()`` overrides
# every value from ``config.yaml`` (``pipeline.stepN.*``) so the daily
# run picks up changes without a code edit. Constants are kept in the
# module (rather than only in YAML) so direct imports / unit tests
# don't have to pull in the config loader to know reasonable values.
#
# Don't touch these to "tune the pipeline"; edit ``config.yaml``
# instead. They exist solely to keep the script bootable when a key
# is missing from a fresh config (e.g. on a brand-new machine before
# the operator has copied the canonical YAML over).

# Step 1: numeric defaults for ``posts_max_age_days`` and
# ``min_comments_per_post`` live in ``src.config`` (``step1_*`` helpers).
# ``apify/instagram-post-scraper`` input ``resultsLimit`` (per username).
# Override via ``pipeline.step1.post_scraper_results_limit``.
DEFAULT_POST_SCRAPER_RESULTS_LIMIT = 20
# Step 1: ``pipeline.step1.discovery_mode`` — ``realtors`` | ``hashtags`` |
# ``cookie_keywords``.
DEFAULT_STEP1_DISCOVERY_MODE = "realtors"

# Step 3: comment re-scan growth threshold + the displayed cost
# estimate per fetched comment (real bill comes from
# ``run.usageTotalUsd`` regardless of this value).
DEFAULT_COMMENTS_GROWTH_PCT = 5.0
DEFAULT_COST_PER_COMMENT = 0.0005

# Step 4: profile-scraper batch size + max new leads pulled per run.
DEFAULT_PROFILE_BATCH_SIZE = 50
DEFAULT_STEP4_BATCH_LIMIT = 1000
# Minimum bbox area (percent of full raster) to accept the avatar as the
# canonical face photo; below this, Step 4 uses the post-photo leader path.
DEFAULT_MIN_AVATAR_FACE_AREA_PCT = 2.0

# Step 5: max leads pulled per run from get_leads_for_sherlock.
# Smaller than Step 4's effective rate because Sherlock tasks are
# slow (~30 s nick / ~135 s photo each), so 1000 leads on a 3-account
# pool is already a multi-hour run; bigger pools could safely raise
# this. The daily run has no CLI flag for it on purpose, to keep
# ``python scripts/pipeline.py`` behavior reproducible across
# machines / cron jobs -- override via config.yaml.
DEFAULT_SHERLOCK_BATCH_LIMIT = 1000
DEFAULT_SHERLOCK_SEQUENTIAL = True
DEFAULT_SHERLOCK_REQUEST_GAP_SECS = 5.0

# Step 3 comment scrapers. louisdeconinck is the primary because its
# snake_case Instagram-raw output maps 1:1 to ``lead_accounts`` columns
# and to ``apify/instagram-profile-scraper`` (Step 4) -- no field
# remapping needed downstream. apidojo-api is the fallback: it has been
# observed to keep working when louisdeconinck silently returns 0 items
# with status=SUCCEEDED. Its camelCase output is normalized via
# :func:`src.comment_normalizer.normalize_apidojo_api` before saving.
#
# These constants are the *defaults*. ``main()`` overrides them from
# ``config.yaml`` (``apify.actors.comments_primary`` /
# ``apify.actors.comments_fallback``) so a switch to a different actor
# is a config edit instead of a code change.
DEFAULT_COMMENTS_PRIMARY_ACTOR = "louisdeconinck/instagram-comments-scraper"
DEFAULT_COMMENTS_FALLBACK_ACTOR = "apidojo/instagram-comments-scraper-api"

# louisdeconinck silently returns 0 items with status=SUCCEEDED if its
# input is missing a per-post comment cap -- bisected via
# ``scripts/test_comment_scrapers.py`` (recipe 1 -> recipe 3). The
# fallback (apidojo-api) uses ``maxItems`` (run-wide); see
# ``DEFAULT_APIFY_COMMENTS_CAP_PER_POST`` / ``apidojo_comments_cap_per_post``.
#
# 10_000 is a *ceiling*, not a target: the actor returns only
# comments that actually exist on the post, so a higher cap doesn't
# raise our bill -- it just protects against losing the tail on a
# viral post. Max ``comments_count`` observed in our DB is ~2_200
# (avg ~130), so 10_000 leaves ~5x headroom for unexpected spikes.
# The cap is applied on the primary's call only -- see
# ``_fetch_comments_with_fallback``. Override via
# ``pipeline.step3.louisdeconinck_comments_cap_per_post`` in config.
DEFAULT_LOUISDECONINCK_COMMENTS_CAP_PER_POST = 10_000

# apidojo-api (Step 3 fallback) exposes ``maxItems`` as a run-wide total,
# not per-post. The pipeline sets ``maxItems = cap * len(urls)`` so the
# knob mirrors louisdeconinck's per-post ceiling. Override via
# ``pipeline.step3.apidojo_comments_cap_per_post`` in config.
DEFAULT_APIFY_COMMENTS_CAP_PER_POST = 10_000

# When true, Step 3 / Step 5 ask ``Proceed? (y/n)`` before spendy work, and
# the script waits for Enter after reporting issues. Set false in config for
# cron / unattended runs (``pipeline.prompt_terminal_confirmation``).
DEFAULT_PROMPT_TERMINAL_CONFIRMATION = True
