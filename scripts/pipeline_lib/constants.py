CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

# Sherlock outcome labels stored in lead_accounts.sherlock_status. Kept
# as a centralized vocabulary so the pipeline summary banner and
# downstream tooling can rely on a closed set of values.
SH_STATUS_FOUND_NICK = "found_nick"
SH_STATUS_FOUND_PHOTO = "found_photo"
SH_STATUS_NO_MATCH = "no_match"
SH_STATUS_NO_FACE_PHOTO = "no_face_photo"
SH_STATUS_ERROR = "error"

# First-row ``status`` substring for Sherlock photo ``result.results``;
# mirrors ``scripts/test_profile_face_pick.py``.
SHERLOCK_EXACT_MATCH_SUBSTRING = "точное совпадение"

# Per-task wallclock estimates from our smoke tests against the live
# service. Used only for the cost-confirmation banner -- actual times
# fluctuate with TG-side latency and pool saturation.
NICK_TASK_ETA_S = 30
PHOTO_TASK_ETA_S = 135
