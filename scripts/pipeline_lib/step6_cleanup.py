from src.avatar_downloader import cleanup_lead_face_assets
from src.db import LeadDB

from scripts.pipeline_lib.io_utils import _banner


def _step_6_cleanup_spent_face_assets(
    db: LeadDB,
    *,
    log,
    issues: list[tuple[str, str]],
) -> None:
    """Delete avatars / face photos for leads Sherlock has finished with.

    Idempotent: once both columns are NULL the lead is excluded from
    :py:meth:`LeadDB.get_leads_with_spent_photos`, so subsequent runs
    only touch new spent leads. Never raises -- per-lead unlink
    failures land in ``issues`` for the summary banner.
    """
    _banner("STEP 6: Cleanup spent face assets")

    candidates = db.get_leads_with_spent_photos()
    if not candidates:
        print("  SKIPPED: no spent face assets to clean.")
        log.info("step6_no_candidates")
        return

    print(f"  Leads to clean:    {len(candidates)}")
    log.info("step6_cleanup_spent_assets", count=len(candidates))

    files_deleted = 0
    files_failed = 0
    leads_cleaned = 0

    for lead in candidates:
        username = lead["username"]
        deleted, failed = cleanup_lead_face_assets(
            lead.get("avatar_path"),
            lead.get("face_photo_path"),
            user_id=lead.get("user_id"),
        )
        db.mark_lead_photos_cleaned(username)
        files_deleted += deleted
        files_failed += failed
        leads_cleaned += 1

    print(
        f"  DONE: cleaned {leads_cleaned} leads, "
        f"{files_deleted} files removed, "
        f"{files_failed} failed"
    )
    log.info(
        "step6_done",
        leads_cleaned=leads_cleaned,
        files_deleted=files_deleted,
        files_failed=files_failed,
    )

    if files_failed:
        issues.append((
            "Step 6",
            f"{files_failed} files failed to unlink during cleanup -- "
            "check warnings in logs (avatar_downloader). Lead rows "
            "were still marked cleaned to avoid re-trying.",
        ))
