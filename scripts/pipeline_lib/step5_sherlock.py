import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from src.db import LeadDB
from src.sherlock_client import SherlockError, make_sherlock_client
from src.telegram_notifier import PipelineTelegramNotifier

from scripts.pipeline_lib.constants import (
    NICK_TASK_ETA_S,
    PHOTO_TASK_ETA_S,
    SHERLOCK_EXACT_MATCH_SUBSTRING,
    SH_STATUS_ERROR,
    SH_STATUS_FOUND_NICK,
    SH_STATUS_FOUND_PHOTO,
    SH_STATUS_NO_FACE_PHOTO,
    SH_STATUS_NO_MATCH,
)
from scripts.pipeline_lib.io_utils import _banner, _format_eta
from scripts.pipeline_lib.logging import log as pipeline_log


def _sherlock_photo_results_list(result: dict | None) -> list:
    """Normalize ``result["results"]`` for photo search (dict or list)."""
    if not result or not isinstance(result, dict):
        return []
    raw = result.get("results")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def _person_for_digest_list(person: object) -> object:
    """Keep the substring before the last space (trim trailing tokens like a DOB)."""
    if not isinstance(person, str):
        return person
    if " " not in person:
        return person
    return person.rsplit(" ", 1)[0]


def _format_candidates_for_prompt(persons: list) -> str:
    """``1) `` + name + ``\\n`` for each entry (1-based)."""
    return "".join(f"{i}) {p}\n" for i, p in enumerate(persons, start=1))


def _parse_usermatch_digit(raw: str) -> int | None:
    """First signed integer in model output, or ``None`` if unparseable."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    return int(m.group(0))


def _deepseek_usermatch_pick_index(
    client: OpenAI,
    *,
    ig_username: str,
    ig_full_name: str,
    persons: list,
    usermatch_prompt: str,
) -> tuple[int | None, str | None]:
    """Return ``(pick, api_error)``.

    * ``pick`` — 1-based candidate index, or ``None`` if the model declines
      (digit ``0``) or the call failed.
    * ``api_error`` — set when the HTTP call or response parsing failed;
      ``None`` when the API returned a usable answer (including decline).
    """
    candidates_block = _format_candidates_for_prompt(persons)
    system_prompt = usermatch_prompt.format(
        username=ig_username,
        full_name=ig_full_name,
        candidates=candidates_block,
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Ответь одной цифрой."},
            ],
            temperature=0,
            max_tokens=16,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        pipeline_log.warning(
            "step5_deepseek_usermatch_failed",
            username=ig_username,
            error=str(exc),
        )
        return None, str(exc)

    pick = _parse_usermatch_digit(raw)
    if pick is None:
        pipeline_log.warning(
            "step5_deepseek_usermatch_unparseable",
            username=ig_username,
            raw=raw[:200],
        )
        return None, f"unparseable response: {raw[:200]}"
    if pick == 0:
        pipeline_log.info(
            "step5_deepseek_usermatch_zero",
            username=ig_username,
        )
        return None, None
    if pick < 1 or pick > len(persons):
        pipeline_log.warning(
            "step5_deepseek_usermatch_out_of_range",
            username=ig_username,
            pick=pick,
            n=len(persons),
            raw=raw[:200],
        )
        return None, f"out of range: pick={pick} n={len(persons)}"
    return pick, None


def _resolve_one_lead_via_sherlock(
    sherlock,
    lead: dict,
    *,
    nick_cfg: dict,
    photo_cfg: dict,
    task_cfg: dict,
    deepseek: OpenAI | None = None,
    usermatch_prompt: str,
) -> dict:
    """Run the full nick->photo flow for one lead.

    Pure function w.r.t. the DB: returns a dict that the orchestrator
    persists via :py:meth:`LeadDB.mark_lead_sherlock`. Never raises --
    every exception path resolves to ``status=error`` with a populated
    ``error`` message so a single misbehaving lead doesn't sink the
    whole batch.
    """
    username = lead["username"]
    out: dict = {
        "username": username,
        "status": SH_STATUS_ERROR,
        "telegram_username": None,
        "phone": None,
        "sherlock_link": None,
        "error": None,
        "nick_skipped_dot": "." in username,
        "nick_search_ran": False,
        "nick_hit": False,
        "photo_search_ran": False,
        "photo_task": None,
        "nick_query": None if "." in username else f"@{username}",
    }
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))

    if "." not in username:
        nick_query = f"@{username}"
        out["nick_search_ran"] = True
        try:
            enq = sherlock.enqueue_nick(
                nick=nick_query,
                search_in="telegram",
                max_pages=int(nick_cfg.get("max_pages", 1)),
                max_attempts=int(nick_cfg.get("max_attempts", 3)),
            )
            task = sherlock.wait_for_task(
                enq["id"],
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            if (task.get("status") or "").lower() == "completed":
                results = ((task.get("result") or {}).get("results")) or []
                if isinstance(results, dict):
                    results = [results]
                match = next(
                    (
                        item for item in results
                        if isinstance(item, dict) and item.get("profile_url")
                    ),
                    None,
                )
                if match:
                    tg_username = match.get("username") or username
                    tg_link = (
                        match.get("link")
                        or f"https://t.me/{tg_username}"
                    )
                    out.update({
                        "status": SH_STATUS_FOUND_NICK,
                        "telegram_username": tg_username,
                        "sherlock_link": tg_link,
                        "nick_hit": True,
                    })
                    return out
        except (SherlockError, TimeoutError) as exc:
            out["error"] = f"nick: {exc}"
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"nick: unexpected {type(exc).__name__}: {exc}"

    face_path_str = lead.get("face_photo_path")
    if not face_path_str:
        out["status"] = SH_STATUS_NO_FACE_PHOTO
        out["error"] = None
        return out

    face_path = Path(face_path_str)
    if not face_path.is_file():
        out["status"] = SH_STATUS_NO_FACE_PHOTO
        out["error"] = f"face_photo missing on disk: {face_path}"
        return out

    try:
        enq = sherlock.enqueue_photo(
            face_path,
            max_pages=int(photo_cfg.get("max_pages", 20)),
            max_attempts=int(photo_cfg.get("max_attempts", 3)),
        )
        task = sherlock.wait_for_task(
            enq["id"],
            poll_interval=poll_interval,
            max_wait=max_wait,
        )
        out["photo_search_ran"] = True
        out["photo_task"] = copy.deepcopy(task)
        final_status = (task.get("status") or "").lower()
        if final_status != "completed":
            out["status"] = SH_STATUS_ERROR
            out["error"] = (
                task.get("error_message")
                or f"photo task ended status={final_status!r}"
            )
            return out
        result_block = task.get("result") or {}
        results = _sherlock_photo_results_list(result_block)
        if not results:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            return out
        first_raw = results[0]
        first = first_raw if isinstance(first_raw, dict) else {}
        first_status = str(first.get("status") or "")

        if SHERLOCK_EXACT_MATCH_SUBSTRING in first_status:
            out.update({
                "status": SH_STATUS_FOUND_PHOTO,
                "phone": first.get("phone"),
                "sherlock_link": first.get("link"),
                "error": None,
                "photo_match_kind": "exact",
                "sherlock_person": first.get("person"),
            })
            pipeline_log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="exact_match",
            )
            return out

        persons: list = []
        raw_persons: list = []
        phones: list = []
        links: list = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if "person" not in item or item.get("person") is None:
                continue
            persons.append(_person_for_digest_list(item.get("person")))
            raw_persons.append(item.get("person"))
            phones.append(item.get("phone"))
            links.append(item.get("link"))

        if not persons:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            pipeline_log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="no_person_candidates",
            )
            return out

        if deepseek is None:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            pipeline_log.warning(
                "step5_sherlock_photo_no_deepseek_client",
                username=username,
            )
            return out

        pick, deepseek_api_error = _deepseek_usermatch_pick_index(
            deepseek,
            ig_username=username,
            ig_full_name=str(lead.get("full_name") or ""),
            persons=persons,
            usermatch_prompt=usermatch_prompt,
        )
        out["step5_deepseek_called"] = True
        out["step5_deepseek_api_failed"] = deepseek_api_error is not None
        if pick is None:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            pipeline_log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="no_contact_after_disambiguation",
            )
            return out

        idx = pick - 1
        out.update({
            "status": SH_STATUS_FOUND_PHOTO,
            "phone": phones[idx],
            "sherlock_link": links[idx],
            "error": None,
            "photo_match_kind": "deepseek",
            "sherlock_person": raw_persons[idx],
        })
        pipeline_log.info(
            "step5_sherlock_photo_outcome",
            username=username,
            branch="deepseek_pick",
            pick=pick,
        )
        return out
    except (SherlockError, TimeoutError) as exc:
        out["status"] = SH_STATUS_ERROR
        out["error"] = f"photo: {exc}"
        return out
    except Exception as exc:  # noqa: BLE001
        out["status"] = SH_STATUS_ERROR
        out["error"] = f"photo: unexpected {type(exc).__name__}: {exc}"
        return out


def _step_5_resolve_contacts_via_sherlock(
    db: LeadDB,
    cfg: dict,
    *,
    batch_limit: int,
    workers_override: int | None,
    sequential: bool,
    request_gap_secs: float,
    auto_yes: bool,
    log,
    issues: list[tuple[str, str]],
    tg_notifier: PipelineTelegramNotifier,
    deepseek: OpenAI | None,
    usermatch_prompt: str,
) -> None:
    """Run Sherlock contact resolution for naked leads (parallel or sequential)."""
    _banner("STEP 5: Resolve contacts via Sherlock")

    sh_cfg = cfg.get("sherlock") or {}
    nick_cfg = sh_cfg.get("nick_search") or {}
    photo_cfg = sh_cfg.get("photo_search") or {}
    task_cfg = sh_cfg.get("task") or {}
    conc_cfg = sh_cfg.get("concurrency") or {}

    try:
        sherlock = make_sherlock_client(cfg)
    except EnvironmentError as exc:
        print(f"  SKIPPED: cannot build Sherlock client ({exc}).")
        log.warning("step5_skip_no_client", error=str(exc))
        issues.append(("Step 5", f"Sherlock client missing: {exc}"))
        return

    try:
        gap_s = max(0.0, float(request_gap_secs))
        if sequential:
            workers = 1
            workers_source = "sequential (pipeline.step5.sequential)"
        elif workers_override is not None:
            workers = max(1, int(workers_override))
            workers_source = "--workers"
        elif conc_cfg.get("workers"):
            workers = max(1, int(conc_cfg["workers"]))
            workers_source = "config.yaml sherlock.concurrency.workers"
        else:
            workers = sherlock.get_pool_idle(fallback=3)
            workers_source = "/v1/health pool.idle"

        candidates = db.get_leads_for_sherlock(limit=batch_limit)
        with_face = sum(1 for c in candidates if c.get("face_photo_path"))

        n = len(candidates)
        if sequential:
            best_eta = n * NICK_TASK_ETA_S
            worst_eta = n * NICK_TASK_ETA_S + with_face * PHOTO_TASK_ETA_S
            if gap_s > 0 and n > 1:
                best_eta += gap_s * (n - 1)
                worst_eta += gap_s * (n - 1)
        else:
            best_eta = (n * NICK_TASK_ETA_S) / max(workers, 1)
            worst_eta = (
                n * NICK_TASK_ETA_S + with_face * PHOTO_TASK_ETA_S
            ) / max(workers, 1)

        if not sequential and gap_s > 0:
            print(
                "  NOTE: request_gap_secs is ignored unless sequential mode "
                "(pipeline.step5.sequential)."
            )
            log.warning(
                "step5_gap_ignored_not_sequential",
                request_gap_secs=gap_s,
            )

        print(f"  Candidates:        {n}")
        print(f"  With face photo:   {with_face}  (eligible for photo fallback)")
        print(f"  Workers:           {workers}  (from {workers_source})")
        if sequential and gap_s > 0:
            print(
                f"  Gap between leads: {gap_s}s  (pipeline.step5.request_gap_secs)"
            )
        print(f"  Best-case ETA:     {_format_eta(best_eta)}  "
              f"(every nick search hits)")
        print(f"  Worst-case ETA:    {_format_eta(worst_eta)}  "
              f"(every photo fallback runs)")

        if n == 0:
            print("  SKIPPED: no candidates.")
            log.info("step5_no_candidates")
            return

        if not auto_yes:
            confirm = input("  Proceed? (y/n): ").strip().lower()
            if confirm != "y":
                print("  SKIPPED by user.")
                log.info("step5_skipped_by_user")
                return

        log.info(
            "step5_start",
            candidates=n,
            with_face=with_face,
            workers=workers,
            workers_source=workers_source,
            sequential=sequential,
            request_gap_secs=gap_s if sequential else 0.0,
        )

        counters: dict[str, int] = {
            SH_STATUS_FOUND_NICK: 0,
            SH_STATUS_FOUND_PHOTO: 0,
            SH_STATUS_NO_MATCH: 0,
            SH_STATUS_NO_FACE_PHOTO: 0,
            SH_STATUS_ERROR: 0,
        }
        step5_deepseek_calls = 0
        step5_deepseek_api_ok = 0

        def _worker_error_payload(username: str, exc: Exception) -> dict:
            return {
                "username": username,
                "status": SH_STATUS_ERROR,
                "telegram_username": None,
                "phone": None,
                "sherlock_link": None,
                "error": f"worker crashed: {type(exc).__name__}: {exc}",
                "nick_skipped_dot": "." in username,
                "nick_search_ran": False,
                "nick_hit": False,
                "photo_search_ran": False,
                "photo_task": None,
                "nick_query": None if "." in username else f"@{username}",
            }

        def _apply_step5_result(lead: dict, res: dict, progress_i: int) -> None:
            nonlocal step5_deepseek_calls, step5_deepseek_api_ok
            username = lead["username"]
            db.mark_lead_sherlock(
                username=username,
                status=res["status"],
                telegram_username=res.get("telegram_username"),
                phone=res.get("phone"),
                sherlock_link=res.get("sherlock_link"),
            )
            tg_notifier.notify_sherlock_lead(lead, res, cfg=cfg)
            if res.get("step5_deepseek_called"):
                step5_deepseek_calls += 1
                if not res.get("step5_deepseek_api_failed"):
                    step5_deepseek_api_ok += 1
            counters[res["status"]] = counters.get(res["status"], 0) + 1
            tag = res["status"]
            detail_bits: list[str] = []
            if res.get("telegram_username"):
                detail_bits.append(f"tg=@{res['telegram_username']}")
            if res.get("phone"):
                detail_bits.append(f"phone={res['phone']}")
            if res.get("sherlock_link"):
                detail_bits.append(f"link={res['sherlock_link']}")
            if res.get("error") and res["status"] == SH_STATUS_ERROR:
                detail_bits.append(f"err={res['error'][:80]}")
            detail = "  " + " ".join(detail_bits) if detail_bits else ""
            print(f"  [{progress_i:>4}/{n}] @{username:<25} -> {tag}{detail}")
            log.info(
                "step5_lead_done",
                username=username,
                status=tag,
                telegram_username=res.get("telegram_username"),
                phone=bool(res.get("phone")),
                error=res.get("error"),
            )

        if sequential:
            for idx, lead in enumerate(candidates, 1):
                if idx > 1 and gap_s > 0:
                    time.sleep(gap_s)
                username = lead["username"]
                try:
                    res = _resolve_one_lead_via_sherlock(
                        sherlock,
                        lead,
                        nick_cfg=nick_cfg,
                        photo_cfg=photo_cfg,
                        task_cfg=task_cfg,
                        deepseek=deepseek,
                        usermatch_prompt=usermatch_prompt,
                    )
                except Exception as exc:  # noqa: BLE001
                    res = _worker_error_payload(username, exc)
                _apply_step5_result(lead, res, idx)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sherlock"
            ) as pool:
                futures = {
                    pool.submit(
                        _resolve_one_lead_via_sherlock,
                        sherlock,
                        lead,
                        nick_cfg=nick_cfg,
                        photo_cfg=photo_cfg,
                        task_cfg=task_cfg,
                        deepseek=deepseek,
                        usermatch_prompt=usermatch_prompt,
                    ): lead
                    for lead in candidates
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    lead = futures[fut]
                    username = lead["username"]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _worker_error_payload(username, exc)
                    _apply_step5_result(lead, res, i)

        print(f"\n  DONE: {n} lead(s) processed via Sherlock")
        tg_notifier.notify_step5_sherlock_summary(
            pulled=n,
            batch_limit=batch_limit,
            counters=counters,
            step5_deepseek_calls=step5_deepseek_calls,
            step5_deepseek_api_ok=step5_deepseek_api_ok,
        )

        log.info(
            "step5_done",
            step5_deepseek_calls=step5_deepseek_calls,
            step5_deepseek_api_ok=step5_deepseek_api_ok,
            **{f"count_{k}": v for k, v in counters.items()},
        )

        error_count = counters.get(SH_STATUS_ERROR, 0)
        tg_notifier.maybe_notify_sherlock_batch_all_failed(
            leads_processed=n,
            error_count=error_count,
        )
        tg_notifier.maybe_notify_deepseek_batch_all_failed(
            deepseek_calls=step5_deepseek_calls,
            deepseek_succeeded=step5_deepseek_api_ok,
            step="Step 5",
            call_kind="usermatch call(s)",
            outcome_label="usermatch picks",
        )

        if error_count:
            issues.append((
                "Step 5",
                f"{error_count} leads finished as error -- "
                "check logs for Sherlock task failures / timeouts",
            ))
    finally:
        sherlock.close()

