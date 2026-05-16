"""Standalone test: same face-photo flow as pipeline Step 4 for one profile URL.

Fetches the profile via ``apify/instagram-profile-scraper``, downloads the
avatar, runs the avatar embedder; promotes the avatar only when there is
exactly one face and its bbox covers at least ``face_detection.min_avatar_face_area_pct``
percent of the image area (same gate as ``scripts/pipeline.py``). Otherwise
probes the last N posts (one image per post), downloads them, and runs
:func:`src.face_leader.resolve_face_leader` with the post embedder —
mirroring ``scripts/pipeline.py`` (without DB writes).

The winning image is copied to ``facetest/profile_face_winner/`` so temp
files under ``data/avatars`` and ``data/lead_photos`` stay aligned with the
production layout while you still get a stable artifact for inspection.

If a winner exists and ``SHERLOCK_API_KEY`` is set, the script then runs the
same ``POST /v1/search/photo`` flow as ``scripts/test_sherlock_photo.py`` on
that file (avatar winner vs post-leader uses ``face_kind`` ``avatar`` / ``post``
for the optional local SCRFD preamble). No Sherlock call is made when there is
no winner, or when the API key is missing.

Edit ``INSTAGRAM_PROFILE_URL`` below, then run::

    python scripts/test_profile_face_pick.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
from openai import OpenAI

from src.face_embedder import make_face_embedder
from src.sherlock_client import (
    API_KEY_ENV_VAR,
    SherlockClient,
    SherlockError,
    make_sherlock_client,
)

# ---------------------------------------------------------------------------
# Hardcoded test target — replace with any public Instagram profile URL.
# ---------------------------------------------------------------------------
INSTAGRAM_PROFILE_URL = "https://www.instagram.com/isabelsir/"

# Output directory (under project root, separate from production DB paths).
OUTPUT_REL = Path("facetest") / "profile_face_winner"

_RESERVED_USER_SEGMENTS = frozenset(
    {
        "p",
        "reel",
        "reels",
        "stories",
        "explore",
        "accounts",
        "direct",
        "tv",
        "legal",
        "about",
    }
)


def _username_from_instagram_url(url: str) -> str:
    m = re.search(r"instagram\.com/([^/?#]+)", url.strip(), re.I)
    if not m:
        raise ValueError(f"Could not parse Instagram username from URL: {url!r}")
    user = m.group(1).strip().lstrip("@")
    if not user or user.lower() in _RESERVED_USER_SEGMENTS:
        raise ValueError(
            f"URL segment {user!r} is not a profile username — use e.g. "
            "https://www.instagram.com/<username>/"
        )
    return user


def _pick_post_images(
    latest_posts: list[dict] | None,
    limit: int,
    *,
    skip_videos: bool = True,
) -> list[str]:
    """Same selection rules as ``scripts.pipeline._pick_post_images``."""
    if not latest_posts:
        return []

    urls: list[str] = []
    for post in latest_posts[:limit]:
        images = post.get("images") or []
        if images and images[0]:
            urls.append(images[0])
            continue
        display_url = post.get("displayUrl")
        video_url = post.get("videoUrl")
        if not display_url:
            continue
        if skip_videos and video_url:
            continue
        urls.append(display_url)
    return urls


# Same default as ``scripts/pipeline.DEFAULT_MIN_AVATAR_FACE_AREA_PCT`` —
# overridden by ``face_detection.min_avatar_face_area_pct`` in config.yaml.
DEFAULT_MIN_AVATAR_FACE_AREA_PCT = 2.0


def face_bbox_percent_of_image(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    """BBox vs full raster: ``(area_percent, width_percent, height_percent)``."""
    x1, y1, x2, y2 = bbox
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    iw = float(image_width)
    ih = float(image_height)
    if iw <= 0.0 or ih <= 0.0:
        return (0.0, 0.0, 0.0)
    area_pct = 100.0 * (bw * bh) / (iw * ih)
    w_pct = 100.0 * bw / iw
    h_pct = 100.0 * bh / ih
    return (area_pct, w_pct, h_pct)


def _mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} (len={len(key)})"


def _print_sherlock_result(result: dict | None) -> None:
    if not result:
        print("  result:  (empty)")
        return
    print("  result:")
    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    for line in formatted.splitlines():
        print("    " + line)


# First-row ``status`` substring for Sherlock photo ``result.results`` (see
# ``scripts/pipeline._resolve_one_lead_via_sherlock`` photo stage).
SHERLOCK_EXACT_MATCH_SUBSTRING = "точное совпадение"
USERMATCH_PROMPT = """\
# Задача
Ты анализируешь Ник пользователя из Instagram (username) и его ФИО из профиля (если оно присутствует), а также пронумерованный список потенциальных кандидатов.
Ты должен определить, кому из кандидатов принадлежит этот аккаунт, либо вернуть 0, если аккаунт не удалось уверенно сопоставить ни с одним кандидатом.

Иногда ник содержит (частично) имя или фамилию кандидата. Если ФИО в профиле пустое или состоит из неинформативных слов (например, «блогер», «рилсмейкер»), приоритет отдаётся нику.
Если ФИО присутствует и похоже на реальное имя, оно получает приоритет, но ник может использоваться для уточнения.

# Алгоритм анализа
Выполняй шаги строго по порядку.

1. **Извлеки значимые части из ФИО пользователя.**
   - Удали из строки всё, что не является именем или фамилией (слова вроде «блогер», «рилсмейкер», «official», эмодзи и т.п.).
   - Оставшиеся слова считай набором значимых частей (порядок не фиксирован). Их может быть 0, 1, 2 или более.
   - Если значимых частей нет, считай ФИО пустым.

2. **Сопоставление по ФИО.**
   - Если в ФИО есть два или более значащих слов, проверь, содержатся ли среди них одновременно имя и фамилия какого-либо кандидата (порядок слов в ФИО и у кандидата роли не играет). При полном совпадении (найдены и имя, и фамилия одного кандидата) уверенность 10/10 – сразу верни номер этого кандидата. Если таких кандидатов несколько (одинаковые ФИО), верни номер первого из них.
   - Если значимое слово одно, сравни его с именами и фамилиями всех кандидатов. Возможные ситуации:
     - Слово совпало с именем одного кандидата (и с фамилиями не совпадает) – зафиксируй «имя найдено» и переходи к шагу 3, используя найденное имя.
     - Слово совпало с фамилией одного кандидата (и с именами не совпадает) – зафиксируй «фамилия найдена» и переходи к шагу 3.
     - Слово совпало с именем одного кандидата и с фамилией другого – однозначного сопоставления нет, уверенность <7/10, верни 0.
     - Если слово не совпало ни с одним именем или фамилией – переходи к шагу 3, считая ФИО не давшим зацепок.

3. **Анализ ника.**
   - Преобразуй ник из латиницы в вероятные русские варианты (транслитерация с латиницы на кириллицу). Используй стандартную обратную транслитерацию, учитывая типичные для Instagram сокращения:
     - `ov` → `ов`, `ev` → `ев`, `iy`/`y` на конце → `ий`/`ый`, `a` → `а`, `o` → `о`, `e` → `е` (или `э`), `zh` → `ж`, `sh` → `ш`, `ch` → `ч`, `ya` → `я`, `yu` → `ю`, `kh` → `х`, `ts` → `ц` и т.д.
     - Примеры: `ivanov` → `Иванов`, `oleg` → `Олег`, `smirnoff` → `Смирнов`, `anna.petrova` → `Анна Петрова` (разделитель `.` / `_` / `-` можно трактовать как пробел).
   - Полученную русскую строку (или набор слов, если были разделители) сравни с именами и фамилиями кандидатов.

   Далее действуй по ситуации (с учётом того, что ФИО могло дать зацепки на шаге 2):

   **А) Если из ФИО уже известно имя** (одно слово совпало с именем кандидата):
      - Среди кандидатов с таким же именем найди того, чья фамилия (полностью или начальная часть) содержится в транслитерированном нике. Например, имя `Олег` и ник `oleg.ivanov` → фамилия `Иванов` найдена.
      - Если такой кандидат ровно один, уверенность 9/10 – верни его номер.
      - Если подходящих кандидатов несколько (однофамильцы с разными именами не могут быть, т.к. мы уже отфильтровали по имени; остаются только дубли ФИО) – верни номер первого из них.
      - Если ник не содержит фамилии ни одного из кандидатов с этим именем, уверенность ниже 7/10 – верни 0.

   **Б) Если из ФИО известна только фамилия** (одно слово совпало с фамилией):
      - Проверь, содержит ли транслитерированный ник ещё и имя того кандидата, чья фамилия найдена.
      - Если имя кандидата обнаружено в нике (вместе с фамилией) и такой кандидат один – уверенность 9/10, верни его номер.
      - Если имя не найдено, но фамилия уникальна (встречается ровно у одного кандидата) – уверенность 8/10, верни номер.
      - Если фамилия есть у нескольких кандидатов **с разными именами** – уверенность ниже 7/10, верни 0.
      - Если фамилия есть у нескольких кандидатов **с одинаковыми ФИО** (дубли) – считай это одним кандидатом и верни номер первого.

   **В) Если ФИО полностью отсутствует или не дало зацепок:**
      - Ищи в транслитерированном нике одновременно имя и фамилию одного кандидата (в любом порядке, возможно с разделителями). При уникальном совпадении – уверенность 9/10, верни номер.
      - Если найдена только фамилия и она принадлежит ровно одному кандидату – уверенность 8/10, верни его номер.
      - Если найдена только фамилия, но она есть у нескольких кандидатов с разными именами – уверенность ниже 7/10, верни 0. Если же это дубли (одинаковые ФИО), верни номер первого.
      - Если найдено только имя, а кандидатов с таким именем больше одного – уверенность ниже 7/10, верни 0.

4. **Финальная проверка уверенности.**
   - Возвращай номер кандидата, только если уверенность по описанным правилам не ниже 7 из 10.
   - Если ни один из шагов не дал достаточной уверенности, верни 0.
   - **Всегда, когда в списке есть кандидаты с полностью одинаковыми ФИО, при совпадении отдавай номер первого из них.**

#Данные:
Ник пользователя из Instagram: "{username}"
ФИО пользователя из Instagram: "{full_name}"

Список потенциальных кандидатов: {candidates}

#Формат ответа:
В ответе ты должен указать номер кандидата, которому принадлежит этот аккаунт только в том случае, если ты уверен, что этот аккаунт принадлежит этому кандидату минимум на 7/10. Если ты не уверен, отдай 0.
Если ты нашел совпадение, но в списке потенциальных кандидатов есть еще такие же одинаковые ФИО, отдай номер первого из них.
Ответь ТОЛЬКО одной цифрой, которая соответствует номеру кандидата, либо 0, если ник пользователя из Instagram и его ФИО не принадлежат ни одному кандидату.
"""


def _sherlock_photo_results_list(result: dict | None) -> list:
    """Normalize ``result["results"]`` like ``pipeline._resolve_one_lead_via_sherlock``."""
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
    """``1) `` + name + ``\\n`` for each entry (1-based), same order as ``persons``."""
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


def _deepseek_usermatch_pick(
    client: OpenAI,
    *,
    ig_username: str,
    ig_full_name: str,
    persons: list,
    phones: list,
    statuses: list,
) -> None:
    """Same transport as ``scripts.pipeline.score_caption`` (OpenAI client → DeepSeek API)."""
    candidates_block = _format_candidates_for_prompt(persons)
    system_prompt = USERMATCH_PROMPT.format(
        username=ig_username,
        full_name=ig_full_name,
        candidates=candidates_block,
    )
    print("-" * 70)
    print("DeepSeek user match (USERMATCH_PROMPT) ...")
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
    except Exception as exc:
        print(f"DeepSeek request failed: {exc}", file=sys.stderr)
        return

    pick = _parse_usermatch_digit(raw)
    if pick is None:
        print(f"DeepSeek: не удалось разобрать ответ ({raw!r}).")
        return
    if pick == 0:
        print("Дипсик совпадений не нашел.")
        return
    idx = pick - 1
    if idx < 0 or idx >= len(persons):
        print(
            f"DeepSeek: индекс {pick} вне диапазона "
            f"(кандидатов {len(persons)}). Ответ: {raw!r}"
        )
        return
    print("DeepSeek pick:")
    print(f"  person:  {persons[idx]!r}")
    print(f"  phone:   {phones[idx]!r}")
    print(f"  status:  {statuses[idx]!r}")


def _print_sherlock_photo_results_digest(
    result: dict | None,
    *,
    ig_username: str,
    ig_full_name: str,
    deepseek: OpenAI | None,
) -> None:
    """After the full result JSON: branch on ``results[0].status`` (pipeline analogy)."""
    results = _sherlock_photo_results_list(result)
    print("-" * 70)
    print("Sherlock results digest:")
    if not results:
        print("  (no result.results)")
        return

    first = results[0]
    if not isinstance(first, dict):
        first = {}
    first_status = str(first.get("status") or "")

    if SHERLOCK_EXACT_MATCH_SUBSTRING in first_status:
        print(f"  person: {first.get('person')!r}")
        print(f"  phone:  {first.get('phone')!r}")
        print(f"  status: {first.get('status')!r}")
        return

    persons: list = []
    phones: list = []
    statuses: list = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if "person" not in item or item.get("person") is None:
            continue
        persons.append(_person_for_digest_list(item.get("person")))
        phones.append(item.get("phone"))
        statuses.append(item.get("status"))

    if not persons:
        print("  (no candidates with a non-null ``person`` field)")
        return
    if deepseek is None:
        print(
            "  DEEPSEEK_API_KEY not set — skipping numbered-candidate match "
            "(arrays not printed)."
        )
        return
    _deepseek_usermatch_pick(
        deepseek,
        ig_username=ig_username,
        ig_full_name=ig_full_name,
        persons=persons,
        phones=phones,
        statuses=statuses,
    )


def describe_local_face_detection(
    photo_path: Path,
    cfg: dict,
    kind: str,
) -> None:
    """Same pre-submit SCRFD pass as ``scripts/test_sherlock_photo.py``."""
    fd = cfg.get("face_detection") or {}
    min_score = float(fd.get("min_det_score", 0.6))
    det_size = int(
        fd.get(
            "avatar_det_size" if kind == "avatar" else "post_det_size",
            320 if kind == "avatar" else 640,
        )
    )

    print(f"  kind:           {kind!r}  (det_size={det_size}x{det_size})")
    print(f"  min_det_score:  {min_score}  (from config.yaml face_detection)")

    embedder = make_face_embedder(cfg, kind=kind)
    embedder.min_det_score = 0.0

    t0 = time.monotonic()
    faces = embedder.embed_faces(photo_path)
    elapsed_ms = (time.monotonic() - t0) * 1000

    embedder.close()

    print(f"  raw detections: {len(faces)}  ({elapsed_ms:.0f} ms incl. cold load)")
    if not faces:
        print(
            "  -> NO face detected at any score. Sherlock will likely "
            "reject the photo or return zero matches."
        )
        return

    kept = [f for f in faces if f.det_score >= min_score]
    print(f"  above threshold: {len(kept)} / {len(faces)}")
    print("  per-face:")
    for i, f in enumerate(sorted(faces, key=lambda x: -x.det_score), 1):
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        verdict = "KEEP" if f.det_score >= min_score else "drop"
        print(
            f"    #{i}  det_score={f.det_score:.3f}  "
            f"bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  "
            f"size={w:.0f}x{h:.0f}  [{verdict}]"
        )

    if len(kept) == 0:
        print(
            "  -> face(s) found but all BELOW min_det_score "
            f"({min_score}). For an avatar this usually means a side "
            "view / occlusion / very small face — Sherlock may still "
            "match but quality is uncertain."
        )
    elif len(kept) == 1:
        print("  -> exactly 1 face above threshold = ideal Sherlock input.")
    else:
        print(
            f"  -> {len(kept)} faces above threshold. Sherlock will "
            "match against the largest/most-confident one."
        )


def _make_sherlock_poll_progress_callback():
    def on_poll(poll_count: int, elapsed_s: float, task: dict) -> None:
        status = (task.get("status") or "").lower()
        attempts = task.get("attempts")
        max_attempts = task.get("max_attempts")
        account_id = task.get("account_id")
        sys.stdout.write(
            f"\r  [poll #{poll_count:>3}]  t+{elapsed_s:>5.1f}s  "
            f"status={status:<10}  attempt={attempts}/{max_attempts}  "
            f"account_id={account_id}    "
        )
        sys.stdout.flush()

    return on_poll


def _run_sherlock_photo_search(
    photo_path: Path,
    cfg: dict,
    *,
    face_kind: str,
    ig_username: str,
    ig_full_name: str,
    deepseek: OpenAI | None,
) -> int:
    """POST ``/v1/search/photo`` and poll. Skips entirely if API key missing."""
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip().strip("'\"")
    if not api_key:
        print(f"{API_KEY_ENV_VAR} not set — skipping Sherlock photo search.")
        return 0

    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    sh_cfg = cfg.get("sherlock") or {}
    photo_cfg = sh_cfg.get("photo_search") or {}
    task_cfg = sh_cfg.get("task") or {}
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))
    max_pages = int(photo_cfg.get("max_pages", 20))
    max_attempts = int(photo_cfg.get("max_attempts", 3))
    priority = int(photo_cfg.get("priority", 0))

    client: SherlockClient
    try:
        client = make_sherlock_client(cfg, api_key=api_key)
    except EnvironmentError as exc:
        print(f"Sherlock client error: {exc}", file=sys.stderr)
        return 1

    base_url = client.base_url
    http_timeout = client.http_timeout

    size_kb = photo_path.stat().st_size / 1024
    print("=" * 70)
    print("Sherlock photo search (same flow as scripts/test_sherlock_photo.py)")
    print(f"Base URL:      {base_url}")
    print(f"Photo:         {photo_path.resolve()}  ({size_kb:.1f} KB)")
    print(f"API key:       {_mask_key(api_key)}")
    print(f"max_pages:     {max_pages}")
    print(f"priority:      {priority}")
    print(f"max_attempts:  {max_attempts}")
    print(f"poll_interval: {poll_interval}s   max_wait: {max_wait}s")
    print("=" * 70)

    print("Step 0: local SCRFD face detection on winner photo ...")
    try:
        describe_local_face_detection(
            photo_path=photo_path,
            cfg=cfg,
            kind=face_kind,
        )
    except Exception as exc:
        print(f"  (face detection skipped: {exc})")
    print("-" * 70)
    print("Step 1: POST /v1/search/photo ...")

    try:
        enq = client.enqueue_photo(
            photo_path,
            max_pages=max_pages,
            priority=priority,
            max_attempts=max_attempts,
        )
    except (requests.RequestException, SherlockError) as exc:
        print(f"ERROR: enqueue failed: {exc}", file=sys.stderr)
        client.close()
        return 1

    task_id = enq.get("id")
    if not task_id:
        print(f"ERROR: enqueue response missing 'id': {enq}", file=sys.stderr)
        client.close()
        return 1

    print(f"  -> task_id:   {task_id}")
    print(f"     scenario:  {enq.get('scenario')!r}")
    print(f"     status:    {enq.get('status')!r}")
    print(f"     priority:  {enq.get('priority')}")
    print(f"     created:   {enq.get('created_at')}")
    print()
    print("Step 2: polling task state until terminal status ...")

    on_poll = _make_sherlock_poll_progress_callback()
    t_poll_start = time.monotonic()
    try:
        task = client.wait_for_task(
            task_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_poll=on_poll,
        )
    except TimeoutError as exc:
        sys.stdout.write("\n")
        print(f"ERROR: {exc}", file=sys.stderr)
        client.close()
        return 1
    except (requests.RequestException, SherlockError) as exc:
        sys.stdout.write("\n")
        print(f"ERROR: polling failed: {exc}", file=sys.stderr)
        client.close()
        return 1
    elapsed = time.monotonic() - t_poll_start
    sys.stdout.write("\n")

    final_status = (task.get("status") or "").lower()
    print()
    print("=" * 70)
    print(f"Step 3: task finished in {elapsed:.1f}s with status={final_status!r}")
    print("-" * 70)
    print(f"  account_id:    {task.get('account_id')}")
    print(f"  attempts:      {task.get('attempts')}/{task.get('max_attempts')}")
    print(f"  started_at:    {task.get('started_at')}")
    print(f"  finished_at:   {task.get('finished_at')}")

    if final_status == "completed":
        res = task.get("result")
        _print_sherlock_result(res)
        _print_sherlock_photo_results_digest(
            res,
            ig_username=ig_username,
            ig_full_name=ig_full_name,
            deepseek=deepseek,
        )
    else:
        err_code = task.get("error_code")
        err_msg = task.get("error_message")
        print(f"  error_code:    {err_code!r}")
        print(f"  error_message: {err_msg!r}")
        if task.get("result"):
            res = task.get("result")
            _print_sherlock_result(res)
            _print_sherlock_photo_results_digest(
                res,
                ig_username=ig_username,
                ig_full_name=ig_full_name,
                deepseek=deepseek,
            )

    client.close()
    print("=" * 70)
    return 0 if final_status == "completed" else 1


def main() -> int:
    repo_root = _REPO_ROOT

    from apify_client import ApifyClient
    from dotenv import load_dotenv

    from src.avatar_downloader import (
        cleanup_lead_photos,
        download_avatar,
        download_post_photos,
    )
    from src.config import load_config
    from src.face_leader import resolve_face_leader
    from src.logger import get_logger, setup_logging

    setup_logging()
    log = get_logger("test_profile_face_pick")

    load_dotenv()
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        print("APIFY_API_TOKEN is not set.", file=sys.stderr)
        return 1

    username = _username_from_instagram_url(INSTAGRAM_PROFILE_URL)
    cfg = load_config()
    fb_cfg = cfg.get("face_fallback") or {}
    fb_limit = int(fb_cfg.get("latest_posts_limit", 5))
    fb_min_cluster = int(fb_cfg.get("min_cluster_size", 2))
    fb_threshold = float(fb_cfg.get("cluster_threshold", 0.5))
    fb_skip_videos = bool(fb_cfg.get("skip_videos", True))
    fb_keep_photos = bool(fb_cfg.get("keep_photos", False))

    fd_cfg = cfg.get("face_detection") or {}
    min_avatar_face_area_pct = float(
        fd_cfg.get("min_avatar_face_area_pct", DEFAULT_MIN_AVATAR_FACE_AREA_PCT)
    )

    avatar_embedder = make_face_embedder(cfg, kind="avatar")
    post_embedder = make_face_embedder(cfg, kind="post")

    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ApifyClient(token)
    log.info("apify_profile_fetch", username=username)
    run = client.actor("apify/instagram-profile-scraper").call(
        run_input={"usernames": [username]},
    )
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        print(f"No dataset items returned for @{username}.", file=sys.stderr)
        return 1

    p = items[0]
    if p.get("username") and p["username"].lower() != username.lower():
        log.warning(
            "username_mismatch",
            requested=username,
            returned=p.get("username"),
        )

    if p.get("private"):
        print(f"Profile @{username} is private — cannot download media.", file=sys.stderr)
        return 1

    # From Apify item (for reuse / logging); URL-derived ``username`` stays the request handle.
    profile_username = p.get("username")
    profile_full_name = p.get("fullName")
    print(
        "Apify profile fields:\n"
        f"\n\n  username: {profile_username!r}\n\n"
        f"\n\n  fullName: {profile_full_name!r}\n\n"
    )

    avatar_url = p.get("profilePicUrlHD") or p.get("profilePicUrl")
    uid = p.get("id") or p.get("pk")
    uid_str = str(uid) if uid else None

    avatar_path_str = download_avatar(
        avatar_url,
        user_id=uid_str,
        username=username,
    )
    if not avatar_path_str:
        print("Avatar download failed (missing URL or HTTP error).", file=sys.stderr)
        return 1

    avatar_path = Path(avatar_path_str)
    avatar_faces = avatar_embedder.embed_faces(avatar_path)
    faces_count = len(avatar_faces)
    winner_path: Path | None = None
    source = "avatar"

    avatar_area_ok = False
    if faces_count == 1:
        img_bgr = cv2.imread(str(avatar_path))
        if img_bgr is not None:
            ih, iw = img_bgr.shape[:2]
            area_pct, _, _ = face_bbox_percent_of_image(
                avatar_faces[0].bbox, iw, ih
            )
            avatar_area_ok = area_pct >= min_avatar_face_area_pct

    if faces_count == 1 and avatar_area_ok:
        winner_path = avatar_path
    elif uid_str:
        post_urls = _pick_post_images(
            p.get("latestPosts"),
            limit=fb_limit,
            skip_videos=fb_skip_videos,
        )
        local_paths = download_post_photos(post_urls, user_id=uid_str)
        result = resolve_face_leader(
            local_paths,
            post_embedder,
            min_cluster_size=fb_min_cluster,
            cluster_threshold=fb_threshold,
        )
        if result:
            winner_path = result.photo_path
            source = "post_fallback"
        if not fb_keep_photos:
            cleanup_lead_photos(uid_str, keep=(result.photo_path if result else None))
    else:
        print(
            "Avatar has no single face and profile has no numeric id — "
            "cannot run post fallback.",
            file=sys.stderr,
        )

    if winner_path is None or not winner_path.is_file():
        hint = ""
        if faces_count == 1 and not avatar_area_ok:
            hint = " (one face on avatar but below min_avatar_face_area_pct — post fallback did not yield a winner)"
        elif faces_count != 1 and uid_str:
            hint = " (post fallback did not yield a winner)"
        print(
            f"No suitable single-face winner (avatar faces={faces_count}){hint}.",
            file=sys.stderr,
        )
        return 1

    dest = out_dir / f"{username}_face_winner.jpg"
    shutil.copy2(winner_path, dest)
    print(f"Winner ({source}): {winner_path}")
    print(f"Copied to: {dest}")
    log.info(
        "test_profile_face_pick_done",
        username=username,
        source=source,
        avatar_faces=faces_count,
        dest=str(dest),
    )

    # Same /v1/search/photo flow as scripts/test_sherlock_photo.py on the winner file only.
    face_kind = "avatar" if source == "avatar" else "post"
    ds_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip().strip("'\"")
    deepseek_client: OpenAI | None = None
    if ds_key:
        deepseek_client = OpenAI(
            api_key=ds_key,
            base_url="https://api.deepseek.com",
        )
    ig_u = (profile_username or username or "").strip()
    ig_fn = (profile_full_name or "")
    if isinstance(ig_fn, str):
        ig_fn = ig_fn.strip()
    else:
        ig_fn = str(ig_fn) if ig_fn is not None else ""
    return _run_sherlock_photo_search(
        winner_path,
        cfg,
        face_kind=face_kind,
        ig_username=ig_u,
        ig_full_name=ig_fn,
        deepseek=deepseek_client,
    )


if __name__ == "__main__":
    raise SystemExit(main())
