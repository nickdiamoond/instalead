"""Send a message with inline confirm/deny buttons; first click wins, then edit text.

Reads ``telegram.result_chat_id`` (fallback: ``report_chat_id``) from ``config.yaml``
and ``TELEGRAM_BOT_TOKEN``
from the environment (same as ``src.telegram_notifier``). Run from repo root:

    python scripts/test_telegram_callback_buttons.py

Optional: ``TELEGRAM_CALLBACK_TEST_TIMEOUT_SEC=3600`` — on timeout, buttons are removed only.

Stop with Ctrl+C if nobody presses a button (when no timeout is set).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv

from src.telegram_notifier import (
    TOKEN_ENV_VAR,
    _parse_report_chat_id,
    _parse_result_chat_id,
)

# Hardcoded body shown before any answer (suffix added after first click).
BASE_MESSAGE_TEXT = "пример работы"

CALLBACK_CONFIRM = "tcb_confirm"
CALLBACK_DENY = "tcb_deny"

_TIMEOUT_ENV = "TELEGRAM_CALLBACK_TEST_TIMEOUT_SEC"


def _load_cfg() -> dict:
    root = Path(__file__).resolve().parent.parent
    path = root / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _timeout_sec() -> float | None:
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


async def main() -> None:
    load_dotenv()
    token = (os.environ.get(TOKEN_ENV_VAR) or "").strip()
    if not token:
        raise SystemExit(f"Set {TOKEN_ENV_VAR} in the environment or .env")

    cfg = _load_cfg()
    chat_id = _parse_result_chat_id(cfg) or _parse_report_chat_id(cfg)
    if chat_id is None:
        raise SystemExit(
            "config.yaml: telegram.result_chat_id and telegram.report_chat_id "
            "missing or invalid"
        )

    bot = Bot(token=token)
    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    lock = asyncio.Lock()
    state = {"handled": False}
    msg_id_holder: dict[str, int] = {}
    done = asyncio.Event()

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Подтвердить",
                    callback_data=CALLBACK_CONFIRM,
                ),
                InlineKeyboardButton(text="❌ Нет", callback_data=CALLBACK_DENY),
            ],
        ]
    )

    sent = await bot.send_message(
        chat_id,
        BASE_MESSAGE_TEXT,
        reply_markup=keyboard,
    )
    msg_id_holder["id"] = sent.message_id
    print(f"sent message_id={sent.message_id} chat_id={chat_id}")

    @router.callback_query(F.data.in_({CALLBACK_CONFIRM, CALLBACK_DENY}))
    async def on_choice(cb: CallbackQuery) -> None:
        if cb.message is None or cb.message.message_id != msg_id_holder["id"]:
            await cb.answer()
            return

        async with lock:
            if state["handled"]:
                await cb.answer("Already handled", show_alert=False)
                return
            state["handled"] = True

        suffix = " (подтверждено)" if cb.data == CALLBACK_CONFIRM else " (нет)"
        new_text = BASE_MESSAGE_TEXT + suffix
        await cb.message.edit_text(new_text, reply_markup=None)
        await cb.answer()
        done.set()
        await dp.stop_polling()

    poll_task = asyncio.create_task(dp.start_polling(bot))
    timeout = _timeout_sec()

    try:
        if timeout is not None:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        else:
            await done.wait()
    except asyncio.TimeoutError:
        print(f"timeout ({timeout}s): removing inline keyboard only")
        async with lock:
            if not state["handled"]:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=msg_id_holder["id"],
                        reply_markup=None,
                    )
                except Exception as exc:  # noqa: BLE001 — dev script
                    print(f"edit_message_reply_markup failed: {exc}")
    finally:
        await dp.stop_polling()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
