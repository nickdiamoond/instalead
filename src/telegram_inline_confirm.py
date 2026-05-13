"""Single-message inline yes/no confirmation via aiogram (first callback wins).

Used by Step 2 human review in ``scripts/pipeline.py``. Do not call from inside
an existing event loop together with ``asyncio.run``-based helpers that start
their own loop (e.g. ``PipelineTelegramNotifier._send_sync``).

Optional timeout: set ``TELEGRAM_CALLBACK_TEST_TIMEOUT_SEC`` to a positive
number of seconds; on timeout the inline keyboard is removed and this returns
``timeout`` without changing caller-side DB state.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Literal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

_TIMEOUT_ENV = "TELEGRAM_CALLBACK_TEST_TIMEOUT_SEC"

# Short ASCII callback_data (Bot API limit 64 bytes per button).
_DEFAULT_CALLBACK_YES = "ic_yes"
_DEFAULT_CALLBACK_NO = "ic_no"

InlineConfirmResult = Literal["yes", "no", "timeout"]


def _timeout_sec() -> float | None:
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


async def await_single_yes_no(
    token: str,
    chat_id: int,
    message_text: str,
    *,
    confirm_button_text: str,
    deny_button_text: str,
    suffix_yes: str,
    suffix_no: str,
    callback_yes: str = _DEFAULT_CALLBACK_YES,
    callback_no: str = _DEFAULT_CALLBACK_NO,
) -> InlineConfirmResult:
    """Send ``message_text`` with two inline buttons; wait for first click.

    Edits the message to append ``suffix_yes`` or ``suffix_no`` and removes
    the keyboard. Returns ``timeout`` if env timeout elapses with no click
    (keyboard stripped only).
    """
    bot = Bot(token=token)
    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    lock = asyncio.Lock()
    state = {"handled": False}
    msg_id_holder: dict[str, int] = {}
    done = asyncio.Event()
    outcome_holder: dict[str, InlineConfirmResult] = {}

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=confirm_button_text,
                    callback_data=callback_yes,
                ),
                InlineKeyboardButton(text=deny_button_text, callback_data=callback_no),
            ],
        ]
    )

    sent = await bot.send_message(chat_id, message_text, reply_markup=keyboard)
    msg_id_holder["id"] = sent.message_id

    @router.callback_query(F.data.in_({callback_yes, callback_no}))
    async def on_choice(cb: CallbackQuery) -> None:
        if cb.message is None or cb.message.message_id != msg_id_holder["id"]:
            await cb.answer()
            return

        async with lock:
            if state["handled"]:
                await cb.answer("Already handled", show_alert=False)
                return
            state["handled"] = True

        picked_yes = cb.data == callback_yes
        suffix = suffix_yes if picked_yes else suffix_no
        new_text = message_text + suffix
        await cb.message.edit_text(new_text, reply_markup=None)
        await cb.answer()
        outcome_holder["v"] = "yes" if picked_yes else "no"
        done.set()
        await dp.stop_polling()

    poll_task = asyncio.create_task(dp.start_polling(bot))
    timeout = _timeout_sec()

    result: InlineConfirmResult = "timeout"
    try:
        if timeout is not None:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        else:
            await done.wait()
        result = outcome_holder.get("v", "timeout")
    except asyncio.TimeoutError:
        async with lock:
            if not state["handled"]:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=msg_id_holder["id"],
                        reply_markup=None,
                    )
                except Exception:
                    pass
    finally:
        await dp.stop_polling()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        await bot.session.close()

    return result
