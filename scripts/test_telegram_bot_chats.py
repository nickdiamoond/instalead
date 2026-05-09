"""
Dev helper: list group / supergroup chats the bot appears in, using Bot API
polling data.

Telegram does not expose a global "list all chats" for bots. This script drains
pending getUpdates() batches (short timeout) and aggregates chats from:

- Messages, edits, channel posts, callback queries
- my_chat_member (bot added / rights changed)

Only chats of type ``group`` and ``supergroup`` are printed.

If the list is empty, send a message in the target group or add the bot, then
re-run. If you use a webhook, delete it first or polling will see nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root on sys.path for consistent imports if we extend this later
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_GROUP_CHAT_TYPES = frozenset({"group", "supergroup"})


def _chat_type_str(chat) -> str:
    t = chat.type
    return t.value if hasattr(t, "value") else str(t)


def _merge_chat(store: dict[int, dict], chat) -> None:
    cid = chat.id
    ctype = _chat_type_str(chat)
    title = getattr(chat, "title", None) or None
    if cid not in store:
        store[cid] = {"type": ctype, "title": title}
        return
    if title:
        store[cid]["title"] = title


def _index_updates(store: dict[int, dict], updates: list) -> None:
    from aiogram.types import Update

    for u in updates:
        if not isinstance(u, Update):
            continue
        if u.message and u.message.chat:
            _merge_chat(store, u.message.chat)
        if u.edited_message and u.edited_message.chat:
            _merge_chat(store, u.edited_message.chat)
        if u.channel_post and u.channel_post.chat:
            _merge_chat(store, u.channel_post.chat)
        if u.edited_channel_post and u.edited_channel_post.chat:
            _merge_chat(store, u.edited_channel_post.chat)
        if u.callback_query and u.callback_query.message and u.callback_query.message.chat:
            _merge_chat(store, u.callback_query.message.chat)
        if u.my_chat_member and u.my_chat_member.chat:
            _merge_chat(store, u.my_chat_member.chat)


async def _drain_pending_updates(bot, *, batch_limit: int = 100, max_batches: int = 50) -> list:
    """Fetch consecutive update pages until empty or max_batches (Telegram buffer)."""
    all_updates: list = []
    offset: int | None = None
    for _ in range(max_batches):
        batch = await bot.get_updates(limit=batch_limit, offset=offset, timeout=1)
        if not batch:
            break
        all_updates.extend(batch)
        offset = batch[-1].update_id + 1
        if len(batch) < batch_limit:
            break
    return all_updates


async def main() -> None:
    load_dotenv(_ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing in environment (.env)")

    from aiogram import Bot

    async with Bot(token=token) as bot:
        wh = await bot.get_webhook_info()
        if wh.url:
            print(
                f"Warning: webhook is set ({wh.url}). "
                "getUpdates may return nothing until the webhook is removed."
            )
        me = await bot.get_me()
        print(f"Bot: @{me.username} (id={me.id})")

        updates = await _drain_pending_updates(bot)
        store: dict[int, dict] = {}
        _index_updates(store, updates)

        groups = [
            (cid, info)
            for cid, info in store.items()
            if info["type"] in _GROUP_CHAT_TYPES
        ]
        groups.sort(key=lambda item: ((item[1]["title"] or "").lower(), item[0]))

        print("\nGroup chats (group + supergroup) from pending updates:")
        if not groups:
            print(
                "  (none — trigger activity in a group the bot is in, then re-run)\n"
                "  e.g. a message, /start, or add the bot so my_chat_member appears."
            )
        else:
            for cid, info in groups:
                title = info["title"] or "(no title)"
                print(f"  chat_id={cid}  type={info['type']}  title={title}")


if __name__ == "__main__":
    asyncio.run(main())
