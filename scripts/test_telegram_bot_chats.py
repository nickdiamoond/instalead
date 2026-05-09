"""
Dev helper: print chat IDs seen in recent Bot API updates.

Telegram does not expose a global "list all chats" for bots. This script
collects IDs from unconfirmed getUpdates() payloads:

- Chats where users sent messages (or edits) to the bot / in groups
- Chats from channel posts if the bot is an admin poster
- Chats from my_chat_member when the bot was added or membership changed

Then sends a fixed test string to ``telegram.report_chat_id`` from config.yaml.

Run once, then send a message to the bot or add it to a group and run again
if the output is empty. If you use a webhook, delete it first or this
script will see no polling updates.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root on sys.path for consistent imports if we extend this later
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class ChatIndex:
    """Aggregated chat ids keyed by how we learned about them."""

    with_messages: dict[int, str] = field(default_factory=dict)
    membership_events: dict[int, str] = field(default_factory=dict)

    def record_message_chat(self, chat_id: int, chat_type: str | None) -> None:
        self.with_messages[chat_id] = chat_type or "unknown"

    def record_member_chat(self, chat_id: int, chat_type: str | None) -> None:
        self.membership_events[chat_id] = chat_type or "unknown"


def _index_updates(index: ChatIndex, updates: list) -> None:
    from aiogram.types import Update

    for u in updates:
        if not isinstance(u, Update):
            continue
        if u.message and u.message.chat:
            c = u.message.chat
            index.record_message_chat(c.id, c.type)
        if u.edited_message and u.edited_message.chat:
            c = u.edited_message.chat
            index.record_message_chat(c.id, c.type)
        if u.channel_post and u.channel_post.chat:
            c = u.channel_post.chat
            index.record_message_chat(c.id, c.type)
        if u.edited_channel_post and u.edited_channel_post.chat:
            c = u.edited_channel_post.chat
            index.record_message_chat(c.id, c.type)
        if u.callback_query and u.callback_query.message and u.callback_query.message.chat:
            c = u.callback_query.message.chat
            index.record_message_chat(c.id, c.type)
        if u.my_chat_member and u.my_chat_member.chat:
            c = u.my_chat_member.chat
            index.record_member_chat(c.id, c.type)


def _load_report_chat_id() -> int:
    path = _ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    telegram_cfg = (cfg or {}).get("telegram") or {}
    raw = telegram_cfg.get("report_chat_id")
    if raw is None:
        raise SystemExit(f"Missing telegram.report_chat_id in {path}")
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"Invalid telegram.report_chat_id: {raw!r}") from e


async def main() -> None:
    load_dotenv(_ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing in environment (.env)")

    report_chat_id = _load_report_chat_id()

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
        updates = await bot.get_updates(limit=100)

        index = ChatIndex()
        _index_updates(index, updates)

        print("\nChats with at least one message-like update in this batch:")
        if not index.with_messages:
            print("  (none — send a DM or a group message mentioning the bot, then re-run)")
        else:
            for cid, ctype in sorted(index.with_messages.items()):
                print(f"  chat_id={cid}  type={ctype}")

        print("\nChats seen via my_chat_member (bot added / rights changed):")
        if not index.membership_events:
            print("  (none in current update window)")
        else:
            for cid, ctype in sorted(index.membership_events.items()):
                print(f"  chat_id={cid}  type={ctype}")

        union = sorted(set(index.with_messages) | set(index.membership_events))
        print("\nUnion (all distinct chat ids above):")
        if not union:
            print("  (empty)")
        else:
            for cid in union:
                print(f"  {cid}")

        #await bot.send_message(chat_id=report_chat_id, text="Hello World")
        #print(f'\nSent "Hello World" to telegram.report_chat_id={report_chat_id}')


if __name__ == "__main__":
    asyncio.run(main())
