"""Shared ASCII table printer for SQLite dump scripts."""

from __future__ import annotations

import sqlite3
import sys
from typing import Any, Iterable, Sequence


def ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows consoles (cp1251)."""
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

# Columns that may hold very long text — tighter cap keeps the table readable.
_WIDE_TEXT_COLUMNS = frozenset(
    {
        "caption",
        "biography",
        "latest_media_urls",
        "profile_pic_url",
        "profile_pic_url_hd",
        "external_url",
        "post_url",
        "sherlock_link",
    }
)

_BOOLISH_COLUMNS = frozenset(
    {
        "is_private",
        "is_verified",
        "is_business",
        "has_cta",
        "profile_fetched",
        "contact_found",
    }
)


def _format_cell(name: str, value: Any, *, max_width: int) -> str:
    if value is None:
        return "—"
    if name in _BOOLISH_COLUMNS:
        if value in (0, "0", False):
            return "no"
        if value in (1, "1", True):
            return "yes"
        return str(value)
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    cap = 24 if name in _WIDE_TEXT_COLUMNS else max_width
    if len(text) > cap:
        return text[: cap - 1] + "…"
    return text


def _col_widths(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_width: int,
) -> list[int]:
    widths = [len(c) for c in columns]
    for row in rows:
        for i, (col, val) in enumerate(zip(columns, row, strict=True)):
            cell = _format_cell(col, val, max_width=max_width)
            widths[i] = max(widths[i], len(cell))
    return [min(w, max_width + 6) for w in widths]


def _pad(text: str, width: int) -> str:
    if len(text) >= width:
        return text
    return text + " " * (width - len(text))


def print_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: str | None = None,
    max_cell_width: int = 36,
    footer_lines: Iterable[str] = (),
) -> None:
    """Print rows as a bordered ASCII table."""
    ensure_utf8_stdout()
    if not columns:
        print("(no columns)")
        return

    widths = _col_widths(columns, rows, max_width=max_cell_width)
    h_sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row_cells(values: Sequence[Any]) -> list[str]:
        return [
            _pad(_format_cell(col, val, max_width=max_cell_width), widths[i])
            for i, (col, val) in enumerate(zip(columns, values, strict=True))
        ]

    def _print_row(cells: Sequence[str]) -> None:
        print("| " + " | ".join(cells) + " |")

    if title:
        print()
        print(title)
        print("=" * len(title))

    print(h_sep)
    _print_row(list(columns))
    print(h_sep)
    if not rows:
        empty = ["(empty)"] + [""] * (len(columns) - 1)
        _print_row(empty[: len(columns)])
    else:
        for row in rows:
            _print_row(_row_cells(row))
    print(h_sep)
    print(f"{len(rows)} row(s), {len(columns)} column(s)")
    for line in footer_lines:
        print(line)


def fetch_all(
    conn: sqlite3.Connection,
    table: str,
    *,
    order_by: str,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Return (column_names, rows) for a whitelisted table."""
    allowed = {"processed_posts", "lead_accounts"}
    if table not in allowed:
        raise ValueError(f"table not allowed: {table!r}")
    cur = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}")
    columns = [d[0] for d in cur.description or []]
    rows = [tuple(r) for r in cur.fetchall()]
    return columns, rows
