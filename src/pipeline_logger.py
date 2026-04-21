"""JSON pipeline logger — writes a structured record for every Apify run.

Each script session creates one JSON file in logs/ with an array of run records.
Easy to load later for cost analysis: json.load() -> list[dict].
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class PipelineLogger:
    """Accumulates run records and flushes them to a single JSON file."""

    def __init__(self, log_dir: str = "logs", session_name: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = session_name or "session"
        self.log_file = self.log_dir / f"{name}_{ts}.json"

        self.records: list[dict] = []

    def log_run(
        self,
        *,
        actor_id: str,
        run_id: str,
        status: str,
        input_params: dict,
        items_count: int,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
        dataset_id: str | None = None,
        sample_items: list[dict] | None = None,
        error: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Log a single Apify actor run. Returns the record dict."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor_id": actor_id,
            "run_id": run_id,
            "status": status,
            "input_params": _sanitize(input_params),
            "items_count": items_count,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "dataset_id": dataset_id,
            "cost_per_item": (
                round(cost_usd / items_count, 6)
                if cost_usd and items_count
                else None
            ),
        }
        if sample_items:
            record["sample_items"] = sample_items
        if error:
            record["error"] = error
        if extra:
            record.update(extra)

        self.records.append(record)
        self._flush()
        return record

    def _flush(self) -> None:
        """Write all records to the JSON file (overwrites each time)."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @property
    def file_path(self) -> str:
        return str(self.log_file)

    def summary(self) -> dict:
        """Return aggregate stats across all logged runs."""
        total_cost = sum(r.get("cost_usd") or 0 for r in self.records)
        total_items = sum(r.get("items_count") or 0 for r in self.records)
        return {
            "total_runs": len(self.records),
            "total_items": total_items,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_item": (
                round(total_cost / total_items, 6) if total_items else None
            ),
            "log_file": self.file_path,
        }


def _sanitize(obj: dict) -> dict:
    """Remove proxy/token fields from input before logging."""
    skip = {"proxy", "token", "apify_token"}
    return {k: v for k, v in obj.items() if k.lower() not in skip}
