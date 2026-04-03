"""SQLite-backed call history for the intelligent router."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelStats:
    """Aggregate statistics for a model over a set of recorded calls."""
    call_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_cost: float = 0.0
    avg_cost: float = 0.0
    success_rate: float = 0.0
    refusal_rate: float = 0.0


@dataclass
class CallRecord:
    """A single recorded LLM call."""
    model_alias: str
    latency_ms: float
    cost: float
    outcome: str   # e.g. "success", "refused", "error", "timeout"
    intent: str
    timestamp: float = field(default_factory=time.time)


class LearningStore:
    """
    SQLite-backed store for per-model call history.

    Records every LLM call (latency, cost, outcome, intent) and
    provides aggregate stats queries with optional intent and time-window filters.
    """

    def __init__(self, db_path: str = "~/.ghostrouter/learning.db") -> None:
        resolved = Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(resolved), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_alias TEXT    NOT NULL,
                latency_ms  REAL    NOT NULL,
                cost        REAL    NOT NULL,
                outcome     TEXT    NOT NULL,
                intent      TEXT    NOT NULL,
                timestamp   REAL    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calls_alias     ON calls (model_alias)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls (timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calls_intent    ON calls (intent)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        model_alias: str,
        latency_ms: float,
        cost: float,
        outcome: str,
        intent: str,
    ) -> None:
        """Insert a call record with the current timestamp."""
        self._conn.execute(
            """
            INSERT INTO calls (model_alias, latency_ms, cost, outcome, intent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_alias, latency_ms, cost, outcome, intent, time.time()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def stats(
        self,
        model_alias: str,
        *,
        intent: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> ModelStats:
        """
        Return aggregate stats for *model_alias*.

        Optional filters:
          intent         — restrict to calls with this intent value
          window_seconds — restrict to calls within the last N seconds
        """
        filters = ["model_alias = ?"]
        params: list = [model_alias]

        if intent is not None:
            filters.append("intent = ?")
            params.append(intent)

        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            filters.append("timestamp >= ?")
            params.append(cutoff)

        where = " AND ".join(filters)

        # --- basic aggregates -------------------------------------------------
        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*)                                          AS call_count,
                AVG(latency_ms)                                   AS avg_latency_ms,
                SUM(cost)                                         AS total_cost,
                AVG(cost)                                         AS avg_cost,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS success_rate,
                SUM(CASE WHEN outcome = 'refused' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS refusal_rate
            FROM calls
            WHERE {where}
            """,
            params,
        ).fetchone()

        call_count = row[0] if row[0] else 0

        if call_count == 0:
            return ModelStats()

        avg_latency_ms = row[1] or 0.0
        total_cost     = row[2] or 0.0
        avg_cost       = row[3] or 0.0
        success_rate   = row[4] or 0.0
        refusal_rate   = row[5] or 0.0

        # --- p95 latency ------------------------------------------------------
        # OFFSET is 0-indexed; we want the value at the 95th percentile position.
        p95_offset = max(0, int(call_count * 0.95))
        p95_row = self._conn.execute(
            f"""
            SELECT latency_ms
            FROM calls
            WHERE {where}
            ORDER BY latency_ms
            LIMIT 1 OFFSET ?
            """,
            params + [p95_offset],
        ).fetchone()
        p95_latency_ms = p95_row[0] if p95_row else avg_latency_ms

        return ModelStats(
            call_count=call_count,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            total_cost=total_cost,
            avg_cost=avg_cost,
            success_rate=success_rate,
            refusal_rate=refusal_rate,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
