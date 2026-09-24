"""src/services/persistence.py

aiosqlite-backed implementations of the LedgerWriter and CacheBackend
Protocols (defined in evaluator.py). Traditional sqlite3 is synchronous and
blocks the event loop on every disk I/O; aiosqlite runs each connection's
operations on a dedicated background thread and awaits the result, keeping
FastAPI's event loop free during writes.

Each class owns exactly one aiosqlite.Connection. aiosqlite serializes all
operations on a connection through that connection's single worker thread,
so concurrent awaits against the same instance are already safe without an
additional asyncio.Lock. WAL mode is enabled on init so ledger writes do not
block concurrent cache reads/writes on a separate connection.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import aiosqlite

from src.models.taxonomy import EvaluationResult

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS compliance_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    locale TEXT NOT NULL,
    business_description TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence REAL NOT NULL,
    risk_level TEXT NOT NULL,
    violated_rule_ids TEXT NOT NULL,
    raw_result TEXT NOT NULL,
    evaluated_at TEXT NOT NULL
)
"""

_LEDGER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ledger_policy_locale
    ON compliance_ledger (policy_id, locale)
"""

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_store (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""


class SQLiteLedgerWriter:
    """Append-only audit log. No UNIQUE constraint on cache_key: re-evaluating
    the same input after cache expiry produces a new row, not an overwrite --
    this is a ledger, not a deduplicated table."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, db_path: str) -> "SQLiteLedgerWriter":
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_LEDGER_SCHEMA)
        await conn.execute(_LEDGER_INDEX)
        await conn.commit()
        return cls(conn)

    async def record(self, *, cache_key: str, business_description: str, result: EvaluationResult) -> None:
        await self._conn.execute(
            """
            INSERT INTO compliance_ledger
                (cache_key, policy_id, locale, business_description, classification,
                 confidence, risk_level, violated_rule_ids, raw_result, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                result.policy_id,
                result.locale.value,
                business_description,
                result.classification.value,
                result.confidence,
                result.risk_level.value,
                json.dumps(result.violated_rule_ids),
                result.model_dump_json(),
                result.evaluated_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def close(self) -> None:
        await self._conn.close()


class SQLiteCacheBackend:
    """Redis-shaped get/set-with-TTL over SQLite. expires_at is stored as an
    ISO-8601 UTC string and compared directly in SQL so an expired row is
    never read back, let alone deserialized."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, db_path: str) -> "SQLiteCacheBackend":
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_CACHE_SCHEMA)
        await conn.commit()
        return cls(conn)

    async def get(self, key: str) -> EvaluationResult | None:
        now = datetime.utcnow().isoformat()
        cursor = await self._conn.execute(
            "SELECT value FROM cache_store WHERE key = ? AND expires_at > ?", (key, now)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return EvaluationResult.model_validate_json(row[0])

    async def set(self, key: str, value: EvaluationResult, ttl_seconds: int) -> None:
        expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
        await self._conn.execute(
            """
            INSERT INTO cache_store (key, value, expires_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at
            """,
            (key, value.model_dump_json(), expires_at),
        )
        await self._conn.commit()

    async def purge_expired(self) -> int:
        """Not invoked automatically -- expired rows are already invisible to
        get(). Call this periodically (e.g. a scheduled task) to reclaim disk
        space; running it on every set() would add a DELETE scan to every
        write for no correctness benefit."""
        now = datetime.utcnow().isoformat()
        cursor = await self._conn.execute("DELETE FROM cache_store WHERE expires_at <= ?", (now,))
        await self._conn.commit()
        deleted = cursor.rowcount
        await cursor.close()
        return deleted

    async def close(self) -> None:
        await self._conn.close()