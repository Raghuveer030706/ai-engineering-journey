# phase3-agents/day11-memory/memory.py
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime


DB_PATH = Path(__file__).parent / "memory.db"


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key         TEXT NOT NULL UNIQUE,
            value       TEXT NOT NULL,
            confidence  REAL NOT NULL DEFAULT 1.0,
            source      TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


# ── Short-term memory (in-session) ────────────────────────────────────────────

class ShortTermMemory:
    """
    Dictionary-backed memory for the current session only.
    Cleared when the process exits.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def store(self, key: str, value: str, confidence: float = 1.0, source: str = "agent"):
        self._store[key] = {
            "value": value,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time(),
        }

    def retrieve(self, key: str) -> dict | None:
        return self._store.get(key)

    def retrieve_all(self) -> dict:
        return dict(self._store)

    def forget(self, key: str):
        self._store.pop(key, None)

    def summary(self) -> str:
        """Returns a compact string injected into the agent system prompt."""
        if not self._store:
            return "No short-term memories."
        lines = []
        for k, v in self._store.items():
            conf_tag = f"[conf={v['confidence']:.2f}]"
            lines.append(f"  • {k}: {v['value']} {conf_tag}")
        return "Short-term memory:\n" + "\n".join(lines)


# ── Long-term memory (SQLite) ─────────────────────────────────────────────────

class LongTermMemory:
    """
    SQLite-backed memory. Survives across sessions.
    Keys are unique; re-storing a key updates value + confidence.
    """

    def __init__(self):
        init_db()

    def store(self, key: str, value: str, confidence: float = 1.0, source: str = "agent"):
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO long_term_memory (key, value, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                confidence = excluded.confidence,
                source     = excluded.source,
                updated_at = excluded.updated_at
        """, (key, value, confidence, source, now, now))
        conn.commit()
        conn.close()

    def retrieve(self, key: str) -> dict | None:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE long_term_memory SET access_count = access_count + 1 WHERE key = ?", (key,)
        )
        conn.commit()
        row = conn.execute(
            "SELECT key, value, confidence, source, updated_at, access_count "
            "FROM long_term_memory WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "key": row[0], "value": row[1], "confidence": row[2],
            "source": row[3], "updated_at": row[4], "access_count": row[5],
        }

    def retrieve_all(self) -> list[dict]:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT key, value, confidence, source, updated_at, access_count "
            "FROM long_term_memory ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [
            {"key": r[0], "value": r[1], "confidence": r[2],
             "source": r[3], "updated_at": r[4], "access_count": r[5]}
            for r in rows
        ]

    def low_confidence(self, threshold: float = 0.6) -> list[dict]:
        """Returns entries below confidence threshold — candidates for re-verification."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT key, value, confidence FROM long_term_memory WHERE confidence < ?",
            (threshold,)
        ).fetchall()
        conn.close()
        return [{"key": r[0], "value": r[1], "confidence": r[2]} for r in rows]

    def degrade_confidence(self, key: str, amount: float = 0.1):
        """Lower confidence on a key (call when RAG contradicts a memory)."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE long_term_memory SET confidence = MAX(0.0, confidence - ?) WHERE key = ?",
            (amount, key)
        )
        conn.commit()
        conn.close()

    def forget(self, key: str):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM long_term_memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()

    def summary(self, threshold: float = 0.6) -> str:
        """Compact string for system prompt injection. Flags low-confidence entries."""
        rows = self.retrieve_all()
        if not rows:
            return "No long-term memories."
        lines = []
        for r in rows:
            flag = " ⚠ LOW CONF" if r["confidence"] < threshold else ""
            lines.append(f"  • {r['key']}: {r['value']} [conf={r['confidence']:.2f}]{flag}")
        return "Long-term memory:\n" + "\n".join(lines)


# ── Unified facade ────────────────────────────────────────────────────────────

class Memory:
    """
    Single interface the agent uses.
    store() writes to both tiers.
    retrieve() checks short-term first, falls back to long-term.
    """

    def __init__(self):
        self.short = ShortTermMemory()
        self.long = LongTermMemory()

    def store(self, key: str, value: str, confidence: float = 1.0,
              source: str = "agent", persist: bool = True):
        """
        persist=True  → write to both tiers (default)
        persist=False → short-term only (ephemeral scratchpad)
        """
        self.short.store(key, value, confidence, source)
        if persist:
            self.long.store(key, value, confidence, source)

    def retrieve(self, key: str) -> dict | None:
        result = self.short.retrieve(key)
        if result:
            result["tier"] = "short"
            return result
        result = self.long.retrieve(key)
        if result:
            result["tier"] = "long"
        return result

    def degrade(self, key: str, amount: float = 0.1):
        self.long.degrade_confidence(key, amount)
        entry = self.short.retrieve(key)
        if entry:
            new_conf = max(0.0, entry["confidence"] - amount)
            entry["confidence"] = new_conf

    def context_block(self) -> str:
        """Full memory context injected into agent system prompt."""
        return self.short.summary() + "\n\n" + self.long.summary()