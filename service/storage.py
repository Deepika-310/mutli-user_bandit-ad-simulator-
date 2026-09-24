"""SQLite persistence for the bandit service.

Plain ``sqlite3`` with parameterised SQL (no ORM) so every query is visible. The
service is stateless: an agent is rebuilt from per-arm impression/click counts
computed in SQL on every request, so any number of API workers can share one DB.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS bandits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    algorithm  TEXT    NOT NULL,
    n_arms     INTEGER NOT NULL CHECK (n_arms BETWEEN 2 AND 50),
    params     TEXT    NOT NULL DEFAULT '{}',
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS impressions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bandit_id  INTEGER NOT NULL REFERENCES bandits(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,                         -- 1, 2, 3... within a bandit
    arm        INTEGER NOT NULL CHECK (arm >= 0),
    clicked    INTEGER NOT NULL DEFAULT 0 CHECK (clicked IN (0, 1)),
    created_at TEXT    NOT NULL,
    clicked_at TEXT,
    UNIQUE (bandit_id, seq)
);

-- serves the per-arm GROUP BY on every /select call
CREATE INDEX IF NOT EXISTS idx_impressions_bandit_arm ON impressions (bandit_id, arm);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    # isolation_level=None: autocommit, transactions are opened explicitly below
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL: readers don't block the writer and commits don't fsync the whole DB file
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(path):
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


class Store:
    """Data-access layer: one instance per request, wrapping one connection."""

    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE takes the write lock up front, so read-then-write is atomic."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ---- bandits -------------------------------------------------------------

    def create_bandit(self, name, algorithm, n_arms, params):
        cur = self.conn.execute(
            "INSERT INTO bandits (name, algorithm, n_arms, params, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, algorithm, n_arms, json.dumps(params), _now()))
        return self.get_bandit(cur.lastrowid)

    def get_bandit(self, bandit_id):
        row = self.conn.execute("SELECT * FROM bandits WHERE id = ?", (bandit_id,)).fetchone()
        if row is None:
            return None
        bandit = dict(row)
        bandit["params"] = json.loads(bandit["params"])
        return bandit

    def list_bandits(self):
        rows = self.conn.execute("SELECT id FROM bandits ORDER BY id").fetchall()
        return [self.get_bandit(r["id"]) for r in rows]

    # ---- impressions and clicks ---------------------------------------------

    def arm_stats(self, bandit_id, n_arms):
        """Per-arm (impressions, clicks) as two lists, computed with a GROUP BY."""
        rows = self.conn.execute(
            "SELECT arm, COUNT(*) AS pulls, SUM(clicked) AS clicks "
            "FROM impressions WHERE bandit_id = ? GROUP BY arm", (bandit_id,)).fetchall()
        pulls, clicks = [0] * n_arms, [0] * n_arms
        for r in rows:
            pulls[r["arm"]], clicks[r["arm"]] = r["pulls"], r["clicks"]
        return pulls, clicks

    def add_impression(self, bandit_id, arm):
        """Log that ``arm`` was shown. Call inside ``transaction()`` (seq is read-then-write)."""
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM impressions WHERE bandit_id = ?",
            (bandit_id,)).fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO impressions (bandit_id, seq, arm, created_at) VALUES (?, ?, ?, ?)",
            (bandit_id, seq, arm, _now()))
        return cur.lastrowid, seq

    def record_click(self, impression_id):
        """Mark an impression as clicked. Returns 'recorded', 'duplicate' or None if unknown."""
        with self.transaction():
            row = self.conn.execute(
                "SELECT clicked FROM impressions WHERE id = ?", (impression_id,)).fetchone()
            if row is None:
                return None
            if row["clicked"]:
                return "duplicate"
            self.conn.execute(
                "UPDATE impressions SET clicked = 1, clicked_at = ? WHERE id = ?",
                (_now(), impression_id))
            return "recorded"

    def ctr_trend(self, bandit_id, bucket):
        """CTR per block of ``bucket`` impressions plus the running cumulative CTR.

        One query: GROUP BY builds the blocks, then window functions running over the
        grouped rows turn them into cumulative totals.
        """
        rows = self.conn.execute(
            """
            WITH blocks AS (
                SELECT (seq - 1) / :bucket AS block,
                       COUNT(*)            AS impressions,
                       SUM(clicked)        AS clicks
                FROM impressions
                WHERE bandit_id = :bandit_id
                GROUP BY block
            )
            SELECT block, impressions, clicks,
                   1.0 * clicks / impressions                              AS ctr,
                   1.0 * SUM(clicks)      OVER (ORDER BY block)
                       / SUM(impressions) OVER (ORDER BY block)            AS cumulative_ctr
            FROM blocks
            ORDER BY block
            """, {"bandit_id": bandit_id, "bucket": bucket}).fetchall()
        return [dict(r) for r in rows]
