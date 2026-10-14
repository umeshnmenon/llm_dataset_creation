"""SQLite commit is the optimizer-step commit boundary, including model/data state.

The append-only event journal is hash chained. Checkpoint JSONs are materialised
views of committed SQLite states, so a crash before file export cannot lose a
committed optimizer update. No committed history is truncated on recovery.
"""
import json
import sqlite3
from pathlib import Path
from .util import atomic_json, canonical, digest


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "ledgers").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "ledgers" / "ledger.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, branch TEXT NOT NULL,
                step INTEGER NOT NULL, payload TEXT NOT NULL, prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE);
            CREATE TABLE IF NOT EXISTS checkpoints (
                branch TEXT NOT NULL, step INTEGER NOT NULL, offset INTEGER NOT NULL,
                state TEXT NOT NULL, state_hash TEXT NOT NULL,
                PRIMARY KEY(branch, step));
            CREATE TABLE IF NOT EXISTS branches (
                branch TEXT PRIMARY KEY, parent TEXT, parent_step INTEGER,
                stream_mode TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT, 'append-only event ledger'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT, 'append-only event ledger'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_checkpoints_update BEFORE UPDATE ON checkpoints
              BEGIN SELECT RAISE(ABORT, 'immutable checkpoint'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_checkpoints_delete BEFORE DELETE ON checkpoints
              BEGIN SELECT RAISE(ABORT, 'immutable checkpoint'); END;
        """)

    def append(self, kind, branch, step, payload):
        row = self.db.execute("SELECT event_hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
        previous = row[0] if row else "0" * 64
        identity = {"kind": kind, "branch": branch, "step": step, "payload": payload, "prev_hash": previous}
        self.db.execute("INSERT INTO events(kind,branch,step,payload,prev_hash,event_hash) VALUES(?,?,?,?,?,?)",
                        (kind, branch, step, canonical(payload), previous, digest(identity)))

    def save_state(self, branch, step, state):
        offset = self.db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
        state = {**state, "branch": branch, "step": step, "ledger_offset": offset}
        self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?)", (branch, step, offset, canonical(state), digest(state)))
        return state

    def get_state(self, branch, step=None):
        query = "SELECT state,state_hash FROM checkpoints WHERE branch=?"
        args = [branch]
        if step is not None:
            query += " AND step=?"; args.append(step)
        query += " ORDER BY step DESC LIMIT 1"
        row = self.db.execute(query, args).fetchone()
        if not row:
            raise ValueError("No checkpoint for requested branch/step")
        state = json.loads(row[0])
        if digest(state) != row[1]:
            raise ValueError("Checkpoint integrity failed")
        return state

    def materialize(self):
        for branch, step, offset, state, state_hash in self.db.execute("SELECT * FROM checkpoints ORDER BY branch,step"):
            atomic_json(self.root / "checkpoints" / f"{branch}-{step:04d}.json", {"state": json.loads(state), "hash": state_hash})
        kinds = [r[0] for r in self.db.execute("SELECT DISTINCT kind FROM events")]
        for kind in kinds:
            with (self.root / "ledgers" / f"{kind}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                for event in self.events(kind=kind):
                    f.write(canonical(event) + "\n")

    def events(self, kind=None, branch=None, step=None):
        query, args, filters = "SELECT id,kind,branch,step,payload,prev_hash,event_hash FROM events", [], []
        for key, value in (("kind", kind), ("branch", branch), ("step", step)):
            if value is not None:
                filters.append(key + "=?"); args.append(value)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY id"
        return [{"id": r[0], "kind": r[1], "branch": r[2], "step": r[3], "payload": json.loads(r[4]), "prev_hash": r[5], "event_hash": r[6]}
                for r in self.db.execute(query, args)]

    def verify(self):
        previous = "0" * 64
        for event in self.events():
            identity = {k: event[k] for k in ("kind", "branch", "step", "payload", "prev_hash")}
            if event["prev_hash"] != previous or digest(identity) != event["event_hash"]:
                raise ValueError("Ledger chain mismatch")
            previous = event["event_hash"]
        return True

    def close(self):
        self.db.close()
