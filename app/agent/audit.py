import json
import sqlite3
from datetime import datetime, timezone

AUDIT_DB = "data/audit.db"


def _init():
    con = sqlite3.connect(AUDIT_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS agent_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        user_id INTEGER,
        amount REAL,
        model_score REAL,
        decision TEXT,
        confidence REAL,
        used_agent INTEGER,
        tool_calls TEXT,
        reasoning TEXT,
        evidence TEXT
    )""")
    con.commit()
    con.close()


_init()


def log_decision(user_id, txn, result):
    con = sqlite3.connect(AUDIT_DB)
    con.execute(
        """INSERT INTO agent_decisions
           (created_at, user_id, amount, model_score, decision, confidence,
            used_agent, tool_calls, reasoning, evidence)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(), user_id, txn["amount"],
         result["model_score"], result["decision"], result["confidence"],
         int(result["used_agent"]), json.dumps(result["tool_calls"]),
         result["reasoning"], json.dumps(result["evidence"])),
    )
    con.commit()
    con.close()


def recent_decisions(limit=20):
    con = sqlite3.connect(AUDIT_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        d["tool_calls"] = json.loads(d["tool_calls"])
        d["evidence"] = json.loads(d["evidence"])
        d["used_agent"] = bool(d["used_agent"])
        out.append(d)
    return out