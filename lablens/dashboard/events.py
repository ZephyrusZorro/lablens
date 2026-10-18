"""Event log for the instructor dashboard: questions, step attempts, failures, latency.

One SQLite file per lab; each bench sets LABLENS_BENCH (e.g. "bench-07").
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class EventLog:
    def __init__(self, path: Path, bench: Optional[str] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.bench = bench or os.getenv("LABLENS_BENCH", "bench-01")
        self.session = time.strftime("%Y%m%d-%H%M%S")
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS events (ts REAL, bench TEXT, session TEXT, kind TEXT,"
                         " project TEXT, step INTEGER, component TEXT, payload TEXT)")
        self._db.commit()

    def log(self, kind: str, project: str = "", step: int = -1, component: str = "", **payload: Any) -> None:
        with self._lock:
            self._db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                             (time.time(), self.bench, self.session, kind, project, step, component,
                              json.dumps(payload, ensure_ascii=False)))
            self._db.commit()

    def query(self, sql: str, params=()):
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def dataframe(self):
        import pandas as pd

        df = pd.read_sql_query("SELECT * FROM events", sqlite3.connect(str(self.path)))
        if not df.empty:
            df["time"] = pd.to_datetime(df["ts"], unit="s")
            df["payload"] = df["payload"].apply(lambda s: json.loads(s) if s else {})
        return df


def stuck_report(df) -> "Any":
    """Per (project, step): attempts, failures, mean seconds spent, top missing parts."""
    import pandas as pd

    if df.empty:
        return pd.DataFrame()
    v = df[df.kind.isin(["step_check", "step_done"])].copy()
    if v.empty:
        return pd.DataFrame()
    v["ok"] = v.payload.apply(lambda p: bool(p.get("ok")))
    v["missing"] = v.payload.apply(lambda p: ", ".join(p.get("missing", [])))
    v["secs"] = v.payload.apply(lambda p: p.get("seconds_on_step", 0.0))
    g = v.groupby(["project", "step"])
    out = pd.DataFrame({
        "checks": g.size(),
        "failed_checks": g.ok.apply(lambda s: int((~s).sum())),
        "benches": g.bench.nunique(),
        "avg_secs_on_step": g.secs.mean().round(1),
        "most_missing": g.missing.agg(lambda s: s[s != ""].mode().iat[0] if (s != "").any() else ""),
    }).reset_index()
    out["fail_rate"] = (out.failed_checks / out.checks).round(2)
    return out.sort_values(["fail_rate", "checks"], ascending=False)
