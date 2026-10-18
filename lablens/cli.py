"""Command-line entry points. No camera or microphone needed for these.

  python -m lablens.cli ask "what is the pinout of the NE555?"
  python -m lablens.cli ingest data/datasheets
  python -m lablens.cli projects
  python -m lablens.cli report
"""
from __future__ import annotations

import argparse
import sys

from lablens.config import get_config


def cmd_ask(args) -> int:
    from lablens.core.assistant import LabLens

    lens = LabLens.from_config(get_config())
    s = lens.ask_stream(" ".join(args.question), pointed=None)
    for tok in s:
        sys.stdout.write(tok)
        sys.stdout.flush()
    print()
    a = s.answer
    print(f"\n[backend={a.backend} latency={a.latency_s:.2f}s cached={a.cached}]")
    for c in a.citations[:3]:
        print(f"  - {c.source} p.{c.page}")
    return 0


def cmd_ingest(args) -> int:
    from lablens.rag.store import build_store

    store = build_store(get_config())
    n = store.build_from_dir(args.dir)
    print(f"indexed {n} new chunks; total {len(store.chunks)} from {len(store.sources())} files")
    return 0


def cmd_projects(args) -> int:
    from lablens.guided.engine import list_projects

    for pid, p in list_projects(get_config().path("guided.projects_dir", "lablens/guided/projects")).items():
        print(f"{pid:20s} {len(p.steps)} steps  {p.title}")
    return 0


def cmd_report(args) -> int:
    from lablens.dashboard.events import EventLog, stuck_report
    from lablens.config import ROOT

    ev = EventLog(get_config().path("dashboard.db_path", "data/lablens_events.sqlite"))
    print(stuck_report(ev.dataframe()).to_string(index=False) or "no step events yet")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lablens")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask a text question (no camera)")
    a.add_argument("question", nargs="+")
    a.set_defaults(fn=cmd_ask)
    i = sub.add_parser("ingest", help="index a folder of datasheet PDFs")
    i.add_argument("dir")
    i.set_defaults(fn=cmd_ingest)
    sub.add_parser("projects", help="list guided build projects").set_defaults(fn=cmd_projects)
    sub.add_parser("report", help="instructor dashboard: where students get stuck").set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
