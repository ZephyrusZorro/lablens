"""Render docs/*.md to standalone HTML (double-click to open in a browser on Windows).

    python scripts/build_docs.py
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CSS = """
:root { color-scheme: light dark; --fg:#1d232b; --bg:#fbfbfa; --muted:#5b6672; --line:#d9dde2; --accent:#0b6e4f;
        --code:#f0f2f4; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6e9ec; --bg:#16191d; --muted:#9aa4ae; --line:#2e343b;
        --accent:#5cc79d; --code:#22272d; } }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 2rem; margin: .2em 0 .6em; } h2 { margin-top: 2.2em; border-bottom: 1px solid var(--line);
     padding-bottom: .3em; } h3 { margin-top: 1.6em; }
a { color: var(--accent); } hr { border: 0; border-top: 1px solid var(--line); margin: 2em 0; }
code { background: var(--code); padding: .1em .35em; border-radius: 4px; font-size: .92em; }
pre { background: var(--code); padding: 12px 14px; border-radius: 8px; overflow-x: auto; }
pre code { padding: 0; background: none; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: var(--code); } blockquote { margin: 1em 0; padding: .2em 1em; border-left: 4px solid var(--accent);
     color: var(--muted); }
"""


def render(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(r"\]\(([\w/.-]+)\.md(#[\w-]*)?\)", lambda m: f"]({m.group(1)}.html{m.group(2) or ''})", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    out = md_path.with_suffix(".html")
    out.write_text(f"<!doctype html><html lang=en><head><meta charset=utf-8>"
                   f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{title}</title>"
                   f"<style>{CSS}</style></head><body><main>{body}</main></body></html>\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    for p in sorted((ROOT / "docs").glob("*.md")):
        print("wrote", render(p).relative_to(ROOT))
