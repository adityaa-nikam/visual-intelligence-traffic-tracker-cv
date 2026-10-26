"""
Minimal Markdown -> styled HTML converter, so the submission write-up can be
pasted into a Google Doc (or printed to PDF) with headings, tables and bold text
intact. Deliberately tiny: no dependency, and it only needs to handle the subset
of Markdown that WRITEUP.md / README.md / ISSUES.md actually use.

Usage:
    python tools/md_to_html.py WRITEUP.md -o WRITEUP.html
Then open the HTML in a browser and either
    * Ctrl+A, Ctrl+C -> paste into a Google Doc (formatting carries over), or
    * Ctrl+P -> "Save as PDF".
"""

from __future__ import annotations

import argparse
import html
import os
import re

CSS = """
body{max-width:860px;margin:36px auto;padding:0 28px;
 font-family:Georgia,'Times New Roman',serif;font-size:15px;line-height:1.62;color:#1a1a1a}
h1{font-size:30px;line-height:1.25;border-bottom:2px solid #222;padding-bottom:10px;margin:0 0 6px}
h2{font-size:22px;margin:34px 0 10px;border-bottom:1px solid #ccc;padding-bottom:5px}
h3{font-size:17px;margin:24px 0 8px}
p{margin:10px 0}
code{font-family:Consolas,'Courier New',monospace;font-size:13px;
 background:#f2f2f2;padding:1px 4px;border-radius:3px}
pre{background:#f7f7f7;border:1px solid #ddd;border-left:3px solid #666;
 padding:12px 14px;overflow-x:auto;border-radius:3px}
pre code{background:none;padding:0;font-size:12.5px;line-height:1.45}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;
 font-family:Arial,Helvetica,sans-serif}
th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#ececec;font-weight:bold}
tr:nth-child(even) td{background:#fafafa}
hr{border:none;border-top:1px solid #ccc;margin:30px 0}
ul,ol{margin:10px 0;padding-left:26px}
li{margin:5px 0}
em{color:#333}
"""


def inline(s: str) -> str:
    """Escape, then re-apply inline markup. Code spans are protected first."""
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{html.escape(spans[int(m.group(1))])}</code>", s)


def row_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]

        if ln.startswith("```"):                        # fenced code
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # table: header row followed by a |---| separator
        if ln.strip().startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            head = row_cells(ln)
            i += 2
            body = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(row_cells(lines[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", ln):
            out.append("<hr>")
            i += 1
            continue

        # list block (bulleted or numbered), allowing wrapped continuation lines
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = m.group(2)[0].isdigit()
            items: list[str] = []
            while i < n:
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if mm:
                    items.append(mm.group(3))
                    i += 1
                elif lines[i].strip() and lines[i].startswith(("  ", "\t")) and items:
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        if not ln.strip():
            i += 1
            continue

        # paragraph: join until a blank line or a block-level start
        buf = [ln]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|```|\||\s*([-*]|\d+\.)\s|-{3,}\s*$)", lines[i]):
            buf.append(lines[i])
            i += 1
        out.append(f"<p>{inline(' '.join(x.strip() for x in buf))}</p>")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    md = open(a.src, encoding="utf-8").read()
    out = a.out or os.path.splitext(a.src)[0] + ".html"
    title = a.title or os.path.splitext(os.path.basename(a.src))[0]
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body>{convert(md)}</body></html>")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"-> {out}  ({len(doc)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
