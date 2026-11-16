"""A small, dependency-free MCP server (stdio) used as the upstream in demos and experiments.

Tools (all confined to --root):
  list_files(dir)          list files under a directory
  read_file(path)          read a text file
  grep(pattern, path)      regex search, returns file:line: text
  sql_query(sql)           read-only SQL over every CSV under root (one table per CSV file)
  fetch_url(url)           HTTP GET a web page (text, tags stripped)

Usage: python -m epistaint.toolbox --root demo/corpus
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
MAX_CHARS = 60_000

TOOLS = [
    {
        "name": "list_files",
        "description": "List files under a directory of the research corpus.",
        "inputSchema": {"type": "object", "properties": {"dir": {"type": "string", "default": "."}}},
    },
    {
        "name": "read_file",
        "description": "Read a text file from the research corpus.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "grep",
        "description": "Search the corpus with a regular expression (case-insensitive). Returns file:line: text.",
        "inputSchema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}},
            "required": ["pattern"],
        },
    },
    {
        "name": "sql_query",
        "description": "Run a read-only SQLite query over the corpus CSV files. Each CSV is a table named "
        "after its file stem (e.g. runs.csv -> runs). Use this for any counting, averaging or other "
        "arithmetic instead of computing by hand.",
        "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    },
    {
        "name": "fetch_url",
        "description": "Fetch a web page over HTTP(S) and return its text.",
        "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
]


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.db = sqlite3.connect(":memory:")
        for csv_path in sorted(self.root.rglob("*.csv")):
            self._load_csv(csv_path)
        self.db.execute("PRAGMA query_only = ON")

    def _load_csv(self, path: Path) -> None:
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header, body = rows[0], rows[1:]

        def typed(values: list[str]) -> str:
            for kind, conv in (("INTEGER", int), ("REAL", float)):
                try:
                    for v in values:
                        if v != "":
                            conv(v)
                    return kind
                except ValueError:
                    continue
            return "TEXT"

        cols = [f'"{h}" {typed([r[i] for r in body if i < len(r)])}' for i, h in enumerate(header)]
        table = re.sub(r"\W", "_", path.stem)
        self.db.execute(f'CREATE TABLE "{table}" ({", ".join(cols)})')
        self.db.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(header))})',
                            [[None if v == "" else v for v in r] for r in body])

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path outside the corpus: {rel}")
        if not p.exists():
            raise ToolError(f"no such file or directory: {rel}")
        return p

    def list_files(self, dir: str = ".") -> str:
        base = self._path(dir)
        files = [str(p.relative_to(self.root)) for p in sorted(base.rglob("*")) if p.is_file()]
        return "\n".join(files) or "(empty)"

    def read_file(self, path: str) -> str:
        p = self._path(path)
        if p.is_dir():
            raise ToolError(f"{path} is a directory")
        return p.read_text(errors="replace")[:MAX_CHARS]

    def grep(self, pattern: str, path: str = ".") -> str:
        rx = re.compile(pattern, re.I)
        base = self._path(path)
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        hits = []
        for f in files:
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.root)}:{n}: {line}")
        return "\n".join(hits[:500]) or "(no matches)"

    def sql_query(self, sql: str) -> str:
        try:
            cur = self.db.execute(sql)
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        names = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(1000)
        lines = ["\t".join(names)] + ["\t".join("" if v is None else str(v) for v in r) for r in rows]
        return "\n".join(lines)

    def fetch_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ToolError("only http(s) URLs are supported")
        req = urllib.request.Request(url, headers={"User-Agent": "epistaint-toolbox/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read(2_000_000).decode(resp.headers.get_content_charset() or "utf-8", "replace")
        body = re.sub(r"(?is)<(script|style).*?</\1>", " ", body)
        text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
        return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()[:MAX_CHARS]

    def call(self, name: str, arguments: dict) -> str:
        if name not in {t["name"] for t in TOOLS}:
            raise ToolError(f"unknown tool {name}")
        return getattr(self, name)(**arguments)


def serve(box: Toolbox) -> None:
    def send(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict) or "id" not in msg:
            continue  # notifications (initialized, cancelled, ...) need no reply
        mid, method, params = msg["id"], msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "epistaint-toolbox", "version": "0.1.0"},
            }})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            try:
                text, is_error = box.call(params.get("name", ""), params.get("arguments") or {}), False
            except (ToolError, OSError, TypeError, re.error) as e:
                text, is_error = f"error: {e}", True
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"content": [{"type": "text", "text": text}], "isError": is_error}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epistaint-toolbox")
    ap.add_argument("--root", required=True, help="corpus directory the tools may read")
    args = ap.parse_args(argv)
    serve(Toolbox(Path(args.root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
