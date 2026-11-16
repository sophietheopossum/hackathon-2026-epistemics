"""Shared pieces: trust tiers, the tier policy, and the hash-chained trace store."""

from __future__ import annotations

import enum
import fnmatch
import hashlib
import json
import re
import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class Tier(enum.IntEnum):
    """Epistemic trust tiers. Lower is more trustworthy."""

    T0 = 0  # deterministic: local, reproducible execution (file read, SQL query, tests)
    T1 = 1  # attested: primary source on an allowlisted domain
    T2 = 2  # untrusted: arbitrary external content, other agents' output
    T3 = 3  # synthetic: model output with no tool evidence behind it


TIER_NAMES = {
    Tier.T0: "DETERMINISTIC",
    Tier.T1: "ATTESTED",
    Tier.T2: "UNTRUSTED",
    Tier.T3: "SYNTHETIC",
}


def tier_label(tier: Tier) -> str:
    return f"{tier.name}_{TIER_NAMES[tier]}"


def parse_tier(text: str) -> Tier:
    """Accept "T0", "t1", "T2_UNTRUSTED", ..."""
    return Tier[text.strip().upper()[:2]]


# --------------------------------------------------------------------- policy


@dataclass
class Rule:
    """One tier rule. First matching rule wins."""

    tool: str = "*"  # glob on the tool name
    tier: Tier = Tier.T2
    reason: str = ""
    arg: str | None = None  # name of the argument the conditions below look at
    match: str | None = None  # regex the argument must match
    domains: list[str] | None = None  # the argument is a URL whose host must be in this list

    def matches(self, tool: str, args: dict) -> bool:
        if not fnmatch.fnmatchcase(tool, self.tool):
            return False
        if self.arg is None:
            return True
        value = args.get(self.arg) if isinstance(args, dict) else None
        if value is None:
            return False
        value = str(value)
        if self.match and not re.search(self.match, value):
            return False
        if self.domains is not None:
            host = (urlparse(value).hostname or "").lower()
            if not any(host == d or host.endswith("." + d) for d in self.domains):
                return False
        return True


@dataclass
class Policy:
    rules: list[Rule]
    default: Tier = Tier.T2

    def classify(self, tool: str, args: dict) -> tuple[Tier, str]:
        for rule in self.rules:
            if rule.matches(tool, args):
                return rule.tier, rule.reason or f"rule for {rule.tool}"
        return self.default, "default tier (no rule matched)"

    @classmethod
    def load(cls, path: str | Path | None) -> "Policy":
        if path is None:
            return cls(rules=[])
        data = tomllib.loads(Path(path).read_text())
        rules = [
            Rule(
                tool=r.get("tool", "*"),
                tier=parse_tier(r["tier"]),
                reason=r.get("reason", ""),
                arg=r.get("arg"),
                match=r.get("match"),
                domains=[d.lower() for d in r["domains"]] if "domains" in r else None,
            )
            for r in data.get("rule", [])
        ]
        return cls(rules=rules, default=parse_tier(data.get("default_tier", "T2")))


# ---------------------------------------------------------------- trace store

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    session       TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    ts            REAL NOT NULL,
    tool          TEXT NOT NULL,
    arguments     TEXT NOT NULL,
    tier          INTEGER NOT NULL,
    tier_reason   TEXT NOT NULL,
    is_error      INTEGER NOT NULL,
    output        TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    chain_sha256  TEXT NOT NULL,
    latency_ms    REAL
);
CREATE INDEX IF NOT EXISTS traces_session ON traces(session, seq);
"""

GENESIS = "0" * 64


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_trace_id(session: str, seq: int, output_sha: str) -> str:
    return "t-" + sha256(f"{session}\x00{seq}\x00{output_sha}")[:8]


def chain_hash(prev: str, trace_id: str, tool: str, arguments: str, tier: int, output_sha: str) -> str:
    """Each record commits to the previous one, so the log is tamper-evident."""
    return sha256("\x00".join([prev, trace_id, tool, arguments, str(tier), output_sha]))


@dataclass
class Trace:
    trace_id: str
    session: str
    seq: int
    ts: float
    tool: str
    arguments: dict
    tier: Tier
    tier_reason: str
    is_error: bool
    output: str
    output_sha256: str
    chain_sha256: str
    latency_ms: float | None = None


def canonical_args(arguments: dict) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def append(
        self,
        session: str,
        ts: float,
        tool: str,
        arguments: dict,
        tier: Tier,
        tier_reason: str,
        is_error: bool,
        output: str,
        latency_ms: float | None = None,
    ) -> Trace:
        row = self.db.execute(
            "SELECT seq, chain_sha256 FROM traces WHERE session=? ORDER BY seq DESC LIMIT 1",
            (session,),
        ).fetchone()
        seq, prev = (row[0] + 1, row[1]) if row else (0, GENESIS)
        args_text = canonical_args(arguments)
        out_sha = sha256(output)
        trace_id = make_trace_id(session, seq, out_sha)
        chain = chain_hash(prev, trace_id, tool, args_text, int(tier), out_sha)
        self.db.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trace_id, session, seq, ts, tool, args_text, int(tier), tier_reason,
             int(is_error), output, out_sha, chain, latency_ms),
        )
        self.db.commit()
        return Trace(trace_id, session, seq, ts, tool, arguments, tier, tier_reason,
                     is_error, output, out_sha, chain, latency_ms)

    def session_traces(self, session: str) -> list[Trace]:
        rows = self.db.execute(
            "SELECT trace_id, session, seq, ts, tool, arguments, tier, tier_reason, is_error,"
            " output, output_sha256, chain_sha256, latency_ms"
            " FROM traces WHERE session=? ORDER BY seq",
            (session,),
        ).fetchall()
        return [
            Trace(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]), Tier(r[6]), r[7], bool(r[8]),
                  r[9], r[10], r[11], r[12])
            for r in rows
        ]

    def latest_session(self) -> str | None:
        row = self.db.execute("SELECT session FROM traces ORDER BY ts DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def sessions(self) -> list[tuple[str, int]]:
        return self.db.execute(
            "SELECT session, COUNT(*) FROM traces GROUP BY session ORDER BY MIN(ts)"
        ).fetchall()


def verify_chain(traces: list[Trace]) -> list[str]:
    """Recompute output hashes and the hash chain; return a list of problems (empty = intact)."""
    problems = []
    prev = GENESIS
    for t in traces:
        if sha256(t.output) != t.output_sha256:
            problems.append(f"{t.trace_id}: output does not match its sha256 (edited after logging)")
        expected = chain_hash(prev, t.trace_id, t.tool, canonical_args(t.arguments), int(t.tier),
                              t.output_sha256)
        if expected != t.chain_sha256:
            problems.append(f"{t.trace_id}: hash chain broken at seq {t.seq}")
        prev = t.chain_sha256
    return problems


# ------------------------------------------------------------ MCP result text


def result_text(result: dict) -> str:
    """Flatten an MCP tools/call result into the text the agent actually saw."""
    parts: list[str] = []
    for item in result.get("content") or []:
        kind = item.get("type")
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "resource":
            res = item.get("resource") or {}
            parts.append(res.get("text") or f"[resource {res.get('uri', '?')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link {item.get('uri', '?')}]")
        else:
            parts.append(f"[{kind} content]")
    if result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)
