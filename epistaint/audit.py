"""Audit an agent's answer against the session's provenance log.

Each sentence (or bullet / table row) of the answer is a claim.  For every claim:

  * citations [[t-xxxxxxxx]] must name traces that exist in the session log
    (otherwise FABRICATED_CITATION);
  * every quoted span ("...", “...”, `...`) must appear verbatim in a cited output
    (otherwise MISQUOTE);
  * every number must appear in a cited output, allowing rounding to the precision the
    claim states (otherwise UNSUPPORTED_NUMBER);
  * a claim with no citation is UNCITED and inherits tier T3 (synthetic).  Its numbers and
    quotes are still looked up in the whole session, to tell "evidence exists but was not
    cited" (attributable) from "appears in no tool output at all" (unanchored).

A claim's taint is the least trustworthy tier among the evidence it relies on (taint
propagation), or T3 if it fails any check.  The log's hashes and hash chain are verified
too, so an edited log is reported.

Exit status: 0 clean, 1 uncited claims with --strict, 2 fabricated/misquoted/unsupported
claims or a broken log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from .core import Store, Tier, Trace, tier_label, verify_chain

CITE_RE = re.compile(r"\[\[\s*(t-[0-9a-fA-F]{4,16})\s*\]\]|\[(?:trace|t)\s*[:=]\s*(t-[0-9a-fA-F]{4,16})\]")
CITE_ONLY_RE = re.compile(r"(?:\s*(?:\[\[\s*t-[0-9a-fA-F]{4,16}\s*\]\]|\[(?:trace|t)\s*[:=]\s*t-[0-9a-fA-F]{4,16}\])[\s.,;]*)+")
# Quotations to verify (backticks mark code and paths, not quotations).
QUOTE_RE = re.compile(r'"([^"\n]{3,}?)"|“([^”\n]{3,}?)”')
# Numbers; a comma only counts as a thousands separator before exactly three digits, so CSV
# text such as "1022,0.852" is read as two numbers.
NUM_RE = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
PLAIN_NUM_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?![\w])")
SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[*_`])"  # full stop, next sentence starts with a capital etc.
    r"|(?<=[.!?][\"”’)])\s+"               # full stop inside a closing quote or bracket
    r"|(?<=\]\])[.,;]?\s+(?=[^\s\[])"      # a citation ends the sentence it supports
)
LIST_MARK = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")

# Statuses, worst first.
FABRICATED = "FABRICATED_CITATION"
MISQUOTE = "MISQUOTE"
UNSUPPORTED = "UNSUPPORTED_NUMBER"
UNCITED = "UNCITED"
GROUNDED = "GROUNDED"
SEVERITY = [FABRICATED, MISQUOTE, UNSUPPORTED, UNCITED, GROUNDED]


@dataclass
class Claim:
    index: int
    text: str
    citations: list[str] = field(default_factory=list)
    status: str = GROUNDED
    tier: Tier = Tier.T3  # taint: least trustworthy evidence tier (T3 when ungrounded)
    evidence_tier: Tier | None = None
    flags: list[str] = field(default_factory=list)
    checked: int = 0  # number of quotes + numbers verified against evidence
    attributable: bool | None = None  # for UNCITED claims: does any session output support it?

    def to_json(self) -> dict:
        return {
            "index": self.index, "text": self.text, "citations": self.citations,
            "status": self.status, "tier": tier_label(self.tier),
            "evidence_tier": tier_label(self.evidence_tier) if self.evidence_tier is not None else None,
            "flags": self.flags, "checked": self.checked, "attributable": self.attributable,
        }


# ------------------------------------------------------------------ parsing


def normalise(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip().casefold()


def strip_citations(text: str) -> str:
    return CITE_RE.sub("", text)


LEADING_CITES = re.compile(r"^((?:\s*(?:\[\[\s*t-[0-9a-fA-F]{4,16}\s*\]\]|\[(?:trace|t)\s*[:=]\s*t-[0-9a-fA-F]{4,16}\]))+)\s*(.+)$")


@dataclass
class Segment:
    text: str
    paragraph: int  # sentences of one line/paragraph share this number
    scope: list[str] = field(default_factory=list)  # citations of a lead-in line ("Results: [[t-..]]")


def segment(answer: str) -> list[Segment]:
    """Split an answer into candidate claims, keeping track of paragraphs and lead-in citations."""
    segs: list[Segment] = []
    in_code = False
    scope: list[str] = []  # a cited lead-in line covers the list or table below it
    table_start: int | None = None
    for para, raw in enumerate(answer.splitlines()):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line:
            table_start = None
            continue
        if line.startswith("#"):
            scope, table_start = [], None
            continue
        is_item = bool(LIST_MARK.match(line))
        line = LIST_MARK.sub("", re.sub(r"^>\s*", "", line))
        if line.startswith("|"):
            if set(line) <= set("|-: "):  # separator: the row above it was the header
                if table_start is not None and len(segs) == table_start + 1:
                    segs.pop()
                continue
            if table_start is None:
                table_start = len(segs)
            segs.append(Segment(line, para, list(scope)))
            continue
        table_start = None
        if not is_item:
            scope = []
        for part in SENT_SPLIT.split(line):
            part = part.strip()
            if not part:
                continue
            # a citation written after the full stop belongs to the sentence before it
            if segs and CITE_ONLY_RE.fullmatch(part):
                segs[-1].text += " " + part
                continue
            m = LEADING_CITES.match(part)
            if segs and m is not None:
                segs[-1].text += " " + m.group(1).strip()
                part = m.group(2).strip()
            segs.append(Segment(part, para, list(scope) if is_item else []))
        last = segs[-1] if segs else None
        if last and last.paragraph == para and not is_claim(last.text) and citations_in(last.text):
            scope = citations_in(last.text)  # lead-in like "Test-split results: [[t-...]]"
    return [s for s in segs if is_claim(s.text)]


def split_claims(answer: str) -> list[str]:
    return [s.text for s in segment(answer)]


def is_claim(text: str) -> bool:
    body = strip_citations(text).strip().strip("*_ ")
    if not re.search(r"[A-Za-z0-9]", body):
        return False
    if body.endswith(":"):
        return False  # lead-in to a list, not an assertion
    return len(body.split()) >= 3 or bool(NUM_RE.search(body))


def citations_in(text: str) -> list[str]:
    return [(a or b).lower() for a, b in CITE_RE.findall(text)]


def quotes_in(text: str) -> list[str]:
    """Quotations of two or more words (a single quoted word is a term, not a quotation)."""
    out = []
    for groups in QUOTE_RE.findall(strip_citations(text)):
        q = next(g for g in groups if g).strip(" .,;:")
        if len(q.split()) >= 2:
            out.append(q)
    return out


def quote_found(quote: str, evidence_norm: str) -> bool:
    """Verbatim match, allowing editorial insertions ("[is]") and ellipses between fragments."""
    frags = [normalise(f).strip(" .,;:") for f in re.split(r"\[[^\]]*\]|\.\.\.|…", quote)]
    pos = 0
    for frag in (f for f in frags if f):
        i = evidence_norm.find(frag, pos)
        if i < 0:
            return False
        pos = i + len(frag)
    return True


def _dec(token: str) -> Decimal | None:
    token = token.replace(",", "").replace("−", "-")
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def numbers_in(text: str) -> list[tuple[str, bool]]:
    """(number, is_percentage) pairs; numbers inside quotations are checked as quotes instead."""
    body = QUOTE_RE.sub(" ", strip_citations(text))
    return [(m.group(0), bool(re.match(r"\s*(%|per ?cent)", body[m.end():]))) for m in NUM_RE.finditer(body)]


def number_supported(claimed: str, evidence_numbers: list[Decimal], percent: bool = False) -> bool:
    """True if some evidence number equals the claimed one at the claim's stated precision
    (rounding half-up or half-even); a percentage may also match a fraction (40% ~ 0.4043)."""
    c = _dec(claimed)
    if c is None or not c.is_finite():
        return True
    exponent = c.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    quantum = Decimal(1).scaleb(-places)
    candidates = list(evidence_numbers) + ([e * 100 for e in evidence_numbers] if percent else [])
    for e in candidates:
        if abs(e) == abs(c):
            return True
        for mode in (ROUND_HALF_EVEN, ROUND_HALF_UP):
            try:
                if abs(e).quantize(quantum, rounding=mode) == abs(c):
                    return True
            except InvalidOperation:
                break
    return False


def evidence_numbers(text: str) -> list[Decimal]:
    tokens = {m.group(0) for rx in (NUM_RE, PLAIN_NUM_RE) for m in rx.finditer(text)}
    return [d for d in (_dec(t) for t in tokens) if d is not None]


# ------------------------------------------------------------------- audit


@dataclass
class Report:
    session: str
    claims: list[Claim]
    integrity: list[str]

    @property
    def summary(self) -> dict:
        counts = {s: 0 for s in SEVERITY}
        tiers = {tier_label(t): 0 for t in Tier}
        for c in self.claims:
            counts[c.status] += 1
            tiers[tier_label(c.tier)] += 1
        n = len(self.claims) or 1
        uncited = [c for c in self.claims if c.status == UNCITED]
        return {
            "claims": len(self.claims),
            "status": counts,
            "taint": tiers,
            "grounded_rate": round(counts[GROUNDED] / n, 3),
            "violations": counts[FABRICATED] + counts[MISQUOTE] + counts[UNSUPPORTED],
            "uncited_attributable": sum(1 for c in uncited if c.attributable),
            "uncited_unanchored_numbers": sum(1 for c in uncited if "unanchored-number" in " ".join(c.flags)),
            # comparable across arms, citations or not: claims with a number or quote that
            # appears in no tool output of the session at all
            "unanchored_claims": sum(1 for c in self.claims
                                     if any("unanchored" in f or "in no trace" in f for f in c.flags)),
            "checkable_claims": sum(1 for c in self.claims
                                    if c.checked or any("number" in f or "quote" in f for f in c.flags)),
            "log_intact": not self.integrity,
        }

    def to_json(self) -> dict:
        return {"session": self.session, "summary": self.summary,
                "integrity": self.integrity, "claims": [c.to_json() for c in self.claims]}


def audit(answer: str, traces: list[Trace], session: str = "") -> Report:
    by_id = {t.trace_id: t for t in traces}
    all_text = normalise("\n".join(t.output for t in traces))
    all_numbers = evidence_numbers("\n".join(t.output for t in traces))
    claims: list[Claim] = []
    segs = segment(answer)
    for i, seg in enumerate(segs, 1):
        text = seg.text
        c = Claim(index=i, text=text, citations=citations_in(text))
        if not c.citations and seg.scope:
            c.citations = list(seg.scope)
            c.flags.append("citation inherited from the lead-in line")
        elif not c.citations:
            later = next((s for s in segs[i:] if s.paragraph == seg.paragraph and citations_in(s.text)), None)
            if later is not None:
                c.citations = citations_in(later.text)
                c.flags.append("citation inherited from the end of the paragraph")
        quotes, numbers = quotes_in(text), numbers_in(text)
        statuses: set[str] = set()
        if not c.citations:
            statuses.add(UNCITED)
            missing_q = [q for q in quotes if not quote_found(q, all_text)]
            missing_n = [n for n, pct in numbers if not number_supported(n, all_numbers, pct)]
            c.flags += [f"unanchored-quote: {q!r}" for q in missing_q]
            c.flags += [f"unanchored-number: {n}" for n in missing_n]
            if quotes or numbers:
                c.attributable = not missing_q and not missing_n
        else:
            known = [by_id[t] for t in c.citations if t in by_id]
            for t in c.citations:
                if t not in by_id:
                    statuses.add(FABRICATED)
                    c.flags.append(f"cites {t}: no such trace in session {session or '?'}")
            if known:
                c.evidence_tier = max(t.tier for t in known)
                cited_text = normalise("\n".join(t.output for t in known))
                cited_numbers = evidence_numbers("\n".join(t.output for t in known))
                for q in quotes:
                    if quote_found(q, cited_text):
                        c.checked += 1
                    else:
                        statuses.add(MISQUOTE)
                        where = "in another trace" if quote_found(q, all_text) else "in no trace"
                        c.flags.append(f"quote not in cited output ({where}): {q!r}")
                for n, pct in numbers:
                    if number_supported(n, cited_numbers, pct):
                        c.checked += 1
                    else:
                        statuses.add(UNSUPPORTED)
                        where = "in another trace" if number_supported(n, all_numbers, pct) else "in no trace"
                        c.flags.append(f"number not in cited output ({where}): {n}")
                if any(t.is_error for t in known):
                    c.flags.append("cites a tool call that returned an error")
        c.status = next((s for s in SEVERITY if s in statuses), GROUNDED)
        c.tier = c.evidence_tier if c.status == GROUNDED and c.evidence_tier is not None else Tier.T3
        claims.append(c)
    return Report(session=session, claims=claims, integrity=verify_chain(traces))


# ------------------------------------------------------------------ render

COLOURS = {
    FABRICATED: "\033[1;35m", MISQUOTE: "\033[1;35m", UNSUPPORTED: "\033[1;35m",
    UNCITED: "\033[31m", "T0": "\033[32m", "T1": "\033[32m", "T2": "\033[33m",
}
RESET, DIM = "\033[0m", "\033[2m"


def render(report: Report, colour: bool) -> str:
    def paint(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if colour else s

    lines = [f"epistaint audit - session {report.session}"]
    for c in report.claims:
        if c.status == GROUNDED:
            key = c.tier.name
            tag = f"{c.tier.name} {'GROUNDED' if c.checked else 'CITED   '}"
        else:
            key = c.status
            tag = {FABRICATED: "FABRICATED", MISQUOTE: "MISQUOTE  ", UNSUPPORTED: "UNSUPPORTD",
                   UNCITED: "T3 UNCITED"}[c.status]
        lines.append(f"{paint(COLOURS[key], f'[{tag}]')} {c.index:>2}. {c.text}")
        for f in c.flags:
            lines.append(paint(DIM, f"      - {f}"))
    s = report.summary
    lines.append("")
    lines.append(
        f"{s['claims']} claims: {s['status'][GROUNDED]} grounded ({s['grounded_rate']:.0%}), "
        f"{s['status'][UNCITED]} uncited ({s['uncited_attributable']} attributable), "
        f"{s['violations']} violations "
        f"[{s['status'][FABRICATED]} fabricated citation, {s['status'][MISQUOTE]} misquote, "
        f"{s['status'][UNSUPPORTED]} unsupported number]"
    )
    lines.append("taint: " + ", ".join(f"{k} {v}" for k, v in s["taint"].items()))
    if report.integrity:
        lines.append(paint(COLOURS[FABRICATED], "LOG INTEGRITY FAILED:"))
        lines += [f"  - {p}" for p in report.integrity]
    else:
        lines.append("log integrity: ok (output hashes and hash chain verified)")
    return "\n".join(lines)


def feedback(report: Report) -> str:
    """A correction message an enforcing harness can send back to the agent."""
    bad = [c for c in report.claims if c.status != GROUNDED]
    if not bad:
        return ""
    out = ["Your answer failed the provenance audit. Fix these claims: re-run a tool if needed, "
           "cite the trace id printed with its output as [[t-xxxxxxxx]], quote and copy numbers "
           "exactly, or remove the claim. Reply with the complete corrected answer only, without "
           "commentary on what you changed."]
    for c in bad:
        out.append(f"- Claim {c.index} ({c.status}): {c.text}" + "".join(f"\n    {f}" for f in c.flags))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epistaint-audit", description="Audit an agent answer against its provenance log.")
    ap.add_argument("answer", nargs="?", default="-", help="answer file (default: stdin)")
    ap.add_argument("--db", default="traces.db")
    ap.add_argument("--session", help="session to audit against (default: the latest)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--feedback", action="store_true", help="print a correction message for the agent")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any claim is uncited")
    ap.add_argument("--no-colour", "--no-color", action="store_true")
    args = ap.parse_args(argv)

    answer = sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text()
    store = Store(args.db)
    session = args.session or store.latest_session()
    if session is None:
        print("no traces in the store", file=sys.stderr)
        return 2
    report = audit(answer, store.session_traces(session), session)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, ensure_ascii=False))
    elif args.feedback:
        print(feedback(report) or "OK")
    else:
        print(render(report, colour=not args.no_colour and sys.stdout.isatty()))
    s = report.summary
    if s["violations"] or report.integrity:
        return 2
    if args.strict and s["status"][UNCITED]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
