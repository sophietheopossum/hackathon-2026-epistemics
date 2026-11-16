"""Experiments.

  inject   Error-injection benchmark (no LLM calls).  Takes every claim the auditor found
           GROUNDED in real agent answers (runs/*/answer*.md) and plants one error per copy:
             fabricate  replace each citation with a trace id that does not exist
             number     change one digit of one number
             quote      change one word inside one quotation
             strip      remove the citations
             swap       cite a different real trace from the same session (misattribution)
           and reports how often the auditor flags each kind, plus the false-positive rate on
           the unmodified claims.

  agents   Runs each question in a file through Claude Code under each arm (baseline,
           provenance, enforce) and tabulates the audit summaries.

    python -m epistaint.bench inject
    python -m epistaint.bench agents --questions demo/questions.txt --arms baseline,provenance,enforce
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audit import CITE_RE, FABRICATED, GROUNDED, MISQUOTE, NUM_RE, QUOTE_RE, UNCITED, UNSUPPORTED, audit
from .core import Store, Trace

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"fabricate": {FABRICATED}, "number": {UNSUPPORTED}, "quote": {MISQUOTE},
            "strip": {UNCITED}, "swap": {UNSUPPORTED, MISQUOTE}}


def _fake_id(seed: str, known: set[str]) -> str:
    n = 0
    while True:
        tid = "t-" + hashlib.sha256(f"{seed}:{n}".encode()).hexdigest()[:8]
        if tid not in known:
            return tid
        n += 1


def mutate(claim: str, kind: str, traces: list[Trace]) -> str | None:
    """One planted error of the given kind, or None if the claim offers nothing to mutate."""
    known = {t.trace_id for t in traces}
    cites = [(a or b).lower() for a, b in CITE_RE.findall(claim)]
    if kind == "fabricate":
        return CITE_RE.sub(lambda m: f"[[{_fake_id(claim + m.group(0), known)}]]", claim)
    if kind == "strip":
        return CITE_RE.sub("", claim).strip()
    if kind == "swap":
        others = [t.trace_id for t in traces if t.trace_id not in cites]
        if not others or not cites:
            return None
        other = others[int(hashlib.sha256(claim.encode()).hexdigest(), 16) % len(others)]
        return CITE_RE.sub(f"[[{other}]]", claim)
    body = CITE_RE.sub(lambda m: "\x00" * len(m.group(0)), claim)  # protect citations
    if kind == "number":
        unquoted = QUOTE_RE.sub(lambda q: "\x00" * len(q.group(0)), body)  # quotes are the quote test's job
        m = next((m for m in NUM_RE.finditer(unquoted)), None)
        if not m:
            return None
        num = m.group(0)
        idx = max(i for i, ch in enumerate(num) if ch.isdigit())
        new = num[:idx] + str((int(num[idx]) + 3) % 10) + num[idx + 1:]
        return claim[:m.start()] + new + claim[m.end():]
    if kind == "quote":
        m = next((m for m in QUOTE_RE.finditer(body) if len(next(g for g in m.groups() if g).split()) >= 2), None)
        if not m:
            return None
        start = next(m.start(g) for g in (1, 2) if m.group(g))
        q = claim[start:start + len(next(g for g in m.groups() if g))]
        words = list(re.finditer(r"[A-Za-z]{3,}", q))
        if not words:
            return None
        w = max(words, key=lambda x: len(x.group(0)))
        replacement = "never" if w.group(0).lower() != "never" else "always"
        return claim[:start + w.start()] + replacement + claim[start + w.end():]
    raise ValueError(kind)


def inject(runs_dir: Path, db: Path) -> dict:
    store = Store(db)
    rows = []
    for answer_file in sorted(runs_dir.glob("*/answer*.md")):
        session = answer_file.parent.name
        traces = store.session_traces(session)
        if not traces:
            continue
        report = audit(answer_file.read_text(), traces, session)
        for c in report.claims:
            if c.status != GROUNDED or not CITE_RE.search(c.text):
                continue  # only claims that carry their own (not inherited) citations
            clean = audit(c.text, traces, session).claims
            rows.append({"session": session, "kind": "clean", "claim": c.text,
                         "status": clean[0].status if clean else "-", "detected": bool(clean) and clean[0].status != GROUNDED,
                         "category_ok": None, "checkable": c.checked > 0})
            for kind in EXPECTED:
                mutated = mutate(c.text, kind, traces)
                if mutated is None or mutated == c.text:
                    continue
                res = audit(mutated, traces, session).claims
                status = res[0].status if res else "-"
                rows.append({"session": session, "kind": kind, "claim": mutated, "status": status,
                             "detected": status != GROUNDED, "category_ok": status in EXPECTED[kind],
                             "checkable": c.checked > 0})
    summary: dict[str, dict] = {}
    for kind in ["clean", *EXPECTED]:
        sel = [r for r in rows if r["kind"] == kind]
        det = sum(r["detected"] for r in sel)
        summary[kind] = {"n": len(sel), "flagged": det, "rate": round(det / len(sel), 3) if sel else None,
                         "right_category": sum(1 for r in sel if r["category_ok"]) if kind != "clean" else None}
        if kind == "swap":
            chk = [r for r in sel if r["checkable"]]
            summary["swap (checkable claims)"] = {
                "n": len(chk), "flagged": sum(r["detected"] for r in chk),
                "rate": round(sum(r["detected"] for r in chk) / len(chk), 3) if chk else None}
    return {"summary": summary, "rows": rows}


def summary_row(meta: dict, s: dict) -> dict:
    return {**meta, "claims": s["claims"], "grounded": s["status"][GROUNDED], "uncited": s["status"][UNCITED],
            "fabricated": s["status"][FABRICATED], "misquote": s["status"][MISQUOTE],
            "unsupported": s["status"][UNSUPPORTED], "unanchored_claims": s["unanchored_claims"],
            "checkable_claims": s["checkable_claims"], "grounded_rate": s["grounded_rate"],
            "t0": s["taint"]["T0_DETERMINISTIC"], "t1": s["taint"]["T1_ATTESTED"],
            "t2": s["taint"]["T2_UNTRUSTED"], "t3": s["taint"]["T3_SYNTHETIC"]}


def reaudit(csv_path: Path, runs_dir: Path, db: Path) -> list[dict]:
    """Re-score the final answers of an earlier `agents` sweep with the current auditor."""
    store = Store(db)
    rows = []
    for r in csv.DictReader(csv_path.open()):
        if r.get("error"):
            rows.append(r)
            continue
        d = runs_dir / r["session"]
        answer = d / "answer.revised.md" if (d / "answer.revised.md").exists() else d / "answer.md"
        report = audit(answer.read_text(), store.session_traces(r["session"]), r["session"])
        (d / "audit.json").write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False))
        meta = {k: r[k] for k in ("question", "q", "arm", "session", "seconds", "cost_usd", "revised")}
        rows.append(summary_row(meta, report.summary))
    return rows


def agents(questions: list[str], arms: list[str], reps: int, jobs: int, runs_dir: Path, db: Path,
           model: str | None) -> list[dict]:
    from .run import run  # imported lazily: only this part needs Claude Code

    tasks = [(q, arm, r) for r in range(reps) for q in questions for arm in arms]

    def one(task):
        q, arm, r = task
        tag = f"q{questions.index(q) + 1}-r{r + 1}"
        try:
            res = run(q, arm, runs_dir, db, ROOT / "demo" / "corpus", ROOT / "policy.toml", model, tag=tag)
            s = res["report"]["summary"]
            print(f"[done] {arm:10s} {tag}: {s['claims']} claims, grounded {s['grounded_rate']:.0%}, "
                  f"violations {s['violations']}, unanchored {s['unanchored_claims']}", file=sys.stderr)
            return summary_row({"question": q, "q": tag, "arm": arm, "session": res["session"],
                                "seconds": res["seconds"], "cost_usd": res["cost_usd"],
                                "revised": res["revised"]}, s)
        except Exception as e:  # keep the sweep going; record the failure
            print(f"[fail] {arm} {tag}: {e}", file=sys.stderr)
            return {"question": q, "q": tag, "arm": arm, "error": str(e)}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, tasks))


def table(rows: list[dict]) -> str:
    ok = [r for r in rows if "error" not in r]
    arms = sorted({r["arm"] for r in ok}, key=["baseline", "provenance", "enforce"].index)
    lines = ["| arm | runs | claims | grounded | uncited (T3) | fabricated | misquote | unsupported number | unanchored claims | T2-tainted |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        sel = [r for r in ok if r["arm"] == arm]
        tot = {k: sum(int(r[k]) for r in sel) for k in ["claims", "grounded", "uncited", "fabricated", "misquote",
                                                   "unsupported", "unanchored_claims", "t2"]}
        pct = lambda k: f"{tot[k]} ({tot[k] / tot['claims']:.0%})" if tot["claims"] else "0"  # noqa: E731
        lines.append(f"| {arm} | {len(sel)} | {tot['claims']} | {pct('grounded')} | {pct('uncited')} | "
                     f"{tot['fabricated']} | {tot['misquote']} | {tot['unsupported']} | {pct('unanchored_claims')} | {tot['t2']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epistaint-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inject")
    pa = sub.add_parser("agents")
    pr = sub.add_parser("reaudit")
    pr.add_argument("csv", type=Path, help="per-run CSV written by an earlier `agents` sweep")
    for p in (pi, pa, pr):
        p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
        p.add_argument("--db", type=Path, default=ROOT / "runs" / "traces.db")
    pa.add_argument("--questions", type=Path, default=ROOT / "demo" / "questions.txt")
    pa.add_argument("--arms", default="baseline,provenance,enforce")
    pa.add_argument("--reps", type=int, default=1)
    pa.add_argument("--jobs", type=int, default=3)
    pa.add_argument("--model")
    args = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.cmd == "inject":
        res = inject(args.runs_dir, args.db)
        out = args.runs_dir / f"inject-{stamp}.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("| planted error | n | flagged | detection rate | expected verdict |\n|---|---|---|---|---|")
        for kind, s in res["summary"].items():
            label = "none (clean claim; flagged = false positive)" if kind == "clean" else kind
            rate = "-" if s["rate"] is None else f"{s['rate']:.0%}"
            cat = "-" if s.get("right_category") is None else str(s["right_category"])
            print(f"| {label} | {s['n']} | {s['flagged']} | {rate} | {cat} |")
        print(f"\n(details: {out})")
        return 0

    if args.cmd == "reaudit":
        rows = reaudit(args.csv, args.runs_dir, args.db)
        out = args.runs_dir / f"reaudit-{stamp}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(table(rows))
        print(f"\n(per-run rows: {out})")
        return 0

    questions = [q.strip() for q in args.questions.read_text().splitlines() if q.strip() and not q.startswith("#")]
    rows = agents(questions, args.arms.split(","), args.reps, args.jobs, args.runs_dir, args.db, args.model)
    out = args.runs_dir / f"agents-{stamp}.csv"
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(table(rows))
    print(f"\n(per-run rows: {out}; answers and audits in {args.runs_dir}/<session>/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
