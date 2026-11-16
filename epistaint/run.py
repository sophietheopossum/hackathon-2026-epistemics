"""Run Claude Code (headless) as the agent, with every tool call going through the proxy.

Built-in tools are disabled (--tools ""), so the only evidence the agent can gather is
through the proxied corpus MCP server, and every observation lands in the trace log.

Arms:
  baseline    proxy logs silently; ordinary research-assistant prompt
  provenance  results are tagged with trace ids; the prompt asks for [[t-...]] citations
  enforce     provenance, then the audit's feedback is sent back to the agent once
              (resumed session) and the revised answer is the one that counts

    python -m epistaint.run "Which model should ship?" --arm provenance
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .audit import audit, feedback, render
from .core import Store

ROOT = Path(__file__).resolve().parents[1]

BASE_PROMPT = (
    "You are a research assistant. Answer the user's question using only the corpus tools "
    "(mcp__corpus__*). Give a concise final answer."
)
PROVENANCE_PROMPT = BASE_PROMPT + (
    " Every tool result starts with an epistaint provenance tag naming a trace id. In your final "
    "answer, end every factual sentence with the trace id(s) of the tool output(s) it relies on, "
    "written as [[t-xxxxxxxx]]. Copy numbers and quotations exactly as they appear in the tool "
    "output, and use sql_query for any arithmetic rather than computing by hand. If the corpus "
    "does not support something, say so instead of guessing."
)
PROMPTS = {"baseline": BASE_PROMPT, "provenance": PROVENANCE_PROMPT, "enforce": PROVENANCE_PROMPT}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32]


def claude(prompt: str, cfg: Path, system: str, cwd: Path, model: str | None,
           resume: str | None = None, timeout: int = 600) -> dict:
    cmd = ["claude", "-p", prompt, "--mcp-config", str(cfg), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__corpus", "--output-format", "json",
           "--append-system-prompt", system]
    if model:
        cmd += ["--model", model]
    if resume:
        cmd += ["--resume", resume]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def run(question: str, arm: str, runs_dir: Path, db: Path, corpus: Path, policy: Path,
        model: str | None = None, tag: str = "") -> dict:
    session = f"{arm}-{time.strftime('%Y%m%d-%H%M%S')}-{tag or slug(question)}"
    out = runs_dir / session
    out.mkdir(parents=True, exist_ok=True)
    proxy_args = ["-m", "epistaint.proxy", "--db", str(db.resolve()), "--session", session,
                  "--policy", str(policy.resolve())]
    if arm == "baseline":
        proxy_args.append("--no-annotate")
    cfg = out / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"corpus": {
        "command": sys.executable,
        "args": proxy_args + ["--", sys.executable, "-m", "epistaint.toolbox", "--root", str(corpus.resolve())],
        "env": {"PYTHONPATH": str(ROOT)},
    }}}, indent=2))

    started = time.time()
    first = claude(question, cfg, PROMPTS[arm], out, model)
    answer = first.get("result", "")
    (out / "answer.md").write_text(answer)
    store = Store(db)
    report = audit(answer, store.session_traces(session), session)
    result = {"session": session, "arm": arm, "question": question, "answer": answer,
              "first_answer": answer, "revised": False, "cost_usd": first.get("total_cost_usd"),
              "seconds": None, "report": report.to_json()}

    if arm == "enforce" and report.summary["status"]["GROUNDED"] < report.summary["claims"]:
        fb = feedback(report)
        (out / "feedback.md").write_text(fb)
        second = claude(fb, cfg, PROMPTS[arm], out, model, resume=first.get("session_id"))
        answer = second.get("result", "")
        (out / "answer.revised.md").write_text(answer)
        report = audit(answer, store.session_traces(session), session)
        result.update(answer=answer, revised=True, report=report.to_json(),
                      cost_usd=(first.get("total_cost_usd") or 0) + (second.get("total_cost_usd") or 0))

    result["seconds"] = round(time.time() - started, 1)
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["_rendered"] = render(report, colour=sys.stdout.isatty())
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epistaint-run", description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument("question")
    ap.add_argument("--arm", choices=sorted(PROMPTS), default="provenance")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "traces.db")
    ap.add_argument("--corpus", type=Path, default=ROOT / "demo" / "corpus")
    ap.add_argument("--policy", type=Path, default=ROOT / "policy.toml")
    ap.add_argument("--model", help="Claude Code model alias or id (default: Claude Code's default)")
    args = ap.parse_args(argv)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    res = run(args.question, args.arm, args.runs_dir, args.db, args.corpus, args.policy, args.model)
    print(res["answer"])
    print()
    print(res["_rendered"])
    print(f"\n(session {res['session']}, {res['seconds']}s, saved in {args.runs_dir / res['session']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
