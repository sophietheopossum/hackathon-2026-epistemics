"""Transparent MCP stdio proxy that provenance-tags every tool call.

    MCP client (Claude Code, ...)  <-stdio->  epistaint-proxy  <-stdio->  upstream MCP server

Every JSON-RPC message is passed through unchanged, except:
  * tools/call responses: the result is logged to the trace store (trace id, trust tier,
    sha256 of the output, hash-chained to the previous record), and a short provenance
    tag naming the trace id is prepended to the result so the agent can cite it;
  * tools/list responses: each tool description gets a one-line citation instruction.

The trust tier is decided here, at the execution boundary, by a policy file - never by
the model.  With --no-annotate the proxy only logs (the baseline arm of an experiment).

Usage:
    python -m epistaint.proxy --db traces.db --policy policy.toml -- <upstream command ...>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

from .core import Policy, Store, result_text, tier_label

TAG = "[epistaint provenance] trace={tid} tier={tier} sha256={sha}... | cite claims based on this output as [[{tid}]]"
DESC_NOTE = " (epistaint: results carry a provenance trace id; cite facts taken from a result as [[t-xxxxxxxx]].)"


def _key(msg_id) -> str:
    return json.dumps(msg_id, sort_keys=True)


def _dump(msg) -> bytes:
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epistaint-proxy", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("EPISTAINT_DB", "traces.db"), help="SQLite trace store")
    ap.add_argument("--session", default=os.environ.get("EPISTAINT_SESSION"),
                    help="session name (default: $EPISTAINT_SESSION or a timestamp)")
    ap.add_argument("--policy", default=os.environ.get("EPISTAINT_POLICY"), help="tier policy (TOML)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="log traces but do not tag results or tool descriptions (baseline)")
    ap.add_argument("upstream", nargs=argparse.REMAINDER, help="-- upstream MCP server command")
    args = ap.parse_args(argv)

    upstream = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
    if not upstream:
        ap.error("give the upstream MCP server command after --")
    session = args.session or time.strftime("s-%Y%m%d-%H%M%S")
    policy = Policy.load(args.policy)
    annotate = not args.no_annotate

    proc = subprocess.Popen(upstream, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    pending_calls: dict[str, tuple[str, dict, float, float]] = {}
    pending_lists: set[str] = set()
    lock = threading.Lock()

    def client_to_upstream() -> None:
        assert proc.stdin is not None
        try:
            for line in sys.stdin.buffer:
                try:
                    msg = json.loads(line)
                except ValueError:
                    msg = None
                if isinstance(msg, dict) and "id" in msg:
                    method = msg.get("method")
                    if method == "tools/call":
                        params = msg.get("params") or {}
                        with lock:
                            pending_calls[_key(msg["id"])] = (
                                str(params.get("name", "?")),
                                params.get("arguments") or {},
                                time.time(),
                                time.monotonic(),
                            )
                    elif method == "tools/list" and annotate:
                        with lock:
                            pending_lists.add(_key(msg["id"]))
                proc.stdin.write(line)
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threading.Thread(target=client_to_upstream, daemon=True).start()

    store = Store(args.db)
    out = sys.stdout.buffer
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if isinstance(msg, dict) and "id" in msg and ("result" in msg or "error" in msg):
            key = _key(msg["id"])
            with lock:
                call = pending_calls.pop(key, None)
                is_list = key in pending_lists
                pending_lists.discard(key)
            if call is not None:
                tool, arguments, started, mono = call
                latency = (time.monotonic() - mono) * 1000
                tier, reason = policy.classify(tool, arguments)
                if "result" in msg and isinstance(msg["result"], dict):
                    result = msg["result"]
                    trace = store.append(session, started, tool, arguments, tier, reason,
                                         bool(result.get("isError")), result_text(result), latency)
                    if annotate:
                        tag = TAG.format(tid=trace.trace_id, tier=tier_label(tier), sha=trace.output_sha256[:12])
                        result["content"] = [{"type": "text", "text": tag}] + list(result.get("content") or [])
                        line = _dump(msg)
                else:  # JSON-RPC error: log it, nothing to annotate
                    store.append(session, started, tool, arguments, tier, reason, True,
                                 json.dumps(msg.get("error"), ensure_ascii=False), latency)
            elif is_list and isinstance(msg.get("result"), dict):
                for tool in msg["result"].get("tools") or []:
                    tool["description"] = (tool.get("description") or "") + DESC_NOTE
                line = _dump(msg)
        if not line.endswith(b"\n"):
            line += b"\n"
        out.write(line)
        out.flush()
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
