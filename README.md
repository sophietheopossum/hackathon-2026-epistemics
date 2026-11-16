# hackathon-2026-epistemics

**Epistemic Taint Tracking: Tracing LLM Agent Claims Back to Their Tool-Call Evidence**
(Apart Research AI x Epistemics Research Sprint, November 2026; the write-up is `report.lyx`).

`epistaint` stops an agent from disguising a guess as a verified fact. It does not try to make
the model hallucinate less; it records, outside the model's control, what the agent actually
observed, and then checks every claim in the agent's answer against that record.

```
 MCP client (Claude Code, ...)  <-stdio->  epistaint proxy  <-stdio->  any MCP server
                                               |
                                               v
                                 traces.db: trace id, trust tier, output,
                                 sha256, hash chain (tamper-evident)
 agent's answer  --------------->  epistaint audit  -->  per-claim verdict + taint
```

1. **Proxy** (`epistaint/proxy.py`): a transparent MCP stdio proxy. Every `tools/call` result is
   logged with a trace id, the SHA-256 of the output and a hash chain over the session, and is
   given a trust tier by a policy file (the model never chooses it). The result is tagged
   `[[t-xxxxxxxx]]` so the agent can cite it. It wraps any MCP server and any MCP client
   without modifying either.
2. **Auditor** (`epistaint/audit.py`): splits the answer into claims (sentences, bullets, table
   rows) and checks each:

   | verdict | meaning |
   |---|---|
   | `GROUNDED` | cites real traces; every quote and number appears in the cited output (numbers may be rounded to the precision stated) |
   | `FABRICATED_CITATION` | cites a trace id that is not in the session log |
   | `MISQUOTE` | a quoted span is not in the cited output |
   | `UNSUPPORTED_NUMBER` | a number is not in the cited output |
   | `UNCITED` | no citation: tier T3 (synthetic). Reported as *attributable* if its numbers/quotes exist in some output, *unanchored* if in none |

   A claim's **taint** is the least trustworthy tier it relies on (T0 deterministic, T1 attested,
   T2 untrusted, T3 synthetic), so a conclusion mixing a SQL result with a forum post is T2.
   The audit also re-verifies every output hash and the hash chain, so an edited log is reported.
   `--feedback` prints a correction message an enforcing harness can send back to the agent.

## Quick start

Python 3.11+, no dependencies.

```sh
python3 -m unittest discover -s tests -v          # 9 tests, incl. a real stdio round trip

# any MCP server behind the proxy (here the demo toolbox):
python3 -m epistaint.proxy --db traces.db --policy policy.toml -- \
    python3 -m epistaint.toolbox --root demo/corpus

# audit an answer against the latest session (exit 2 on violations, --json, --feedback, --strict):
python3 -m epistaint.audit answer.md --db traces.db
```

### With Claude Code as the agent

`epistaint.run` starts Claude Code headless with its built-in tools disabled (`--tools ""`), so the
only evidence it can gather comes through the proxied corpus server:

```sh
python3 -m epistaint.run "Does any model version meet the deployment criterion?" --arm provenance
```

Arms: `baseline` (proxy logs silently, plain prompt), `provenance` (tagged results, citation
prompt), `enforce` (provenance, then the audit's feedback is sent back once and the revised
answer counts). Answers, audits and the MCP config are saved in `runs/<session>/`.

To put the proxy in front of a real server in your own client config, wrap its command:

```json
{"mcpServers": {"fs": {"command": "python3", "args": ["-m", "epistaint.proxy", "--db", "/abs/traces.db",
  "--policy", "/abs/policy.toml", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/abs/dir"],
  "env": {"PYTHONPATH": "/abs/path/to/this/repo"}}}}
```

## Trust tiers

`policy.toml` maps tool calls to tiers; the first matching rule wins and rules can look at
arguments (a regex on a path, a domain allowlist for a URL):

- local SQL, file reads, grep: **T0** deterministic, reproducible observation
- `fetch_url` on allowlisted domains: **T1** attested primary source
- other web content, and local copies of it (`web-cache/`): **T2** untrusted
- anything with no tool evidence: **T3** synthetic

## Experiments

```sh
python3 -m epistaint.bench agents        # demo/questions.txt x {baseline, provenance, enforce}
python3 -m epistaint.bench inject        # plant errors in real grounded claims, measure detection
```

The demo corpus (`demo/corpus/`) is a fictional field trial, so the model cannot answer from
memory, and it contains traps: an unverified forum post claiming "97% accuracy everywhere",
validation numbers next to test numbers, a site score the trial lead says should not be
trusted, and a question the corpus does not answer.

## Limitations

- Numbers and quotes are matched against the cited output as text: a citation of a different
  output that happens to contain the same number passes, and a claim with nothing checkable
  in it (no number, no quote) can only be checked for citation existence.
- Whether a claim *correctly interprets* its evidence (e.g. validation vs test accuracy) is not
  checked; that needs a semantic entailment step (an NLI model or LLM judge) on top of the log.
- Tiers describe how an observation was made, not whether the source is right: a T0 read of a
  wrong file is still wrong.
- The proxy speaks MCP over stdio; HTTP transports are not wrapped yet.
