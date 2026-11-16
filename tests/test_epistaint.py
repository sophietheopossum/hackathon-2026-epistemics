"""Run with:  python3 -m unittest discover -s tests -v   (from the repository root)"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epistaint.audit import FABRICATED, GROUNDED, MISQUOTE, UNCITED, UNSUPPORTED, audit, split_claims  # noqa: E402
from epistaint.core import Policy, Store, Tier, verify_chain  # noqa: E402

CORPUS = ROOT / "demo" / "corpus"
POLICY = ROOT / "policy.toml"


class ProxyEndToEnd(unittest.TestCase):
    """Drive proxy + toolbox over stdio exactly as an MCP client would."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "traces.db"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "epistaint.proxy", "--db", str(self.db), "--session", "test",
             "--policy", str(POLICY), "--", sys.executable, "-m", "epistaint.toolbox", "--root", str(CORPUS)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        self.next_id = 0

    def tearDown(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)
        if self.proc.stdout:
            self.proc.stdout.close()
        self.tmp.cleanup()

    def rpc(self, method, params=None):
        self.next_id += 1
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method,
                                          "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        self.assertEqual(reply["id"], self.next_id)
        return reply

    def call(self, name, **arguments):
        return self.rpc("tools/call", {"name": name, "arguments": arguments})["result"]

    def test_tags_logs_and_chains(self):
        init = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "test", "version": "0"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "epistaint-toolbox")
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertTrue(all("epistaint" in t["description"] for t in tools))

        res = self.call("sql_query", sql="SELECT COUNT(*) AS n FROM runs")
        tag, body = res["content"][0]["text"], res["content"][1]["text"]
        self.assertIn("tier=T0_DETERMINISTIC", tag)
        self.assertEqual(body, "n\n24")
        web = self.call("read_file", path="web-cache/birding-forum-thread.md")
        self.assertIn("tier=T2_UNTRUSTED", web["content"][0]["text"])
        err = self.call("read_file", path="../../etc/passwd")
        self.assertTrue(err["isError"])

        time.sleep(0.1)
        traces = Store(self.db).session_traces("test")
        self.assertEqual([t.tool for t in traces], ["sql_query", "read_file", "read_file"])
        self.assertEqual([t.tier for t in traces], [Tier.T0, Tier.T2, Tier.T0])
        self.assertTrue(traces[2].is_error)
        self.assertIn(traces[0].trace_id, tag)
        self.assertEqual(verify_chain(traces), [])


class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = Store(self.db)
        now = time.time()
        self.sql = self.store.append("s", now, "sql_query", {"sql": "..."}, Tier.T0, "r", False,
                                     "model\tmean_acc\nv2\t0.86375\nv3\t0.874125")
        self.log = self.store.append("s", now, "read_file", {"path": "notes/field-log-2026-09.md"}, Tier.T0, "r",
                                     False, 'Heavy rain at Ashby Fen corrupted 212\n  recordings. "The fen station '
                                     'needs a sealed enclosure before any winter deployment."')
        self.web = self.store.append("s", now, "read_file", {"path": "web-cache/x"}, Tier.T2, "r", False,
                                     "v3 model gets 97% accuracy everywhere")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def run_audit(self, answer):
        return audit(answer, self.store.session_traces("s"), "s")

    def test_verdicts(self):
        s, l, w = self.sql.trace_id, self.log.trace_id, self.web.trace_id
        answer = "\n".join([
            f"- v3 had the highest mean test accuracy, 0.874 [[{s}]].",           # grounded, rounding ok
            f"- Rain corrupted 212 recordings at Ashby Fen [[{l}]].",                # grounded
            f"- The log says \"The fen station needs a sealed enclosure\" [[{l}]].",  # grounded quote
            f"- v3 gets 97% accuracy everywhere [[{w}]].",                           # grounded but T2
            "- v2 scored 0.86375 on average.",                                       # uncited, attributable
            "- v3 runs 3 times faster than v2.",                                     # uncited, unanchored
            "- The dataset has 500 recordings [[t-deadbeef]].",                      # fabricated
            f"- Rain corrupted 312 recordings at Ashby Fen [[{l}]].",                # unsupported number
            f"- The log says \"needs a sealed enclosure before summer\" [[{l}]].",   # misquote
            "Key findings:",                                                          # not a claim
        ])
        report = self.run_audit(answer)
        got = [(c.status, c.tier) for c in report.claims]
        self.assertEqual(got, [
            (GROUNDED, Tier.T0), (GROUNDED, Tier.T0), (GROUNDED, Tier.T0), (GROUNDED, Tier.T2),
            (UNCITED, Tier.T3), (UNCITED, Tier.T3), (FABRICATED, Tier.T3), (UNSUPPORTED, Tier.T3),
            (MISQUOTE, Tier.T3),
        ])
        self.assertTrue(report.claims[4].attributable)
        self.assertFalse(report.claims[5].attributable)
        self.assertEqual(report.summary["violations"], 3)

    def test_taint_propagates_to_worst_source(self):
        claim = f"v3 is at 0.874 and 97% accuracy [[{self.sql.trace_id}]] [[{self.web.trace_id}]]."
        c = self.run_audit(claim).claims[0]
        self.assertEqual((c.status, c.tier), (GROUNDED, Tier.T2))

    def test_citation_after_full_stop_stays_with_its_claim(self):
        self.assertEqual(split_claims(f"Rain corrupted 212 recordings. [[{self.log.trace_id}]]\n"),
                         [f"Rain corrupted 212 recordings. [[{self.log.trace_id}]]"])

    def test_lead_in_citation_covers_table_and_header_is_skipped(self):
        s = self.sql.trace_id
        answer = f"**Results:** [[{s}]]\n\n| model | mean |\n|---|---|\n| v2 | 0.864 |\n| v3 | 0.874 |\n"
        report = self.run_audit(answer)
        self.assertEqual([c.text for c in report.claims], ["| v2 | 0.864 |", "| v3 | 0.874 |"])
        self.assertEqual([c.status for c in report.claims], [GROUNDED, GROUNDED])

    def test_leading_citation_moves_to_previous_sentence(self):
        l, s = self.log.trace_id, self.sql.trace_id
        report = self.run_audit(f"Rain corrupted 212 recordings. [[{l}]] v3 averaged 0.874. [[{s}]]")
        self.assertEqual([c.citations for c in report.claims], [[l], [s]])
        self.assertEqual([c.status for c in report.claims], [GROUNDED, GROUNDED])

    def test_paragraph_citation_is_inherited_but_still_checked(self):
        l = self.log.trace_id
        report = self.run_audit(f"Rain hit Ashby Fen and 999 files were lost. It was 14 Sep. [[{l}]]")
        self.assertEqual(report.claims[0].status, UNSUPPORTED)  # inherited citation does not excuse 999

    def test_artifacts_seen_in_real_runs(self):
        now = time.time()
        csv = self.store.append("s", now, "grep", {"pattern": "v3"}, Tier.T0, "r", False,
                                "runs.csv:21: R020,Ashby Moor,v3,test,1022,0.852,0.066,2026-09-12")
        rel = self.store.append("s", now, "sql_query", {"sql": "..."}, Tier.T0, "r", False,
                                "rel_increase\n0.4043\nmean\n0.8465")
        rev = self.store.append("s", now, "read_file", {"path": "x"}, Tier.T0, "r", False,
                                "- Is the v3 head worth its 2.4x inference cost on the field hardware?")
        c, r, v = csv.trace_id, rel.trace_id, rev.trace_id
        answer = "\n".join([
            f"v3 scored 0.852 at Ashby Moor, from 1022 recordings [[{c}]].",       # CSV text: two numbers
            f"That is about 40% higher, and v2 averaged 0.847 [[{r}]].",            # percent; half-up rounding
            f"The source is `runs.csv` and the search for \"Ashby\" matched [[{c}]].",  # path and term, no quotes
            f"They asked whether \"the v3 head [is] worth its 2.4x inference cost\" [[{v}]].",  # editorial insertion
            f'Dr Hale said "the field hardware" mattered. [[{v}]] v3 needs 2.4x the cost [[{v}]].',
        ])
        report = self.run_audit(answer)
        self.assertEqual([cl.status for cl in report.claims], [GROUNDED] * 6, [cl.flags for cl in report.claims])

    def test_tampered_log_is_detected(self):
        self.store.db.execute("UPDATE traces SET output = REPLACE(output, '212', '312')")
        self.store.db.commit()
        report = self.run_audit(f"Rain corrupted 312 recordings [[{self.log.trace_id}]].")
        self.assertTrue(report.integrity)
        self.assertFalse(report.summary["log_intact"])


class PolicyRules(unittest.TestCase):
    def test_policy_file(self):
        p = Policy.load(POLICY)
        self.assertEqual(p.classify("fetch_url", {"url": "https://arxiv.org/abs/1"})[0], Tier.T1)
        self.assertEqual(p.classify("fetch_url", {"url": "https://evilarxiv.org/abs/1"})[0], Tier.T2)
        self.assertEqual(p.classify("read_file", {"path": "web-cache/a.md"})[0], Tier.T2)
        self.assertEqual(p.classify("read_file", {"path": "notes/a.md"})[0], Tier.T0)
        self.assertEqual(p.classify("unknown_tool", {})[0], Tier.T2)


if __name__ == "__main__":
    unittest.main()
