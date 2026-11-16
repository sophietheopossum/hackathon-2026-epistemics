"""epistaint: epistemic taint tracking for LLM agents.

Tag every tool call an agent makes with a provenance record (trace id, trust tier, output
hash, hash chain) at the execution boundary, then audit the agent's final answer claim by
claim against that log.
"""

__version__ = "0.1.0"
