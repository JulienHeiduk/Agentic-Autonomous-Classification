"""The linear orchestrator: proposes strategies restricted to the linear family.

A thin binding over agents.orchestrator so the shared machinery -- ledger context, the
feasibility and novelty guards, the curated playbook -- is not duplicated per family. What
differs per family is the estimator enum and the contract guidance, both of which live in
agents/families.py.
"""
from agents import orchestrator

FAMILY = "linear"


def propose(model: str = None, tries: int = 4) -> dict:
    spec = orchestrator.propose(model=model, tries=tries, family=FAMILY)
    spec["orchestrator"] = FAMILY
    return spec


def build_context() -> str:
    return orchestrator.build_context(FAMILY)
