"""The nn orchestrator: proposes strategies restricted to sklearn MLP models.

A thin binding over agents.orchestrator, like the gbm and linear ones -- the shared
machinery (ledger context, feasibility and novelty guards, curated playbook) is not
duplicated per family. The estimator enum and the contract guidance live in
agents/families.py.
"""
from agents import orchestrator

FAMILY = "nn"


def propose(model: str = None, tries: int = 4) -> dict:
    spec = orchestrator.propose(model=model, tries=tries, family=FAMILY)
    spec["orchestrator"] = FAMILY
    return spec


def build_context() -> str:
    return orchestrator.build_context()
