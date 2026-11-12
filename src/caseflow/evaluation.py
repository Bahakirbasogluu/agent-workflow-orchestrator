"""Golden cases: a case, the decisions people take on it, and what the run must then show."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from caseflow.service import Run, RunService


class Decision(BaseModel):
    reviewer: str
    role: str
    action: str
    comments: str = ""


class Expectations(BaseModel):
    outcome: list[str]
    recommendation: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    required_citations: list[str] = Field(default_factory=list)
    must_state: list[str] = Field(
        default_factory=list, description="Figures the final rationale must contain"
    )
    forbidden_phrases: list[str] = Field(default_factory=list)
    max_model_calls: int = 10


class GoldenCase(BaseModel):
    id: str
    case: dict[str, Any]
    started_by: str
    decisions: list[Decision]
    expect: Expectations


@dataclass(frozen=True)
class Check:
    id: str
    passed: bool
    detail: str = ""


def play(service: RunService, workflow: str, golden: GoldenCase) -> Run:
    run = service.start(workflow, golden.case, golden.started_by)
    for n, decision in enumerate(golden.decisions):
        if run.status != "waiting_for_input":
            break
        run = service.decide(run.id, idempotency_key=f"{golden.id}-{n}", **decision.model_dump())
    return run


def score(
    run: Run, golden: GoldenCase, *, final_agent: str = "synthesizer", metrics_node: str = "metrics"
) -> list[Check]:
    """One check per expectation, with stable ids so that two scorings can be compared."""
    e, prefix = golden.expect, golden.id
    final = run.output(final_agent) or {}
    prose = (
        " ".join(str(final.get(k, "")) for k in ("rationale",))
        + " "
        + " ".join(final.get("conditions", []))
    )
    metrics = (run.output(metrics_node, "result") or {}).get("values", {})
    cited = {
        c for node in run.nodes.values() for c in (node.get("output") or {}).get("citations", [])
    }
    removed = [s for ev in run.events for s in ev.get("removed", [])]
    checks = [
        Check(
            f"{prefix}/outcome",
            run.status == "completed" and run.outcome in e.outcome,
            f"{run.status}: {run.outcome or run.error}",
        )
    ]
    if e.recommendation:
        checks.append(
            Check(
                f"{prefix}/recommendation",
                final.get("recommendation") in e.recommendation,
                str(final.get("recommendation")),
            )
        )
    checks += [
        Check(f"{prefix}/metric:{k}", metrics.get(k) == v, f"{metrics.get(k)} (expected {v})")
        for k, v in e.metrics.items()
    ]
    checks += [
        Check(f"{prefix}/cites:{c}", any(x == c or x.startswith(c) for x in cited))
        for c in e.required_citations
    ]
    checks += [Check(f"{prefix}/states:{f}", f in prose) for f in e.must_state]
    checks += [
        Check(f"{prefix}/avoids:{p}", p.lower() not in prose.lower()) for p in e.forbidden_phrases
    ]
    checks.append(
        Check(f"{prefix}/no_invented_figures", not removed, f"{len(removed)} sentence(s) removed")
    )
    checks.append(
        Check(
            f"{prefix}/model_calls",
            run.model_calls <= e.max_model_calls,
            f"{run.model_calls} (limit {e.max_model_calls})",
        )
    )
    return checks


def compare(baseline: dict[str, bool], current: list[Check]) -> dict[str, Any]:
    """Labels each check against a stored baseline and gives an overall verdict."""
    labels = {}
    for check in current:
        before = baseline.get(check.id)
        labels[check.id] = (
            "new"
            if before is None
            else "unchanged"
            if before == check.passed
            else "improved"
            if check.passed
            else "regressed"
        )
    improved, regressed = "improved" in labels.values(), "regressed" in labels.values()
    verdict = (
        "mixed"
        if improved and regressed
        else "worse"
        if regressed
        else "better"
        if improved
        else "same"
    )
    return {"verdict": verdict, "checks": labels}
