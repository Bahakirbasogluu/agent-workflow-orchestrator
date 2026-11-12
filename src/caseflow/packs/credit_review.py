"""A demonstration pack: a credit review on invented companies.

A pack carries everything a use case needs on top of the engine: the tools,
the output contracts, the agent and workflow definitions, the document
template and the golden cases that guard them.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from pydantic import BaseModel

from caseflow.agents import Evidence, ModelRequest
from caseflow.compiler import Toolbox
from caseflow.definitions import AgentDef, EdgeDef, NodeDef, WorkflowDef
from caseflow.evaluation import Decision, Expectations, GoldenCase
from caseflow.registry import Registry

WORKFLOW = "credit-review"
POLICY = {"max_leverage": 4.0, "min_interest_cover": 2.0, "min_current_ratio": 1.0}
SENIOR_APPROVAL_ABOVE = 1_000_000


def _ratio(a: float, b: float) -> float:
    return float((Decimal(str(a)) / Decimal(str(b))).quantize(Decimal("0.01"), ROUND_HALF_UP))


class MetricsTool:
    """Deterministic calculations. Agents quote these; they never compute figures themselves."""

    name, version = "metrics", "1"

    def run(self, case: Evidence) -> dict[str, Any]:
        f = case.values
        return {"result": Evidence("calc", {
            "leverage": _ratio(f["total_debt"], f["ebitda"]),
            "interest_cover": _ratio(f["ebitda"], f["interest_expense"]),
            "current_ratio": _ratio(f["current_assets"], f["current_liabilities"]),
        })}  # fmt: skip


class PolicyTool:
    name, version = "policy_check", "1"

    def run(self, metrics: Evidence) -> dict[str, Any]:
        m = metrics.values
        breaches = [name for name, broken in (
            ("leverage", m["leverage"] > POLICY["max_leverage"]),
            ("interest_cover", m["interest_cover"] < POLICY["min_interest_cover"]),
            ("current_ratio", m["current_ratio"] < POLICY["min_current_ratio"]),
        ) if broken]  # fmt: skip
        return {"result": Evidence("policy", dict(POLICY)), "breaches": breaches}


class FinancialAssessment(BaseModel):
    assessment: str
    strengths: list[str]
    concerns: list[str]
    citations: list[str]


class PolicyFindings(BaseModel):
    findings: str
    breaches: list[str]
    citations: list[str]


class CreditRecommendation(BaseModel):
    recommendation: Literal["approve", "approve_with_conditions", "decline"]
    rationale: str
    conditions: list[str]
    citations: list[str]


def credit_memo(
    case: Evidence,
    info: dict,
    metrics: Evidence,
    recommendation: dict,
    reviewer: str,
    senior_reviewer: str | None = None,
) -> str:
    lines = [
        f"# Credit memo: {info['borrower']}", "",
        f"Requested amount: {case.values['requested_amount']:,} {info['currency']}", "",
        "| Metric | Value |", "| --- | --- |",
        *[f"| {name.replace('_', ' ')} | {value} |" for name, value in metrics.values.items()], "",
        f"**Recommendation:** {recommendation['recommendation'].replace('_', ' ')}", "",
        recommendation["rationale"], "",
    ]  # fmt: skip
    if recommendation["conditions"]:
        lines += ["Conditions:", *[f"- {c}" for c in recommendation["conditions"]], ""]
    lines.append(f"Sources: {', '.join(recommendation['citations'])}")
    lines.append(
        f"Approved by: {reviewer}" + (f" and {senior_reviewer}" if senior_reviewer else "")
    )
    return "\n".join(lines) + "\n"


def toolbox() -> Toolbox:
    return Toolbox(
        tools={"metrics": MetricsTool(), "policy_check": PolicyTool()},
        templates={"credit_memo": credit_memo},
        contracts={
            c.__name__: c for c in (FinancialAssessment, PolicyFindings, CreditRecommendation)
        },
    )


AGENTS = [
    AgentDef(key="financial_analyst", inputs=["case", "metrics"], contract="FinancialAssessment", prose_fields=["assessment"],
             instructions="You are a financial analyst. Assess the borrower's capacity to service debt from the facts and the calculated metrics."),
    AgentDef(key="policy_analyst", inputs=["metrics", "policy"], contract="PolicyFindings", prose_fields=["findings"],
             instructions="You are a credit policy analyst. Compare each metric with its policy threshold and name every breach."),
    AgentDef(key="credit_synthesizer", inputs=["case", "metrics", "policy", "financial", "policy_view"], optional_inputs=["revision"],
             contract="CreditRecommendation", prose_fields=["rationale"],
             instructions="You write the credit recommendation from the two analyses. If a reviewer asked for changes, address them."),
]  # fmt: skip

_REVIEW_FORM = [{"name": "action", "type": "choice"}, {"name": "comments", "type": "text"}]


def workflow() -> WorkflowDef:
    n, e = NodeDef, EdgeDef
    return WorkflowDef(
        key=WORKFLOW,
        nodes=[
            n(id="start", type="start"),
            n(id="metrics", type="tool", ref="metrics", inputs={"case": "start.case"}),
            n(id="policy", type="tool", ref="policy_check", inputs={"metrics": "metrics.result"}),
            n(id="financial_analyst", type="agent", ref="financial_analyst", inputs={"case": "start.case", "metrics": "metrics.result"}),
            n(id="policy_analyst", type="agent", ref="policy_analyst", inputs={"metrics": "metrics.result", "policy": "policy.result"}),
            n(id="synthesizer", type="agent", ref="credit_synthesizer", inputs={
                "case": "start.case", "metrics": "metrics.result", "policy": "policy.result",
                "financial": "financial_analyst.output", "policy_view": "policy_analyst.output", "revision": "review.comments"}),
            n(id="review", type="human_review", inputs={"recommendation": "synthesizer.output"},
              config={"role": "credit_officer", "max_rounds": 2, "form": _REVIEW_FORM}),
            n(id="exposure", type="condition", inputs={"value": "start.case"},
              config={"field": "requested_amount", "op": ">", "right": SENIOR_APPROVAL_ABOVE}),
            n(id="senior_review", type="human_review", inputs={"recommendation": "synthesizer.output"},
              config={"role": "senior_credit_officer", "max_rounds": 1, "actions": ["approved", "rejected"], "form": _REVIEW_FORM}),
            n(id="memo", type="artifact", ref="credit_memo", inputs={
                "case": "start.case", "info": "start.info", "metrics": "metrics.result",
                "recommendation": "synthesizer.output", "reviewer": "review.reviewer", "senior_reviewer": "senior_review.reviewer"}),
            n(id="done", type="end"),
            n(id="stopped", type="end", config={"outcome": "returned_without_approval"}),
        ],
        edges=[
            e(source="start", target="metrics"),
            e(source="metrics", target="policy"),
            e(source="metrics", target="financial_analyst"),
            e(source="policy", target="policy_analyst"),
            e(source="financial_analyst", target="synthesizer"),
            e(source="policy_analyst", target="synthesizer"),
            e(source="synthesizer", target="review"),
            e(source="review", target="exposure", route="approved"),
            e(source="review", target="synthesizer", route="changes_requested"),
            e(source="review", target="stopped", route="rounds_exhausted"),
            e(source="exposure", target="senior_review", route="true"),
            e(source="exposure", target="memo", route="false"),
            e(source="senior_review", target="memo", route="approved"),
            e(source="senior_review", target="stopped", route="rejected"),
            e(source="memo", target="done"),
        ],
    )  # fmt: skip


def install(by: str = "pack:credit-review") -> tuple[Registry, Toolbox]:
    box = toolbox()
    registry = Registry(set(box.tools), set(box.templates))
    for agent in AGENTS:
        registry.publish_agent(agent, by)
    registry.publish_workflow(workflow(), by)
    return registry, box


class FixtureModel:
    """A deterministic stand-in for a language model, used by the demo, the tests and the release gate.

    It answers the way a careful model would on this pack: it quotes evidence
    values and cites their identifiers. Because it is deterministic, a change
    in a golden-case result always comes from a change in the workflow.
    """

    def complete(self, request: ModelRequest) -> dict:
        return getattr(self, request.agent)(request.inputs)

    def financial_analyst(self, i: dict) -> dict:
        m, f = i["metrics"].values, i["case"].values
        strong, weak = [], []
        (strong if m["interest_cover"] >= 3 else weak).append(
            f"Interest cover of {m['interest_cover']}x"
        )
        (strong if m["leverage"] <= 3 else weak).append(f"Leverage of {m['leverage']}x")
        (strong if m["current_ratio"] >= 1.2 else weak).append(
            f"Current ratio of {m['current_ratio']}"
        )
        text = (f"EBITDA of {f['ebitda']} covers interest {m['interest_cover']} times. "
                f"Debt stands at {m['leverage']} times EBITDA and the current ratio is {m['current_ratio']}.")  # fmt: skip
        return {"assessment": text, "strengths": strong, "concerns": weak,
                "citations": ["fact:ebitda", "calc:interest_cover", "calc:leverage", "calc:current_ratio"]}  # fmt: skip

    def policy_analyst(self, i: dict) -> dict:
        m, p = i["metrics"].values, i["policy"].values
        tests = (("leverage", m["leverage"] > p["max_leverage"], "max_leverage"),
                 ("interest_cover", m["interest_cover"] < p["min_interest_cover"], "min_interest_cover"),
                 ("current_ratio", m["current_ratio"] < p["min_current_ratio"], "min_current_ratio"))  # fmt: skip
        breaches = [name for name, broken, _ in tests if broken]
        parts = [
            f"{name.replace('_', ' ')} is {m[name]} against a threshold of {p[limit]}"
            for name, _, limit in tests
        ]
        return {"findings": "; ".join(parts).capitalize() + ".", "breaches": breaches,
                "citations": [f"calc:{n}" for n, _, _ in tests] + [f"policy:{limit}" for _, _, limit in tests]}  # fmt: skip

    def credit_synthesizer(self, i: dict) -> dict:
        m, breaches = i["metrics"].values, i["policy_view"]["breaches"]
        verdict = (
            "approve"
            if not breaches
            else "approve_with_conditions"
            if len(breaches) == 1
            else "decline"
        )
        conditions = (
            [f"Covenant on {b.replace('_', ' ')}, tested quarterly" for b in breaches]
            if len(breaches) == 1
            else []
        )
        if i.get("revision"):
            conditions.append(f"Reviewer request addressed: {i['revision']}")
        text = (f"Leverage is {m['leverage']}x and interest cover is {m['interest_cover']}x. "
                + ("All metrics are within policy." if not breaches else f"Policy is breached on {', '.join(b.replace('_', ' ') for b in breaches)}."))  # fmt: skip
        return {"recommendation": verdict, "rationale": text, "conditions": conditions,
                "citations": ["calc:leverage", "calc:interest_cover", "policy:max_leverage"]}  # fmt: skip


def _case(borrower: str, amount: int, **facts: int) -> dict:
    return {"borrower": borrower, "currency": "EUR", "facts": {"requested_amount": amount, **facts}}


CASES = {
    "healthy": _case("Northwind Supplies Ltd.", 400_000, revenue=12_000_000, ebitda=2_000_000, total_debt=4_000_000,
                     interest_expense=400_000, current_assets=3_000_000, current_liabilities=2_000_000),
    "large": _case("Bluefield Components GmbH", 2_500_000, revenue=30_000_000, ebitda=4_000_000, total_debt=18_000_000,
                   interest_expense=1_000_000, current_assets=9_000_000, current_liabilities=6_000_000),
    "weak": _case("Harbor Freight Partners Ltd.", 300_000, revenue=5_000_000, ebitda=500_000, total_debt=3_000_000,
                  interest_expense=400_000, current_assets=1_200_000, current_liabilities=1_000_000),
}  # fmt: skip

GOLDEN = [
    GoldenCase(id="healthy", case=CASES["healthy"], started_by="analyst.a",
               decisions=[Decision(reviewer="officer.b", role="credit_officer", action="approved")],
               expect=Expectations(outcome=["completed"], recommendation=["approve"],
                                   metrics={"leverage": 2.0, "interest_cover": 5.0, "current_ratio": 1.5},
                                   required_citations=["calc:leverage", "policy:"], must_state=["2.0x", "5.0x"],
                                   forbidden_phrases=["guarantee"], max_model_calls=3)),
    GoldenCase(id="large", case=CASES["large"], started_by="analyst.a",
               decisions=[Decision(reviewer="officer.b", role="credit_officer", action="approved"),
                          Decision(reviewer="senior.c", role="senior_credit_officer", action="approved")],
               expect=Expectations(outcome=["completed"], recommendation=["approve_with_conditions"],
                                   metrics={"leverage": 4.5, "interest_cover": 4.0, "current_ratio": 1.5},
                                   required_citations=["calc:leverage"], must_state=["4.5x"], max_model_calls=3)),
    GoldenCase(id="weak", case=CASES["weak"], started_by="analyst.a",
               decisions=[Decision(reviewer="officer.b", role="credit_officer", action="changes_requested", comments="State the covenant headroom"),
                          Decision(reviewer="officer.b", role="credit_officer", action="approved")],
               expect=Expectations(outcome=["completed"], recommendation=["decline"],
                                   metrics={"leverage": 6.0, "interest_cover": 1.25, "current_ratio": 1.2},
                                   required_citations=["calc:interest_cover"], must_state=["6.0x", "1.25x"], max_model_calls=4)),
]  # fmt: skip
