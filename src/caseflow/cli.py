from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caseflow.evaluation import compare, play, score
from caseflow.packs import credit_review as pack
from caseflow.service import RunService


def _service(backend: str) -> RunService:
    registry, toolbox = pack.install()
    if backend == "azure":
        from caseflow.agents import AzureOpenAIModel

        return RunService(registry, toolbox, AzureOpenAIModel(), model_profile="azure-openai")
    return RunService(registry, toolbox, pack.FixtureModel())


def _trace(run, seen: int) -> int:
    for event in run.events[seen:]:
        extra = ", ".join(
            f"{k}={v}"
            for k, v in event.items()
            if k not in ("node", "type") and v not in ([], None)
        )
        print(f"  {event['node']:<18} {event['type']:<13} {extra}")
    return len(run.events)


def demo(service: RunService, name: str) -> int:
    golden = next(g for g in pack.GOLDEN if g.id == name)
    run = service.start(pack.WORKFLOW, golden.case, golden.started_by)
    print(f"run {run.id[:8]} on {run.workflow}, started by {golden.started_by}")
    seen = _trace(run, 0)
    for n, decision in enumerate(golden.decisions):
        if run.status != "waiting_for_input":
            break
        print(
            f"\n  waiting for {run.pending['role']} at '{run.pending['node']}' (round {run.pending['round']})"
        )
        print(
            f"  {decision.reviewer} decides: {decision.action}"
            + (f" - {decision.comments}" if decision.comments else "")
            + "\n"
        )
        run = service.decide(run.id, idempotency_key=f"demo-{n}", **decision.model_dump())
        seen = _trace(run, seen)
    print(f"\nstatus: {run.status}, outcome: {run.outcome}, model calls: {run.model_calls}\n")
    if document := run.output("memo", "document"):
        print(document)
    return 0


def evaluate(service: RunService, baseline: Path | None, save: Path | None) -> int:
    checks = [
        c for golden in pack.GOLDEN for c in score(play(service, pack.WORKFLOW, golden), golden)
    ]
    for check in checks:
        print(f"{'pass' if check.passed else 'FAIL'}  {check.id:<34} {check.detail}")
    failed = sum(not c.passed for c in checks)
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed")
    if save:
        save.write_text(json.dumps({c.id: c.passed for c in checks}, indent=2), encoding="utf-8")
    if baseline:
        result = compare(json.loads(baseline.read_text(encoding="utf-8")), checks)
        changed = {k: v for k, v in result["checks"].items() if v != "unchanged"}
        print(f"against baseline: {result['verdict']}" + (f" {changed}" if changed else ""))
        failed += result["verdict"] in ("worse", "mixed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="caseflow")
    parser.add_argument("--backend", choices=("fixture", "azure"), default="fixture")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="run one sample case with its review decisions")
    d.add_argument("case", choices=[g.id for g in pack.GOLDEN], nargs="?", default="large")
    e = sub.add_parser("eval", help="run the golden cases; exits 1 on a failed or regressed check")
    e.add_argument("--baseline", type=Path)
    e.add_argument("--save", type=Path)
    args = parser.parse_args(argv)
    service = _service(args.backend)
    return (
        demo(service, args.case)
        if args.command == "demo"
        else evaluate(service, args.baseline, args.save)
    )


if __name__ == "__main__":
    sys.exit(main())
