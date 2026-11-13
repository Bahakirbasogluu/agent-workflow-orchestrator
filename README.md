# caseflow

[![CI](https://github.com/Bahakirbasogluu/agent-workflow-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/Bahakirbasogluu/agent-workflow-orchestrator/actions/workflows/ci.yml)

A small engine for multi-agent workflows that people have to sign off. Workflows are documents: they are drafted, validated, published as immutable revisions and compiled into a [LangGraph](https://github.com/langchain-ai/langgraph) graph. Agents exchange results through typed ports, a run stops for human review and resumes from its checkpoint, and every run pins the exact versions it started on.

The design comes from a credit risk platform I built for a bank and later generalised into a sector-neutral case-agent platform. The code here is written from scratch around an invented credit review; it contains no client code or data.

## The example workflow

```
start ─> metrics ─┬─> financial_analyst ─────────────┐
                  └─> policy_check ─> policy_analyst ─┴─> synthesizer ─> review ──approved──> exposure
                                                             ^             │                    │
                                                             └─ changes ───┤          > 1,000,000 ?
                                                                requested  │            │        │
                                                              (2 rounds)   │           yes       no
                                                                           v            v        │
                                                                        stopped <── senior_review│
                                                                                     │ approved  │
                                                                                     v           v
                                                                                    memo ──────> done
```

Two deterministic tools calculate the ratios and check them against policy. Two analysts work in parallel on different inputs and a synthesizer joins their results into a recommendation. A credit officer approves or sends it back, at most twice. Large exposures need a second, different person. Only then is the memo written.

## Design decisions

- **Workflows are data.** A workflow is a JSON-serialisable document of nodes, input bindings and edges. Seven node types cover the example: `start`, `tool`, `agent`, `condition`, `human_review`, `artifact`, `end`. The compiler turns a published document into a LangGraph graph; there is no second planner deciding what runs next.
- **Agents talk through ports, not a shared conversation.** Each node writes only to its own slot in the state, and each input names the node and port it reads. What an agent saw can always be reconstructed, and parallel branches cannot overwrite each other.
- **A draft has to earn publication.** Validation rejects, by code, a loop that no human review bounds, an artifact that could be written without review, an unbound agent input, a missing route, an unreachable node and more. A published revision gets a number and a SHA-256 digest and never changes.
- **A run pins what it depends on.** The manifest records the workflow revision, each agent revision with its digest, tool versions and the model profile. Publishing a new revision does not affect a run in progress; a dependency that changed underneath a run stops it with `execution_profile_changed` instead of letting it finish on mixed versions.
- **An agent step is a contract, not a chat reply.** The output must fit a schema, with one repair attempt. It may cite only identifiers it was given. Every figure in its prose must equal a value in its evidence (as written, rounded, as a percentage, or in thousands or millions); a sentence with any other figure is removed. An earlier agent's output never counts as evidence for a figure.
- **Tools calculate; agents interpret.** Ratios and threshold checks are deterministic tools. Agents quote them and cannot change them.
- **Human review is a durable interrupt.** The run stops at the review node holding no worker and resumes from the checkpoint when a decision arrives. Exactly one decision is accepted per review; the same decision delivered twice is applied once. The reviewer must hold the role the node asks for and must be neither the person who started the run nor someone who already approved it.
- **Golden cases are the release gate.** A golden case is a case, the decisions people take on it, and what the run must then show: outcome, recommendation, exact metric values, required citations, figures the rationale must state, no removed sentences, and a ceiling on model calls. Checks have stable ids, so two scorings can be compared and a regression fails the gate.

## Quick start

```bash
pip install -e ".[dev]"
caseflow demo large
caseflow eval
pytest
```

`caseflow demo large` runs a 2.5 million request through both reviews:

```
run 2c0369a3 on credit-review@1, started by analyst.a
  start              start
  metrics            tool          tool=metrics@1
  financial_analyst  agent         agent=financial_analyst@1, model_calls=1
  policy             tool          tool=policy_check@1
  policy_analyst     agent         agent=policy_analyst@1, model_calls=1
  synthesizer        agent         agent=credit_synthesizer@1, model_calls=1

  waiting for credit_officer at 'review' (round 1)
  officer.b decides: approved

  review             human_review  action=approved, reviewer=officer.b
  exposure           condition     value=2500000

  waiting for senior_credit_officer at 'senior_review' (round 1)
  senior.c decides: approved

  senior_review      human_review  action=approved, reviewer=senior.c
  memo               artifact      template=credit_memo
  done               end

status: completed, outcome: completed, model calls: 3

# Credit memo: Bluefield Components GmbH

Requested amount: 2,500,000 EUR
...
**Recommendation:** approve with conditions

Leverage is 4.5x and interest cover is 4.0x. Policy is breached on leverage.

Conditions:
- Covenant on leverage, tested quarterly

Sources: calc:leverage, calc:interest_cover, policy:max_leverage
Approved by: officer.b and senior.c
```

`caseflow demo weak` shows the revision loop: the officer asks for changes, the synthesizer runs again with the comment as input, and the second round is approved. `caseflow eval` runs the three golden cases and prints one line per check (31 in total); `--save` stores the result and `--baseline` compares against a stored one, exiting 1 on a failed or regressed check.

## Using it from Python

```python
from caseflow import RunService
from caseflow.packs import credit_review as pack

registry, toolbox = pack.install()                 # publishes the agents and the workflow
service = RunService(registry, toolbox, pack.FixtureModel())

run = service.start("credit-review", pack.CASES["healthy"], started_by="analyst.a")
run.status                                          # "waiting_for_input"
run.pending["show"]["recommendation"]               # what the reviewer sees

run = service.decide(run.id, reviewer="officer.b", role="credit_officer",
                     action="approved", idempotency_key="decision-1")
print(run.output("memo", "document"))
```

Pass any LangGraph checkpointer to `RunService` to keep waiting runs across restarts; the tests resume a run from a second service instance. The agents run on `FixtureModel`, a deterministic stand-in that keeps the gate reproducible, or on a model:

```bash
pip install -e ".[azure]"
caseflow --backend azure demo large
```

## Writing a pack

The engine knows nothing about credit. `packs/credit_review.py` is everything the use case adds: two tools, three output contracts, three agent definitions, the workflow, the memo template, the sample cases and the golden set. A new use case is a new pack with the same parts.

## Tests

38 tests. They cover the validation codes, revision pinning, the agent contract (repair, citations, grounding), the parallel branches and the join, both review gates and the four-eyes rule, idempotent decisions, the bounded revision loop, resuming after a restart, a run keeping its revisions while new ones are published, a changed dependency stopping a run, and the golden-case gate failing when an agent invents a figure.

## Layout

```
src/caseflow/
  definitions.py   agent, node, edge and workflow documents
  validate.py      checks a draft must pass to be published
  registry.py      immutable revisions with digests
  agents.py        the agent step: contract, citations, grounding
  compiler.py      workflow document -> LangGraph graph
  service.py       start, review decisions, resume, manifest
  evaluation.py    golden cases: play, score, compare
  packs/           credit_review: tools, contracts, agents, workflow, golden set
  cli.py           caseflow demo / caseflow eval
tests/
```

## License

MIT
