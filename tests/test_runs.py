import pytest
from langgraph.checkpoint.memory import InMemorySaver

from caseflow import ExecutionProfileChanged, ReviewError, RunService
from caseflow.evaluation import Check, compare, play, score
from caseflow.packs import credit_review as pack

OFFICER = {"reviewer": "officer.b", "role": "credit_officer"}
SENIOR = {"reviewer": "senior.c", "role": "senior_credit_officer"}


@pytest.fixture
def service():
    registry, toolbox = pack.install()
    return RunService(registry, toolbox, pack.FixtureModel())


def start(service, case="healthy"):
    return service.start(pack.WORKFLOW, pack.CASES[case], "analyst.a")


def test_a_run_stops_at_the_review_with_what_the_reviewer_needs(service):
    run = start(service)
    assert run.status == "waiting_for_input"
    assert run.pending["role"] == "credit_officer" and run.pending["round"] == 1
    assert run.pending["show"]["recommendation"]["recommendation"] == "approve"
    assert run.output("memo", "document") is None  # nothing is written before approval
    assert run.manifest["workflow"]["ref"] == "credit-review@1"
    assert run.manifest["agents"].keys() == {
        "financial_analyst@1",
        "policy_analyst@1",
        "credit_synthesizer@1",
    }


def test_both_analysts_finish_before_the_synthesizer_starts(service):
    order = [e["node"] for e in start(service).events]
    assert order.index("synthesizer") > max(
        order.index("financial_analyst"), order.index("policy_analyst")
    )
    assert order.count("synthesizer") == 1


def test_approval_writes_the_memo_and_small_exposures_skip_the_senior_review(service):
    run = service.decide(start(service).id, action="approved", idempotency_key="k1", **OFFICER)
    assert (run.status, run.outcome) == ("completed", "completed")
    assert "Approved by: officer.b" in run.output("memo", "document")
    assert "senior_review" not in run.nodes


def test_large_exposures_need_a_second_person(service):
    run = service.decide(
        start(service, "large").id, action="approved", idempotency_key="k1", **OFFICER
    )
    assert run.status == "waiting_for_input" and run.pending["node"] == "senior_review"
    with pytest.raises(ReviewError) as error:  # the first approver cannot also be the second
        service.decide(
            run.id,
            action="approved",
            idempotency_key="k2",
            reviewer="officer.b",
            role="senior_credit_officer",
        )
    assert error.value.code == "four_eyes"
    run = service.decide(run.id, action="approved", idempotency_key="k3", **SENIOR)
    assert "Approved by: officer.b and senior.c" in run.output("memo", "document")


def test_a_senior_rejection_ends_the_run_without_a_memo(service):
    run = service.decide(
        start(service, "large").id, action="approved", idempotency_key="k1", **OFFICER
    )
    run = service.decide(run.id, action="rejected", idempotency_key="k2", **SENIOR)
    assert (run.status, run.outcome) == ("completed", "returned_without_approval")
    assert "memo" not in run.nodes


@pytest.mark.parametrize(
    ("decision", "code"),
    [
        ({"reviewer": "analyst.a", "role": "credit_officer", "action": "approved"}, "four_eyes"),
        ({"reviewer": "officer.b", "role": "intern", "action": "approved"}, "wrong_role"),
        (
            {"reviewer": "officer.b", "role": "credit_officer", "action": "rubber_stamp"},
            "unknown_action",
        ),
    ],
)
def test_decisions_that_are_not_accepted(service, decision, code):
    run = start(service)
    with pytest.raises(ReviewError) as error:
        service.decide(run.id, idempotency_key="k1", **decision)
    assert error.value.code == code and service.runs[run.id].status == "waiting_for_input"


def test_the_same_decision_delivered_twice_is_applied_once(service):
    run = start(service, "weak")
    service.decide(
        run.id, action="changes_requested", comments="Add headroom", idempotency_key="k1", **OFFICER
    )
    again = service.decide(
        run.id, action="changes_requested", comments="Add headroom", idempotency_key="k1", **OFFICER
    )
    assert again.pending["round"] == 2 and again.model_calls == 4
    done = service.decide(run.id, action="approved", idempotency_key="k2", **OFFICER)
    with pytest.raises(ReviewError) as error:
        service.decide(done.id, action="approved", idempotency_key="k9", **OFFICER)
    assert error.value.code == "not_waiting"


def test_requested_changes_reach_the_synthesizer_and_the_loop_is_bounded(service):
    run = service.decide(
        start(service, "weak").id,
        action="changes_requested",
        comments="State the covenant headroom",
        idempotency_key="k1",
        **OFFICER,
    )
    assert "State the covenant headroom" in run.output("synthesizer")["conditions"][-1]
    run = service.decide(
        run.id, action="changes_requested", comments="Again", idempotency_key="k2", **OFFICER
    )
    assert (run.status, run.outcome) == ("completed", "returned_without_approval")
    assert [e["node"] for e in run.events].count("synthesizer") == 2


def test_a_waiting_run_survives_a_restart():
    registry, toolbox = pack.install()
    checkpoints, runs = InMemorySaver(), {}
    first = RunService(registry, toolbox, pack.FixtureModel(), checkpointer=checkpoints, runs=runs)
    run_id = start(first).id
    second = RunService(registry, toolbox, pack.FixtureModel(), checkpointer=checkpoints, runs=runs)
    run = second.decide(run_id, action="approved", idempotency_key="k1", **OFFICER)
    assert run.status == "completed" and run.model_calls == 3  # earlier steps were not run again


def test_a_run_keeps_the_revisions_it_started_on(service):
    waiting = start(service)
    service.registry.publish_agent(
        pack.AGENTS[2].model_copy(update={"instructions": "New wording."}), "editor"
    )
    service.registry.publish_workflow(pack.workflow(), "editor")
    finished = service.decide(waiting.id, action="approved", idempotency_key="k1", **OFFICER)
    assert (
        finished.manifest["workflow"]["ref"] == "credit-review@1" and finished.status == "completed"
    )
    assert start(service).manifest["agents"].keys() >= {"credit_synthesizer@2"}


def test_a_changed_dependency_stops_the_run_instead_of_mixing_versions(service):
    run = start(service)
    service.toolbox.tools["metrics"].version = "2"
    with pytest.raises(ExecutionProfileChanged):
        service.decide(run.id, action="approved", idempotency_key="k1", **OFFICER)
    assert (service.runs[run.id].status, service.runs[run.id].error) == (
        "failed",
        "execution_profile_changed",
    )


def test_an_agent_that_breaks_its_contract_fails_the_run():
    class Careless(pack.FixtureModel):
        def credit_synthesizer(self, inputs):
            return {"recommendation": "maybe"}

    registry, toolbox = pack.install()
    run = start(RunService(registry, toolbox, Careless()))
    assert run.status == "failed" and "CreditRecommendation contract" in run.error


def test_golden_cases_pass_on_the_fixture_model(service):
    checks = [c for g in pack.GOLDEN for c in score(play(service, pack.WORKFLOW, g), g)]
    assert len(checks) == 31 and all(c.passed for c in checks)


def test_an_invented_figure_is_removed_and_fails_the_gate():
    class Inventing(pack.FixtureModel):
        def credit_synthesizer(self, inputs):
            answer = super().credit_synthesizer(inputs)
            answer["rationale"] += " Revenue is expected to grow 37% next year."
            return answer

    registry, toolbox = pack.install()
    golden = pack.GOLDEN[0]
    run = play(RunService(registry, toolbox, Inventing()), pack.WORKFLOW, golden)
    assert "37%" not in run.output("synthesizer")["rationale"]
    failed = [c.id for c in score(run, golden) if not c.passed]
    assert failed == ["healthy/no_invented_figures"]


def test_compare_labels_each_check_against_the_baseline():
    baseline = {"a": True, "b": False, "c": True}
    result = compare(
        baseline, [Check("a", True), Check("b", True), Check("c", False), Check("d", True)]
    )
    assert result == {
        "verdict": "mixed",
        "checks": {"a": "unchanged", "b": "improved", "c": "regressed", "d": "new"},
    }
    assert (
        compare(baseline, [Check("a", True), Check("b", False), Check("c", True)])["verdict"]
        == "same"
    )
    assert compare(baseline, [Check("c", False)])["verdict"] == "worse"
