import pytest

from caseflow.definitions import EdgeDef, NodeDef
from caseflow.packs import credit_review as pack
from caseflow.registry import PublishError
from caseflow.validate import validate_workflow

AGENTS = {a.key: a.inputs for a in pack.AGENTS}
TOOLS, TEMPLATES = {"metrics", "policy_check"}, {"credit_memo"}


def codes(workflow):
    return {i.code for i in validate_workflow(workflow, AGENTS, TOOLS, TEMPLATES)}


def test_the_pack_workflow_is_valid():
    assert codes(pack.workflow()) == set()


def test_a_loop_that_no_person_bounds_is_rejected():
    wf = pack.workflow()
    wf.edges.append(EdgeDef(source="synthesizer", target="financial_analyst"))
    assert "unbounded_cycle" in codes(wf)


def test_an_artifact_that_can_skip_review_is_rejected():
    wf = pack.workflow()
    wf.edges.append(EdgeDef(source="synthesizer", target="memo"))
    assert "artifact_without_review" in codes(wf)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda wf: wf.node("financial_analyst").inputs.pop("metrics"), "unbound_input"),
        (lambda wf: setattr(wf.node("financial_analyst"), "ref", "ghost_agent"), "unknown_agent"),
        (lambda wf: setattr(wf.node("metrics"), "ref", "ghost_tool"), "unknown_tool"),
        (lambda wf: wf.node("review").config.update(max_rounds=0), "unbounded_review"),
        (lambda wf: wf.nodes.append(NodeDef(id="orphan", type="end")), "unreachable"),
        (lambda wf: wf.node("metrics").inputs.update(case="memo.document"), "binding_not_upstream"),
        (
            lambda wf: wf.edges.remove(next(e for e in wf.edges if e.route == "rounds_exhausted")),
            "missing_route",
        ),
        (lambda wf: wf.edges.append(EdgeDef(source="review", target="memo")), "unrouted_edge"),
        (
            lambda wf: wf.edges.append(EdgeDef(source="metrics", target="nowhere")),
            "unknown_edge_node",
        ),
        (lambda wf: wf.nodes.append(NodeDef(id="start", type="start")), "duplicate_node"),
    ],
)
def test_draft_problems_are_reported_by_code(mutate, code):
    wf = pack.workflow()
    mutate(wf)
    assert code in codes(wf)


def test_publishing_pins_agent_references_and_keeps_revisions_immutable():
    registry, _ = pack.install()
    first, workflow = registry.workflow("credit-review")
    assert first.ref == "credit-review@1" and len(first.digest) == 64
    assert workflow.node("synthesizer").ref == "credit_synthesizer@1"

    changed = pack.AGENTS[2].model_copy(update={"instructions": "Be brief."})
    assert registry.publish_agent(changed, "editor").ref == "credit_synthesizer@2"
    second = registry.publish_workflow(pack.workflow(), "editor")
    assert second.ref == "credit-review@2" and second.digest != first.digest
    assert registry.workflow("credit-review@1")[0].digest == first.digest
    assert registry.workflow("credit-review@1")[1].node("synthesizer").ref == "credit_synthesizer@1"
    assert registry.workflow("credit-review")[1].node("synthesizer").ref == "credit_synthesizer@2"


def test_an_invalid_draft_cannot_be_published():
    registry, _ = pack.install()
    wf = pack.workflow()
    wf.edges.append(EdgeDef(source="synthesizer", target="memo"))
    with pytest.raises(PublishError) as error:
        registry.publish_workflow(wf, "editor")
    assert [i.code for i in error.value.issues] == ["artifact_without_review"]
