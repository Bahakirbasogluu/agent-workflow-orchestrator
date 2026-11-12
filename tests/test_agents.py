import pytest
from pydantic import BaseModel

from caseflow.agents import ContractError, Evidence, ground, render_inputs, run_agent
from caseflow.definitions import AgentDef


class Note(BaseModel):
    text: str
    citations: list[str]


AGENT = AgentDef(
    key="writer", instructions="Write.", inputs=["facts"], contract="Note", prose_fields=["text"]
)
FACTS = Evidence("fact", {"revenue": 12_000_000, "margin": 0.185, "leverage": 4.5})


class Scripted:
    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def complete(self, request):
        self.requests.append((request.repair, request.user))
        return self.answers.pop(0)


def test_output_is_checked_against_the_contract_and_repaired_once():
    model = Scripted(
        {"text": "Leverage is 4.5x."}, {"text": "Leverage is 4.5x.", "citations": ["fact:leverage"]}
    )
    result = run_agent(AGENT, {"facts": FACTS}, model, Note)
    assert result.model_calls == 2 and result.output["citations"] == ["fact:leverage"]
    assert model.requests[0][0] is None and "citations" in model.requests[1][0]


def test_a_second_invalid_answer_fails_the_step():
    with pytest.raises(ContractError, match="after one repair"):
        run_agent(AGENT, {"facts": FACTS}, Scripted({"text": "x"}, {"text": "y"}), Note)


def test_only_identifiers_that_were_given_can_be_cited():
    model = Scripted(
        {
            "text": "Leverage is 4.5x.",
            "citations": ["fact:leverage", "fact:made_up", "web:somewhere"],
        }
    )
    result = run_agent(AGENT, {"facts": FACTS}, model, Note)
    assert result.output["citations"] == ["fact:leverage"]
    assert result.dropped_citations == ["fact:made_up", "web:somewhere"]
    with pytest.raises(ContractError, match="cites nothing"):
        run_agent(
            AGENT,
            {"facts": FACTS},
            Scripted({"text": "Fine.", "citations": ["fact:made_up"]}),
            Note,
        )


def test_sentences_with_figures_that_are_not_in_the_evidence_are_removed():
    text = "Revenue is 12,000,000. Margin is 18.5% on 12.0 million. Growth was 23% last year. There are 3 covenants."
    kept, removed = ground(text, {"facts": FACTS})
    assert removed == ["Growth was 23% last year."]
    assert kept == "Revenue is 12,000,000. Margin is 18.5% on 12.0 million. There are 3 covenants."


def test_an_earlier_agent_output_does_not_ground_a_figure():
    earlier = {"assessment": "Revenue will reach 99,000,000."}
    _, removed = ground("Revenue will reach 99,000,000.", {"facts": FACTS, "earlier": earlier})
    assert removed == ["Revenue will reach 99,000,000."]
    rendered = render_inputs({"facts": FACTS, "earlier": earlier})
    assert "[fact:revenue] revenue = 12000000" in rendered and "not a source of figures" in rendered


def test_a_missing_required_input_stops_the_step_before_any_model_call():
    model = Scripted()
    with pytest.raises(ContractError, match="missing input facts"):
        run_agent(AGENT, {"facts": None}, model, Note)
    assert model.requests == []
