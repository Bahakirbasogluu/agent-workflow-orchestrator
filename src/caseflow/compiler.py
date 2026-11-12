"""Compiles a published workflow document into a LangGraph graph.

Every node writes only to its own slot in the state. Nodes exchange values
through bindings ("this input reads that node's port"), never through a shared
conversation, so what an agent saw can always be reconstructed.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from caseflow.agents import Evidence, Model, restore, run_agent
from caseflow.definitions import NodeDef, WorkflowDef
from caseflow.registry import Registry


def _merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class State(TypedDict, total=False):
    case: dict
    nodes: Annotated[dict, _merge]
    rounds: Annotated[dict, _merge]
    events: Annotated[list, operator.add]


class Tool(Protocol):
    name: str
    version: str

    def run(self, **inputs: Any) -> dict[str, Any]: ...


@dataclass
class Toolbox:
    tools: dict[str, Tool] = field(default_factory=dict)
    templates: dict[str, Callable[..., str]] = field(default_factory=dict)
    contracts: dict[str, type[BaseModel]] = field(default_factory=dict)


_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}  # fmt: skip


def _dump(value: Any) -> Any:
    return value.dump() if isinstance(value, Evidence) else value


def _bound(state: State, node: NodeDef) -> dict[str, Any]:
    """Resolves each input of a node to the value its source node published, or None."""
    values = {}
    for name, source in node.inputs.items():
        src, _, port = source.partition(".")
        values[name] = restore(state.get("nodes", {}).get(src, {}).get(port))
    return values


def _executor(
    node: NodeDef, registry: Registry, toolbox: Toolbox, model: Model
) -> Callable[[State], dict]:
    def done(output: dict, **event: Any) -> dict:
        return {
            "nodes": {node.id: output},
            "events": [{"node": node.id, "type": node.type, **event}],
        }

    if node.type == "start":
        return lambda state: done(
            {"case": Evidence("fact", state["case"]["facts"]).dump(), "info": state["case"]}
        )

    if node.type == "tool":
        tool = toolbox.tools[node.ref]
        return lambda state: done(
            {k: _dump(v) for k, v in tool.run(**_bound(state, node)).items()},
            tool=f"{tool.name}@{tool.version}",
        )

    if node.type == "agent":
        revision, agent = registry.agent(node.ref)
        schema = toolbox.contracts[agent.contract]

        def run(state: State) -> dict:
            result = run_agent(agent, _bound(state, node), model, schema)
            return done({"output": result.output}, agent=revision.ref, model_calls=result.model_calls,
                        removed=result.removed, dropped_citations=result.dropped_citations)  # fmt: skip

        return run

    if node.type == "condition":
        compare = _OPS[node.config["op"]]

        def decide(state: State) -> dict:
            subject = _bound(state, node)["value"]
            value = (subject.values if isinstance(subject, Evidence) else subject)[
                node.config["field"]
            ]
            return done(
                {"route": "true" if compare(value, node.config["right"]) else "false"}, value=value
            )

        return decide

    if node.type == "human_review":
        actions = node.config.get("actions", ["approved", "changes_requested"])
        max_rounds = int(node.config["max_rounds"])

        def review(state: State) -> dict:
            asked = state.get("rounds", {}).get(node.id, 0)
            # The run stops here and holds no worker until a person decides.
            decision = interrupt({
                "node": node.id, "role": node.config["role"], "actions": actions, "round": asked + 1,
                "form": node.config.get("form", []), "show": {k: _dump(v) for k, v in _bound(state, node).items()},
            })  # fmt: skip
            route = decision["action"]
            if route == "changes_requested" and asked + 1 >= max_rounds:
                route = "rounds_exhausted"
            output = {
                "route": route,
                "action": decision["action"],
                "reviewer": decision["reviewer"],
                "comments": decision.get("comments", ""),
            }
            return {
                **done(output, action=decision["action"], reviewer=decision["reviewer"]),
                "rounds": {node.id: asked + 1},
            }

        return review

    if node.type == "artifact":
        template = toolbox.templates[node.ref]
        return lambda state: done({"document": template(**_bound(state, node))}, template=node.ref)

    return lambda state: done({"outcome": node.config.get("outcome", "completed")})


def compile_workflow(
    workflow: WorkflowDef, registry: Registry, toolbox: Toolbox, model: Model, checkpointer: Any
):
    graph = StateGraph(State)
    for node in workflow.nodes:
        graph.add_node(node.id, _executor(node, registry, toolbox, model))
        if node.type == "start":
            graph.add_edge(START, node.id)
        if node.type == "end":
            graph.add_edge(node.id, END)

    static: dict[str, list[str]] = {}
    routed: dict[str, dict[str, list[str]]] = {}
    for edge in workflow.edges:
        if edge.route is None:
            static.setdefault(edge.target, []).append(edge.source)
        else:
            routed.setdefault(edge.source, {}).setdefault(edge.route, []).append(edge.target)
    for target, sources in static.items():
        # Several unconditional predecessors form a join: the node waits for all of them.
        graph.add_edge(sources[0] if len(sources) == 1 else sorted(set(sources)), target)
    for source, routes in routed.items():
        targets = sorted({t for ts in routes.values() for t in ts})
        graph.add_conditional_edges(
            source, lambda state, s=source, r=routes: r[state["nodes"][s]["route"]], targets
        )
    return graph.compile(checkpointer=checkpointer)
