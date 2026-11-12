"""Checks a workflow draft must pass before it can be published."""

from __future__ import annotations

from dataclasses import dataclass

from caseflow.definitions import ROUTING_TYPES, WorkflowDef


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    node: str | None = None


def _reachable(start: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(edges.get(node, ()))
    return seen


def _has_cycle(nodes: set[str], edges: dict[str, set[str]]) -> bool:
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for nxt in edges.get(node, ()):
            if nxt in nodes and (state.get(nxt) == 1 or (state.get(nxt) is None and visit(nxt))):
                return True
        state[node] = 2
        return False

    return any(state.get(n) is None and visit(n) for n in sorted(nodes))


def validate_workflow(
    wf: WorkflowDef, agents: dict[str, list[str]], tools: set[str], templates: set[str]
) -> list[Issue]:
    """`agents` maps each published agent key to its required input ports."""
    issues: list[Issue] = []
    ids = [n.id for n in wf.nodes]
    by_id = {n.id: n for n in wf.nodes}
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        issues.append(Issue("duplicate_node", f"node id {dup} is used more than once", dup))
    starts = [n.id for n in wf.nodes if n.type == "start"]
    if len(starts) != 1:
        issues.append(
            Issue("start_count", f"a workflow needs exactly one start node, found {len(starts)}")
        )
    if not any(n.type == "end" for n in wf.nodes):
        issues.append(Issue("no_end", "a workflow needs an end node"))

    forward: dict[str, set[str]] = {}
    for edge in wf.edges:
        for end in (edge.source, edge.target):
            if end not in by_id:
                issues.append(Issue("unknown_edge_node", f"edge refers to unknown node {end}"))
        forward.setdefault(edge.source, set()).add(edge.target)
        source = by_id.get(edge.source)
        if source is not None:
            if source.type in ROUTING_TYPES and edge.route is None:
                issues.append(
                    Issue(
                        "unrouted_edge",
                        "an edge leaving a routing node must name a route",
                        source.id,
                    )
                )
            if source.type not in ROUTING_TYPES and edge.route is not None:
                issues.append(
                    Issue(
                        "unexpected_route",
                        "only condition and human_review nodes choose routes",
                        source.id,
                    )
                )
    if issues:
        return issues  # the checks below need a well-formed graph

    reachable = _reachable(starts[0], forward)
    ancestors = {
        n.id: {a for a in by_id if n.id in _reachable(a, forward) and a != n.id} for n in wf.nodes
    }
    for node in wf.nodes:
        if node.id not in reachable:
            issues.append(Issue("unreachable", "no path from the start node", node.id))
        if node.type == "agent":
            key = (node.ref or "").split("@")[0]
            if key not in agents:
                issues.append(Issue("unknown_agent", f"agent {node.ref} is not published", node.id))
            else:
                for port in agents[key]:
                    if port not in node.inputs:
                        issues.append(
                            Issue("unbound_input", f"required input {port} is not bound", node.id)
                        )
        if node.type == "tool" and node.ref not in tools:
            issues.append(Issue("unknown_tool", f"tool {node.ref} is not registered", node.id))
        if node.type == "artifact" and node.ref not in templates:
            issues.append(
                Issue("unknown_template", f"template {node.ref} is not registered", node.id)
            )
        if node.type == "human_review" and int(node.config.get("max_rounds", 0)) < 1:
            issues.append(
                Issue("unbounded_review", "a review needs max_rounds of at least 1", node.id)
            )
        if node.type in ROUTING_TYPES:
            needed = (
                {"true", "false"}
                if node.type == "condition"
                else set(node.config.get("actions", ["approved", "changes_requested"]))
            )
            if "changes_requested" in needed:
                needed.add("rounds_exhausted")
            declared = {e.route for e in wf.edges if e.source == node.id}
            for route in sorted(needed - declared):
                issues.append(Issue("missing_route", f"no edge for the route {route}", node.id))
        for port, source in node.inputs.items():
            src = source.split(".")[0]
            if src not in by_id:
                issues.append(
                    Issue("unknown_binding", f"input {port} reads from unknown node {src}", node.id)
                )
            elif src not in ancestors[node.id]:
                issues.append(
                    Issue(
                        "binding_not_upstream",
                        f"input {port} reads from {src}, which does not run before",
                        node.id,
                    )
                )

    # Every loop must pass through a human review, which bounds it.
    unattended = {n.id for n in wf.nodes if n.type != "human_review"}
    if _has_cycle(unattended, {s: {t for t in ts if t in unattended} for s, ts in forward.items()}):
        issues.append(
            Issue("unbounded_cycle", "a loop without a human review in it never has to stop")
        )
    # Nothing is written without a person having approved it.
    without_review = {
        s: {t for t in ts if t in unattended} for s, ts in forward.items() if s in unattended
    }
    unreviewed = _reachable(starts[0], without_review)
    for node in wf.nodes:
        if node.type == "artifact" and node.id in unreviewed:
            issues.append(
                Issue(
                    "artifact_without_review",
                    "an artifact can be produced without passing a human review",
                    node.id,
                )
            )
    return issues
