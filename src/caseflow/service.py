"""Starting, pausing and resuming runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from caseflow.agents import ContractError, Model
from caseflow.compiler import Toolbox, compile_workflow
from caseflow.registry import Registry


class ReviewError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ExecutionProfileChanged(Exception):
    """Something the run pinned at start is no longer what it was."""


@dataclass
class Run:
    id: str
    workflow: str
    manifest: dict
    started_by: str
    status: str = "running"  # running, waiting_for_input, completed, failed
    pending: dict | None = None
    approvers: list[str] = field(default_factory=list)
    decisions: dict[str, str] = field(default_factory=dict)  # idempotency key -> action
    nodes: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    error: str | None = None

    def output(self, node: str, port: str = "output") -> Any:
        return self.nodes.get(node, {}).get(port)

    @property
    def outcome(self) -> str | None:
        return next((n["outcome"] for n in self.nodes.values() if "outcome" in n), None)

    @property
    def model_calls(self) -> int:
        return sum(e.get("model_calls", 0) for e in self.events)


class RunService:
    def __init__(self, registry: Registry, toolbox: Toolbox, model: Model, *, model_profile: str = "fixture",
                 checkpointer: Any = None, runs: dict[str, Run] | None = None):  # fmt: skip
        self.registry, self.toolbox, self.model, self.model_profile = (
            registry,
            toolbox,
            model,
            model_profile,
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.runs = runs if runs is not None else {}
        self._graphs: dict[str, Any] = {}

    def manifest(self, workflow_ref: str) -> dict:
        """Everything a run's result depends on, by reference and digest."""
        revision, workflow = self.registry.workflow(workflow_ref)
        agents = {
            n.ref: self.registry.agent(n.ref)[0].digest for n in workflow.nodes if n.type == "agent"
        }
        tools = {
            n.ref: self.toolbox.tools[n.ref].version for n in workflow.nodes if n.type == "tool"
        }
        return {
            "workflow": {"ref": revision.ref, "digest": revision.digest},
            "agents": agents,
            "tools": tools,
            "model": self.model_profile,
        }

    def start(self, workflow_ref: str, case: dict, started_by: str) -> Run:
        revision, _ = self.registry.workflow(workflow_ref)
        run = Run(uuid.uuid4().hex, revision.ref, self.manifest(revision.ref), started_by)
        self.runs[run.id] = run
        return self._advance(run, {"case": case, "nodes": {}, "rounds": {}, "events": []})

    def decide(
        self,
        run_id: str,
        *,
        reviewer: str,
        role: str,
        action: str,
        comments: str = "",
        idempotency_key: str,
    ) -> Run:
        """Answers the pending review. Exactly one decision is accepted for it."""
        run = self.runs[run_id]
        if idempotency_key in run.decisions:
            return run  # a repeated delivery of the same decision changes nothing
        if run.status != "waiting_for_input":
            raise ReviewError("not_waiting", f"run is {run.status}; there is nothing to decide")
        pending = run.pending
        if action not in pending["actions"]:
            raise ReviewError("unknown_action", f"{action} is not one of {pending['actions']}")
        if role != pending["role"]:
            raise ReviewError("wrong_role", f"this review needs the role {pending['role']}")
        if reviewer == run.started_by or reviewer in run.approvers:
            raise ReviewError(
                "four_eyes",
                "a review must come from someone who neither started nor already approved the run",
            )
        run.decisions[idempotency_key] = action
        if action == "approved":
            run.approvers.append(reviewer)
        return self._advance(
            run, Command(resume={"action": action, "reviewer": reviewer, "comments": comments})
        )

    def _advance(self, run: Run, payload: Any) -> Run:
        if self.manifest(run.workflow) != run.manifest:
            run.status, run.error = "failed", "execution_profile_changed"
            raise ExecutionProfileChanged(
                f"run {run.id} was started on a different profile than the one now installed"
            )
        graph = self._graphs.get(run.workflow)
        if graph is None:
            _, workflow = self.registry.workflow(run.workflow)
            graph = self._graphs[run.workflow] = compile_workflow(
                workflow, self.registry, self.toolbox, self.model, self.checkpointer
            )
        try:
            state = graph.invoke(payload, {"configurable": {"thread_id": run.id}})
        except ContractError as error:
            run.status, run.error = "failed", str(error)
            return run
        run.nodes, run.events = state.get("nodes", {}), state.get("events", [])
        waiting = state.get("__interrupt__")
        run.status, run.pending = (
            ("waiting_for_input", waiting[0].value) if waiting else ("completed", None)
        )
        return run
