"""Published revisions are immutable. A run refers to them by key@revision and by digest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from caseflow.definitions import AgentDef, WorkflowDef
from caseflow.validate import Issue, validate_workflow


class PublishError(Exception):
    def __init__(self, issues: list[Issue]):
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues))
        self.issues = issues


def digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Revision:
    ref: str  # key@n
    digest: str
    document: dict
    published_by: str


class Registry:
    def __init__(self, tools: set[str], templates: set[str]):
        self._agents: dict[str, list[Revision]] = {}
        self._workflows: dict[str, list[Revision]] = {}
        self._tools, self._templates = tools, templates

    def publish_agent(self, agent: AgentDef, by: str) -> Revision:
        return self._append(self._agents, agent.key, agent.model_dump(), by)

    def publish_workflow(self, workflow: WorkflowDef, by: str) -> Revision:
        """Validates the draft and pins every agent reference to a revision before storing it."""
        required = {key: AgentDef(**revs[-1].document).inputs for key, revs in self._agents.items()}
        if issues := validate_workflow(workflow, required, self._tools, self._templates):
            raise PublishError(issues)
        pinned = workflow.model_copy(deep=True)
        for node in pinned.nodes:
            if node.type == "agent" and "@" not in node.ref:
                node.ref = self._agents[node.ref][-1].ref
        return self._append(self._workflows, workflow.key, pinned.model_dump(), by)

    @staticmethod
    def _append(table: dict[str, list[Revision]], key: str, document: dict, by: str) -> Revision:
        revisions = table.setdefault(key, [])
        revision = Revision(f"{key}@{len(revisions) + 1}", digest(document), document, by)
        revisions.append(revision)
        return revision

    def _get(self, table: dict[str, list[Revision]], ref: str) -> Revision:
        key, _, number = ref.partition("@")
        revisions = table[key]
        return revisions[int(number) - 1] if number else revisions[-1]

    def agent(self, ref: str) -> tuple[Revision, AgentDef]:
        revision = self._get(self._agents, ref)
        return revision, AgentDef(**revision.document)

    def workflow(self, ref: str) -> tuple[Revision, WorkflowDef]:
        revision = self._get(self._workflows, ref)
        return revision, WorkflowDef(**revision.document)
