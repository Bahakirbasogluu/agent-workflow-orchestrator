"""Agents and workflows are documents, not code. They are drafted, validated and published."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

NodeType = Literal["start", "tool", "agent", "condition", "human_review", "artifact", "end"]
ROUTING_TYPES = ("condition", "human_review")


class AgentDef(BaseModel):
    key: str
    instructions: str
    inputs: list[str] = Field(description="Names of the input ports the agent reads")
    optional_inputs: list[str] = Field(default_factory=list)
    contract: str = Field(description="Name of the registered output schema")
    prose_fields: list[str] = Field(
        default_factory=list, description="Output fields whose figures must be grounded"
    )


class NodeDef(BaseModel):
    id: str
    type: NodeType
    ref: str | None = Field(
        None, description="Tool name, agent key (optionally key@revision) or template name"
    )
    inputs: dict[str, str] = Field(
        default_factory=dict, description="input port -> 'node.port' it reads"
    )
    config: dict[str, Any] = Field(default_factory=dict)


class EdgeDef(BaseModel):
    """Ordering. An edge with a route is followed only when the source node chose that route."""

    source: str
    target: str
    route: str | None = None


class WorkflowDef(BaseModel):
    key: str
    nodes: list[NodeDef]
    edges: list[EdgeDef]

    def node(self, node_id: str) -> NodeDef:
        return next(n for n in self.nodes if n.id == node_id)
