from dataclasses import dataclass
from typing import Any, Callable, Dict

from state.worldstate import WorldState


@dataclass
class ToolContext:
    """Shared dependencies passed to tools that interact with robot state."""
    
    worldstate: WorldState

@dataclass(frozen=True)
class Tool:
    """Pairs an LLM tool JSON schema with its Python handler."""
    
    name: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    needs_context: bool = False
