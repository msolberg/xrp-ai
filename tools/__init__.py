import json
import logging
from typing import Any, Dict, List, Union

from tools.base import Tool, ToolContext
from tools.math_tools import CALCULATOR_TOOL

logger = logging.getLogger(__name__)

_ALL_TOOLS: List[Tool] = [
    CALCULATOR_TOOL,
]

_REGISTRY: Dict[str, Tool] = {tool.name: tool for tool in _ALL_TOOLS}
TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": tool.schema} for tool in _ALL_TOOLS
]


def execute_tool(
    name: str,
    raw_arguments: Union[str, Dict[str, Any], None],
    ctx: ToolContext,
) -> str:
    """Execute a registered tool by name and return JSON-encoded output."""
    tool = _REGISTRY.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    try:
        if isinstance(raw_arguments, dict):
            args = raw_arguments
        elif raw_arguments:
            args = json.loads(raw_arguments)
        else:
            args = {}
        
        if not isinstance(args, dict):
            return json.dumps({"error": f"Expected JSON object arguments for {name}"})
        
        result = tool.handler(ctx, **args) if tool.needs_context else tool.handler(**args)
        return json.dumps(result)
    except Exception as exc:
        logger.warning("Tool '%s' failed: %s", name, exc)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
