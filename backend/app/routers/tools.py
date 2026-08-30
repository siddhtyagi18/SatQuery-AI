from fastapi import APIRouter

from ..schemas import ToolDefinition
from ..services.tool_registry import list_tools

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("", response_model=list[ToolDefinition])
def get_tools():
    return list_tools()
