"""AI analysis MCP tools."""

from typing import Optional

from ..services.data_service import DataService
from ..utils.errors import MCPError, DataNotFoundError
from ..utils.validators import validate_top_n, validate_theme


class AIAnalysisTools:
    """Expose AI analysis results via MCP tools."""

    def __init__(self, project_root: Optional[str] = None):
        self.data_service = DataService(project_root)

    def get_latest_ai_analysis(self) -> dict:
        try:
            analysis = self.data_service.get_ai_analysis()
            return {"success": True, "analysis": analysis}
        except DataNotFoundError as e:
            return {"success": False, "error": e.to_dict()}
        except MCPError as e:
            return {"success": False, "error": e.to_dict()}
        except Exception as exc:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                },
            }

    def get_top_events(self, top_n: Optional[int] = None) -> dict:
        try:
            top_n = validate_top_n(top_n, default=5)
            result = self.data_service.get_top_ai_events(top_n)
            return {"success": True, **result}
        except MCPError as e:
            return {"success": False, "error": e.to_dict()}
        except Exception as exc:
            return {
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
            }

    def search_events_by_theme(self, theme: Optional[str]) -> dict:
        try:
            theme = validate_theme(theme)
            result = self.data_service.search_ai_events_by_theme(theme)
            return {"success": True, **result}
        except MCPError as e:
            return {"success": False, "error": e.to_dict()}
        except Exception as exc:
            return {
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
            }
