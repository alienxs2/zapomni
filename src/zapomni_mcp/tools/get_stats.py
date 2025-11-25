"""
GetStats MCP Tool - Full Implementation.

Retrieves memory system statistics including memory count, chunks, database size, and performance metrics.
Delegates to MemoryProcessor for all statistics operations.

Also includes SSE transport metrics when available (active connections, total requests, etc.).

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import structlog

from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import DatabaseError

if TYPE_CHECKING:
    from zapomni_mcp.session_manager import SessionManager


logger = structlog.get_logger(__name__)


class GetStatsTool:
    """
    MCP tool for retrieving system statistics.

    This tool delegates to MemoryProcessor.get_stats() to retrieve
    comprehensive system statistics and formats them for display.
    When running in SSE mode, also includes transport metrics.

    Attributes:
        name: Tool identifier ("get_stats")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation (empty - no params)
        memory_processor: MemoryProcessor instance for retrieving stats
        session_manager: Optional SessionManager for SSE metrics
        logger: Structured logger for operations
    """

    name = "get_stats"
    description = (
        "Get statistics about the memory system including total memories, "
        "chunks, database size, and performance metrics."
    )
    input_schema = {
        "type": "object",
        "properties": {},  # No properties - tool takes no arguments
        "required": [],  # No required fields
        "additionalProperties": False,
    }

    def __init__(
        self,
        memory_processor: MemoryProcessor,
        session_manager: Optional["SessionManager"] = None,
        mcp_server: Optional[Any] = None,
    ) -> None:
        """
        Initialize GetStatsTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for retrieving statistics.
                Must be initialized and connected to database.
            session_manager: Optional SessionManager for SSE transport metrics.
                When provided, includes connection metrics in response.
            mcp_server: Optional MCPServer instance to dynamically access session_manager.
                If session_manager is not provided, will try to access _session_manager
                attribute from mcp_server at runtime (for SSE mode).

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = GetStatsTool(memory_processor=processor)

            # With SSE metrics:
            >>> tool = GetStatsTool(memory_processor=processor, session_manager=session_mgr)

            # Or via MCPServer for dynamic access:
            >>> tool = GetStatsTool(memory_processor=processor, mcp_server=server)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self._session_manager = session_manager
        self._mcp_server = mcp_server
        self.logger = logger.bind(tool=self.name)

        self.logger.info(
            "get_stats_tool_initialized",
            has_session_manager=session_manager is not None,
            has_mcp_server=mcp_server is not None,
        )

    @property
    def session_manager(self) -> Optional["SessionManager"]:
        """
        Get session manager (either direct or via MCPServer).

        Returns:
            SessionManager if available, None otherwise
        """
        if self._session_manager is not None:
            return self._session_manager
        if self._mcp_server is not None and hasattr(self._mcp_server, '_session_manager'):
            return self._mcp_server._session_manager
        return None

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_stats tool and return system statistics.

        This method retrieves statistics from the memory processor and formats
        them into an MCP-compliant response. Since this tool requires no
        parameters, the arguments dict is expected to be empty (but is
        accepted for consistency with the MCP tool interface).

        Args:
            arguments: Dictionary of arguments (should be empty {}).
                Any provided arguments are ignored as per tool spec.

        Returns:
            Dictionary in MCP response format:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Formatted statistics string"
                    }
                ],
                "isError": False
            }

        Example:
            >>> tool = GetStatsTool(memory_processor=processor)
            >>> result = await tool.execute({})
            >>> print(result["isError"])
            False
        """
        log = self.logger.bind(request_id=id(arguments))

        try:
            # Validate arguments (should be empty dict, but accept any)
            if not isinstance(arguments, dict):
                log.warning(
                    "get_stats_invalid_arguments_type",
                    type=type(arguments).__name__,
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: Arguments must be a dictionary (expected empty {})",
                        }
                    ],
                    "isError": True,
                }

            # Retrieve statistics from memory processor
            log.info("get_stats_requested")
            stats = await self.memory_processor.get_stats()
            log.debug("stats_retrieved", stats_keys=list(stats.keys()))

            # Format response
            response = self._format_response(stats)
            log.info("get_stats_success", stats_keys=list(stats.keys()))

            return response

        except DatabaseError as e:
            # Database error
            log.error(
                "get_stats_database_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Failed to retrieve statistics - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except Exception as e:
            # Unexpected error
            log.error(
                "get_stats_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: An unexpected error occurred while retrieving statistics",
                    }
                ],
                "isError": True,
            }

    def _format_response(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw statistics into MCP response structure.

        Transforms the statistics dictionary from memory processor into a
        human-readable formatted text response suitable for display.

        Args:
            stats: Statistics dictionary from MemoryProcessor.get_stats()
                Expected keys:
                - total_memories: int (required)
                - total_chunks: int (required)
                - database_size_mb: float (required)
                - avg_chunks_per_memory: float (required)
                - avg_query_latency_ms: int (optional)
                - cache_hit_rate: float (optional, 0.0-1.0)
                - total_entities: int (optional)
                - total_relationships: int (optional)

        Returns:
            MCP response dictionary with formatted text content
        """
        # Build formatted text output
        lines = ["Memory System Statistics:"]

        # Required fields (always present)
        lines.append(f"Total Memories: {stats.get('total_memories', 0):,}")
        lines.append(f"Total Chunks: {stats.get('total_chunks', 0):,}")
        lines.append(f"Database Size: {stats.get('database_size_mb', 0.0):.2f} MB")
        lines.append(
            f"Average Chunks per Memory: {stats.get('avg_chunks_per_memory', 0.0):.1f}"
        )

        # Optional fields (only if present and not None)
        if "total_entities" in stats and stats["total_entities"] is not None:
            lines.append(f"Total Entities: {stats['total_entities']:,}")

        if "total_relationships" in stats and stats["total_relationships"] is not None:
            lines.append(f"Total Relationships: {stats['total_relationships']:,}")

        if "cache_hit_rate" in stats and stats["cache_hit_rate"] is not None:
            hit_rate_pct = stats["cache_hit_rate"] * 100
            lines.append(f"Cache Hit Rate: {hit_rate_pct:.1f}%")

        if "avg_query_latency_ms" in stats and stats["avg_query_latency_ms"] is not None:
            lines.append(f"Avg Query Latency: {stats['avg_query_latency_ms']:.1f} ms")

        if "oldest_memory_date" in stats and stats["oldest_memory_date"] is not None:
            lines.append(f"Oldest Memory: {stats['oldest_memory_date']}")

        if "newest_memory_date" in stats and stats["newest_memory_date"] is not None:
            lines.append(f"Newest Memory: {stats['newest_memory_date']}")

        # Add SSE transport metrics if session manager is available
        if self.session_manager is not None:
            lines.append("")  # Blank line separator
            lines.append("SSE Transport Metrics:")
            metrics = self.session_manager.get_metrics()
            lines.append(f"Active Connections: {metrics.current_active_connections}")
            lines.append(f"Peak Connections: {metrics.peak_connections}")
            lines.append(f"Total Connections Created: {metrics.total_connections_created}")
            lines.append(f"Total Connections Closed: {metrics.total_connections_closed}")
            lines.append(f"Total Requests Processed: {metrics.total_requests_processed}")
            lines.append(f"Total Errors: {metrics.total_errors}")

        # Join into single text block
        formatted_text = "\n".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted_text,
                }
            ],
            "isError": False,
        }
