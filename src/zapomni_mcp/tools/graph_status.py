"""
GraphStatus MCP Tool - Knowledge Graph Statistics and Health Metrics.

Retrieves comprehensive statistics about the knowledge graph including node counts,
relationship counts, entity types, and overall health status.

Delegates to FalkorDBClient.get_stats() and additional graph queries.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict

import structlog

from zapomni_core.exceptions import DatabaseError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class GraphStatusTool:
    """
    MCP tool for retrieving knowledge graph statistics and health metrics.

    This tool queries the FalkorDB graph for comprehensive statistics including
    node and relationship counts, entity type breakdown, and overall graph health.

    Attributes:
        name: Tool identifier ("graph_status")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation (empty - no params)
        memory_processor: MemoryProcessor instance for accessing database
        logger: Structured logger for operations
    """

    name = "graph_status"
    description = (
        "Get statistics about the knowledge graph including node counts, "
        "relationship counts, entity types, and overall health metrics."
    )
    input_schema = {
        "type": "object",
        "properties": {},  # No properties - tool takes no arguments
        "required": [],  # No required fields
        "additionalProperties": False,
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize GraphStatusTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for accessing database.
                Must be initialized and connected to FalkorDB.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = GraphStatusTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("graph_status_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph_status tool and return graph statistics.

        This method retrieves statistics from the FalkorDB client and formats
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
                        "text": "Formatted graph status string"
                    }
                ],
                "isError": False
            }

        Example:
            >>> tool = GraphStatusTool(memory_processor=processor)
            >>> result = await tool.execute({})
            >>> print(result["isError"])
            False
        """
        log = self.logger.bind(request_id=id(arguments))

        try:
            # Validate arguments (should be empty dict, but accept any)
            if not isinstance(arguments, dict):
                log.warning(
                    "graph_status_invalid_arguments_type",
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

            # Retrieve statistics from database client
            log.info("graph_status_requested")
            stats = await self._get_graph_stats()
            log.debug("graph_stats_retrieved", stats_keys=list(stats.keys()))

            # Format response
            response = self._format_response(stats)
            log.info("graph_status_success", nodes=stats.get("nodes", {}).get("total"))

            return response

        except DatabaseError as e:
            # Database error
            log.error(
                "graph_status_database_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Failed to retrieve graph statistics - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except Exception as e:
            # Unexpected error
            log.error(
                "graph_status_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: An unexpected error occurred getting graph statistics",
                    }
                ],
                "isError": True,
            }

    async def _get_graph_stats(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive graph statistics from the database.

        Returns:
            Dictionary containing nodes, relationships, entity types, and health

        Raises:
            DatabaseError: If database queries fail
        """
        try:
            # Get base statistics from memory processor's database client
            db_client = self.memory_processor.db_client
            if not db_client:
                raise DatabaseError("Database client not initialized")

            # Retrieve base stats
            stats = await db_client.get_stats()

            # Get entity type breakdown
            entity_types = await self._get_entity_types()

            # Calculate health status
            health = self._calculate_health(stats)

            return {
                "nodes": stats.get("nodes", {}),
                "relationships": stats.get("relationships", {}),
                "entity_types": entity_types,
                "health": health,
                "graph_name": db_client.graph_name,
            }

        except Exception as e:
            self.logger.error("get_graph_stats_error", error=str(e))
            raise DatabaseError(f"Failed to retrieve graph statistics: {e}")

    async def _get_entity_types(self) -> Dict[str, int]:
        """
        Get breakdown of entity types in the graph.

        Returns:
            Dictionary with entity type counts

        Raises:
            DatabaseError: If query fails
        """
        try:
            db_client = self.memory_processor.db_client
            if not db_client:
                return {}

            cypher = "MATCH (e:Entity) RETURN e.type AS type, count(e) AS count"
            result = await db_client._execute_cypher(cypher, {})

            entity_types = {}
            for row in result.rows:
                entity_type = row.get("type", "UNKNOWN")
                count = row.get("count", 0)
                entity_types[entity_type] = count

            # Ensure all standard types are present
            standard_types = [
                "TECHNOLOGY",
                "PERSON",
                "ORG",
                "GPE",
                "CONCEPT",
                "PRODUCT",
                "EVENT",
            ]
            for etype in standard_types:
                if etype not in entity_types:
                    entity_types[etype] = 0

            return entity_types

        except Exception as e:
            self.logger.warning("get_entity_types_error", error=str(e))
            # Return empty dict on error - don't fail the whole tool
            return {}

    def _calculate_health(self, stats: Dict[str, Any]) -> str:
        """
        Calculate graph health status based on statistics.

        Health levels:
        - "Healthy": Graph has nodes and entities
        - "Warning": Graph has nodes but no entities
        - "Critical": Graph has no nodes

        Args:
            stats: Statistics from database

        Returns:
            Health status string
        """
        try:
            nodes = stats.get("nodes", {})
            total_nodes = nodes.get("total", 0)
            entity_nodes = nodes.get("entity", 0)

            if total_nodes == 0:
                return "Critical"
            elif entity_nodes == 0:
                return "Warning"
            else:
                return "Healthy"

        except Exception as e:
            self.logger.warning("calculate_health_error", error=str(e))
            return "Unknown"

    def _format_response(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw statistics into MCP response structure.

        Transforms the statistics dictionary into a human-readable formatted
        text response suitable for display.

        Args:
            stats: Statistics dictionary containing nodes, relationships,
                entity_types, and health

        Returns:
            MCP response dictionary with formatted text content
        """
        # Build formatted text output
        lines = ["Knowledge Graph Status:", ""]

        # Nodes section
        nodes = stats.get("nodes", {})
        lines.append("Nodes:")
        lines.append(f"  Total: {nodes.get('total') or 0:,}")
        lines.append(f"  Memories: {nodes.get('memory') or 0:,}")
        lines.append(f"  Chunks: {nodes.get('chunk') or 0:,}")
        lines.append(f"  Entities: {nodes.get('entity') or 0:,}")
        lines.append(f"  Documents: {nodes.get('document') or 0:,}")
        lines.append("")

        # Relationships section
        relationships = stats.get("relationships", {}) or {}
        lines.append("Relationships:")
        lines.append(f"  Total: {relationships.get('total') or 0:,}")
        lines.append(f"  HAS_CHUNK: {relationships.get('has_chunk') or 0:,}")
        lines.append(f"  MENTIONS: {relationships.get('mentions') or 0:,}")
        lines.append(f"  RELATED_TO: {relationships.get('related_to') or 0:,}")
        lines.append("")

        # Entity Types section
        entity_types = stats.get("entity_types", {})
        if entity_types:
            lines.append("Entity Types:")
            # Sort by count descending, then by name
            sorted_types = sorted(entity_types.items(), key=lambda x: (-x[1], x[0]))
            for etype, count in sorted_types:
                if count > 0:
                    lines.append(f"  {etype}: {count:,}")
            lines.append("")

        # Health section
        health = stats.get("health", "Unknown")
        lines.append(f"Graph Health: {health}")

        # Graph name (optional)
        graph_name = stats.get("graph_name")
        if graph_name:
            lines.append(f"Graph Name: {graph_name}")

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
