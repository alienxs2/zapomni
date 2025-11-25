"""
Entry point for zapomni_mcp MCP server.

This module serves as the main entry point when running:
    python -m zapomni_mcp.server

CRITICAL: Logging is configured FIRST before importing any other zapomni modules
to prevent "Logging not configured" errors from module-level logger initialization.

The initialization sequence is:
1. Configure logging service
2. Import configuration and core modules
3. Initialize MemoryProcessor with database, chunker, and embedder
4. Create MCPServer and register all tools
5. Start the async server loop

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import os
import sys
from typing import Optional

# CRITICAL: Configure logging BEFORE any other zapomni imports
# This prevents "Logging not configured" errors from modules that call
# get_logger(__name__) at module level (e.g., embedding_cache.py, memory_processor.py)
from zapomni_core.logging_service import LoggingService

LoggingService.configure_logging(level="INFO", format="json")

# Get logger after logging is configured
logger = LoggingService.get_logger(__name__)

# Now safe to import other zapomni modules (after logging is configured)
from zapomni_core.config import ZapomniSettings
from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.extractors.entity_extractor import EntityExtractor
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_db import FalkorDBClient
from zapomni_mcp.server import MCPServer
from zapomni_mcp.config import Settings


async def main() -> None:
    """
    Main entry point for MCP server.

    Orchestrates the complete initialization sequence:
    1. Reads configuration from environment variables with sensible defaults
    2. Initializes database client (FalkorDB)
    3. Initializes text chunker (SemanticChunker)
    4. Initializes embedding generator (OllamaEmbedder)
    5. Creates MemoryProcessor that coordinates the pipeline
    6. Creates MCPServer with the processor
    7. Registers all standard tools
    8. Runs the async server loop

    Environment Variables:
        FALKORDB_HOST: FalkorDB host (default: "localhost")
        FALKORDB_PORT: FalkorDB port (default: "6379")
        OLLAMA_BASE_URL: Ollama base URL (default: "http://localhost:11434")
        OLLAMA_EMBEDDING_MODEL: Embedding model name (default: "nomic-embed-text")

    Raises:
        ValueError: If any configuration is invalid
        ConnectionError: If unable to connect to FalkorDB or Ollama
        RuntimeError: If initialization fails
    """
    logger.info(
        "Starting MCP server initialization",
        log_level="INFO",
    )

    try:
        # STAGE 1: Load configuration from environment
        logger.info("Loading configuration from environment")
        settings = ZapomniSettings(
            falkordb_host=os.getenv("FALKORDB_HOST", "localhost"),
            falkordb_port=int(os.getenv("FALKORDB_PORT", "6379")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
        )
        logger.info(
            "Configuration loaded",
            falkordb_host=settings.falkordb_host,
            falkordb_port=settings.falkordb_port,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )

        # STAGE 2: Initialize database client
        logger.info("Initializing FalkorDB client")
        db_client = FalkorDBClient(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            graph_name=settings.graph_name,
            password=settings.falkordb_password.get_secret_value() if settings.falkordb_password else None,
            pool_size=settings.falkordb_pool_size,
        )
        logger.info(
            "FalkorDB client initialized",
            host=settings.falkordb_host,
            port=settings.falkordb_port,
        )

        # STAGE 3: Initialize text chunker
        logger.info("Initializing semantic chunker")
        chunker = SemanticChunker(
            chunk_size=settings.max_chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info(
            "Semantic chunker initialized",
            chunk_size=settings.max_chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # STAGE 4: Initialize embedding generator
        logger.info("Initializing Ollama embedder")
        embedder = OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embedding_model,
            timeout=settings.ollama_embedding_timeout,
        )
        logger.info(
            "Ollama embedder initialized",
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embedding_model,
        )

        # STAGE 5: Initialize EntityExtractor for knowledge graph building
        logger.info("Initializing EntityExtractor")
        import spacy
        spacy_model = spacy.load("en_core_web_sm")
        extractor = EntityExtractor(spacy_model=spacy_model)
        logger.info("EntityExtractor initialized")

        # STAGE 6: Initialize MemoryProcessor
        logger.info("Initializing MemoryProcessor")
        processor = MemoryProcessor(
            db_client=db_client,
            chunker=chunker,
            embedder=embedder,
            extractor=extractor,
            config=None,  # Use defaults
        )
        logger.info("MemoryProcessor initialized")

        # STAGE 6: Create MCP server
        logger.info("Creating MCPServer")
        server_config = Settings()
        server = MCPServer(core_engine=processor, config=server_config)
        logger.info(
            "MCPServer created",
            server_name=server_config.server_name,
            version=server_config.version,
        )

        # STAGE 7: Register all tools
        logger.info("Registering MCP tools")
        server.register_all_tools(memory_processor=processor)
        logger.info(
            "All MCP tools registered successfully",
            tool_count=len(server._tools),
            tools=list(server._tools.keys()),
        )

        # STAGE 8: Run the server
        logger.info(
            "Starting MCP server loop",
            transport="stdio",
        )
        await server.run()

    except ValueError as e:
        logger.error(
            "Configuration error",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)
    except ConnectionError as e:
        logger.error(
            "Connection error",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)
    except RuntimeError as e:
        logger.error(
            "Runtime error",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        sys.exit(0)
    except Exception as e:
        logger.error(
            "Unexpected error during initialization",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
