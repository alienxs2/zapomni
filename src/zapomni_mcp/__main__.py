"""
Entry point for zapomni_mcp MCP server.

This module serves as the main entry point when running:
    python -m zapomni_mcp

Supports two transport modes:
- SSE (default): HTTP-based Server-Sent Events for concurrent connections
- stdio: Standard I/O for single client connections

Usage:
    python -m zapomni_mcp                          # Start with SSE (default)
    python -m zapomni_mcp --transport sse          # Start with SSE explicitly
    python -m zapomni_mcp --transport stdio        # Start with stdio
    python -m zapomni_mcp --host 0.0.0.0 --port 9000  # Custom SSE host/port

CRITICAL: Logging is configured FIRST before importing any other zapomni modules
to prevent "Logging not configured" errors from module-level logger initialization.

The initialization sequence is:
1. Configure logging service
2. Parse command-line arguments
3. Import configuration and core modules
4. Initialize MemoryProcessor with database, chunker, and embedder
5. Create MCPServer and register all tools
6. Start the async server loop (SSE or stdio based on transport)

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import argparse
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

from zapomni_core.chunking import SemanticChunker

# Now safe to import other zapomni modules (after logging is configured)
from zapomni_core.config import ZapomniSettings
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

# EntityExtractor is loaded lazily by MemoryProcessor when needed
from zapomni_core.memory_processor import MemoryProcessor, ProcessorConfig
from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_db import FalkorDBClient
from zapomni_db.pool_config import PoolConfig, RetryConfig
from zapomni_mcp.config import Settings, SSEConfig
from zapomni_mcp.server import MCPServer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Zapomni MCP Server - Memory and Knowledge Graph for AI Assistants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m zapomni_mcp                          # Start with SSE (default)
  python -m zapomni_mcp --transport sse          # Start with SSE explicitly
  python -m zapomni_mcp --transport stdio        # Start with stdio
  python -m zapomni_mcp --host 0.0.0.0 --port 9000  # Custom SSE host/port

Environment Variables:
  FALKORDB_HOST              FalkorDB host (default: localhost)
  FALKORDB_PORT              FalkorDB port (default: 6381)
  OLLAMA_BASE_URL            Ollama base URL (default: http://localhost:11434)
  OLLAMA_EMBEDDING_MODEL     Embedding model (default: nomic-embed-text)
  ZAPOMNI_SSE_HOST           SSE server host (default: 127.0.0.1)
  ZAPOMNI_SSE_PORT           SSE server port (default: 8000)
  ZAPOMNI_SSE_CORS_ORIGINS   Comma-separated CORS origins (default: *)
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport type: 'sse' for HTTP Server-Sent Events (default), 'stdio' for standard I/O",
    )

    parser.add_argument(
        "--host",
        default=None,
        help="SSE server host (default: 127.0.0.1, or ZAPOMNI_SSE_HOST env var)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="SSE server port (default: 8000, or ZAPOMNI_SSE_PORT env var)",
    )

    parser.add_argument(
        "--cors-origins",
        default=None,
        help="Comma-separated CORS origins for SSE (default: *, or ZAPOMNI_SSE_CORS_ORIGINS env var)",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
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
    8. Runs the async server loop (SSE or stdio based on args.transport)

    Args:
        args: Parsed command-line arguments

    Environment Variables:
        FALKORDB_HOST: FalkorDB host (default: "localhost")
        FALKORDB_PORT: FalkorDB port (default: "6381")
        OLLAMA_BASE_URL: Ollama base URL (default: "http://localhost:11434")
        OLLAMA_EMBEDDING_MODEL: Embedding model name (default: "nomic-embed-text")

    Raises:
        ValueError: If any configuration is invalid
        ConnectionError: If unable to connect to FalkorDB or Ollama
        RuntimeError: If initialization fails
    """
    logger.info(
        "Starting MCP server initialization",
        transport=args.transport,
        log_level="INFO",
    )

    try:
        # STAGE 1: Load configuration from environment
        logger.info("Loading configuration from environment")
        settings = ZapomniSettings(
            falkordb_host=os.getenv("FALKORDB_HOST", "localhost"),
            falkordb_port=int(os.getenv("FALKORDB_PORT", "6381")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        )
        logger.info(
            "Configuration loaded",
            falkordb_host=settings.falkordb_host,
            falkordb_port=settings.falkordb_port,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )

        # STAGE 2: Initialize database client with async connection pool
        logger.info("Initializing FalkorDB client with connection pooling")

        # Create pool configuration from settings
        pool_config = PoolConfig(
            min_size=settings.falkordb_pool_min_size,
            max_size=settings.falkordb_pool_max_size,
            timeout=settings.falkordb_pool_timeout,
            socket_timeout=settings.falkordb_socket_timeout,
            health_check_interval=settings.falkordb_health_check_interval,
        )

        # Create retry configuration from settings
        retry_config = RetryConfig(
            max_retries=settings.falkordb_max_retries,
            initial_delay=settings.falkordb_retry_initial_delay,
            max_delay=settings.falkordb_retry_max_delay,
        )

        db_client = FalkorDBClient(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            graph_name=settings.graph_name,
            password=(
                settings.falkordb_password.get_secret_value()
                if settings.falkordb_password
                else None
            ),
            pool_config=pool_config,
            retry_config=retry_config,
        )

        # Initialize async connection pool
        await db_client.init_async()

        logger.info(
            "FalkorDB connection pool initialized",
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            pool_max_size=pool_config.max_size,
            pool_timeout=pool_config.timeout,
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

        # STAGE 5: Initialize MemoryProcessor
        # Note: EntityExtractor and GraphBuilder are loaded LAZILY on first use
        # This speeds up MCP server startup significantly (~3 sec faster)
        logger.info("Initializing MemoryProcessor (SpaCy/EntityExtractor loaded lazily)")

        # Read feature flags from environment (all enabled by default except semantic cache)
        enable_hybrid_search = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
        enable_knowledge_graph = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
        enable_code_indexing = os.getenv("ENABLE_CODE_INDEXING", "true").lower() == "true"
        enable_semantic_cache = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"

        # Create ProcessorConfig with feature flags
        processor_config = ProcessorConfig(
            enable_cache=enable_semantic_cache,
            enable_extraction=enable_knowledge_graph,
            enable_graph=enable_knowledge_graph,
            enable_llm_refinement=True,
            search_mode="hybrid" if enable_hybrid_search else "vector",
        )

        logger.info(
            "Feature flags loaded",
            hybrid_search=enable_hybrid_search,
            knowledge_graph=enable_knowledge_graph,
            code_indexing=enable_code_indexing,
            semantic_cache=enable_semantic_cache,
            search_mode=processor_config.search_mode,
        )

        processor = MemoryProcessor(
            db_client=db_client,
            chunker=chunker,
            embedder=embedder,
            extractor=None,  # Lazy loading - SpaCy loads on first build_graph call
            config=processor_config,
        )
        logger.info("MemoryProcessor initialized")

        # Attach code_indexer if code indexing is enabled
        if enable_code_indexing:
            logger.info("Initializing CodeRepositoryIndexer")
            processor.code_indexer = CodeRepositoryIndexer()
            logger.info("CodeRepositoryIndexer attached to MemoryProcessor")

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

        # STAGE 8: Run the server with selected transport
        if args.transport == "stdio":
            # Standard I/O transport (original behavior)
            logger.info(
                "Starting MCP server with stdio transport",
                transport="stdio",
            )
            await server.run()
        else:
            # SSE transport (new default)
            # Load SSE config from environment with CLI overrides
            sse_config = SSEConfig.from_env()

            # Apply CLI overrides
            host = args.host if args.host is not None else sse_config.host
            port = args.port if args.port is not None else sse_config.port

            # Parse CORS origins
            if args.cors_origins is not None:
                cors_origins = [origin.strip() for origin in args.cors_origins.split(",")]
            else:
                cors_origins = sse_config.cors_origins

            logger.info(
                "Starting MCP server with SSE transport",
                transport="sse",
                host=host,
                port=port,
                cors_origins=cors_origins,
            )
            await server.run_sse(host=host, port=port, cors_origins=cors_origins)

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
    args = parse_args()
    asyncio.run(main(args))
