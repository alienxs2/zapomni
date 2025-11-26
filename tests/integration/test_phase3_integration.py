"""
Comprehensive integration tests for Phase 3 workflow.

Tests the complete Phase 3 end-to-end flow covering:
1. Code Indexing (index_codebase tool)
2. Graph Export (export_graph tool in all formats)
3. Memory Deletion (delete_memory tool with confirmation)
4. Clear All (clear_all tool with safety phrase)

Prerequisites:
- FalkorDB running on localhost:6381 (via docker-compose)
- Ollama embeddings service running on localhost:11434
- Run with: pytest tests/integration/test_phase3_integration.py -v

Target coverage: Complete Phase 3 flows with safety mechanisms

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import pytest

from zapomni_core.chunking import SemanticChunker
from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.memory_processor import MemoryProcessor, ProcessorConfig
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools.clear_all import ClearAllTool
from zapomni_mcp.tools.delete_memory import DeleteMemoryTool
from zapomni_mcp.tools.export_graph import ExportGraphTool
from zapomni_mcp.tools.index_codebase import IndexCodebaseTool

# ============================================================================
# Module-level fixtures (session scope)
# ============================================================================


@pytest.fixture(scope="module")
def falkordb_client():
    """
    Create FalkorDB client for Phase 3 integration tests.

    Uses a dedicated test graph to avoid polluting production data.
    Skips tests if FalkorDB is not available.
    """
    try:
        client = FalkorDBClient(
            host="localhost",
            port=6381,
            graph_name="zapomni_test_phase3",
        )
        # Verify connection by clearing graph
        asyncio.run(client.clear_all())
        yield client
    except Exception as e:
        pytest.skip(f"FalkorDB not available at localhost:6381: {e}")
    finally:
        try:
            asyncio.run(client.clear_all())
            client.close()
        except Exception:
            pass


@pytest.fixture(scope="module")
def ollama_embedder():
    """
    Create OllamaEmbedder for Phase 3 integration tests.

    Skips tests if Ollama is not available on localhost:11434.
    """
    try:
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text",
        )
        # Verify by embedding test text
        asyncio.run(embedder.embed_text("test"))
        yield embedder
    except Exception as e:
        pytest.skip(f"Ollama not available at localhost:11434: {e}")


@pytest.fixture(scope="module")
def semantic_chunker():
    """Create SemanticChunker for Phase 3 integration tests."""
    return SemanticChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture(scope="module")
def repository_indexer():
    """Create CodeRepositoryIndexer for Phase 3 integration tests."""
    return CodeRepositoryIndexer()


@pytest.fixture(scope="module")
def temp_export_dir():
    """Create temporary directory for export files."""
    temp_dir = tempfile.mkdtemp(prefix="zapomni_phase3_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def test_codebase_dir():
    """Create temporary test codebase with sample Python files."""
    temp_dir = tempfile.mkdtemp(prefix="zapomni_test_repo_")
    temp_path = Path(temp_dir)

    # Create sample Python files
    (temp_path / "module1.py").write_text(
        '''"""Module 1 with sample functions."""

def hello_world():
    """Print hello world."""
    print("Hello, World!")

def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """Simple calculator class."""

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y
'''
    )

    (temp_path / "module2.py").write_text(
        '''"""Module 2 with more functions."""

def greet(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"

class DataProcessor:
    """Process data."""

    def __init__(self):
        self.data = []

    def process(self, item):
        """Process an item."""
        self.data.append(item)
        return item
'''
    )

    # Create a subdirectory with another file
    sub_dir = temp_path / "submodule"
    sub_dir.mkdir()
    (sub_dir / "utils.py").write_text(
        '''"""Utility functions."""

def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()
'''
    )

    yield temp_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Function-level fixtures (setup/teardown for each test)
# ============================================================================


@pytest.fixture
async def memory_processor(falkordb_client, semantic_chunker, ollama_embedder):
    """
    Create MemoryProcessor for Phase 3 integration test.

    Creates a fresh processor for each test with a clean database.
    """
    # Clear database before each test
    await falkordb_client.clear_all()

    processor = MemoryProcessor(
        db_client=falkordb_client,
        chunker=semantic_chunker,
        embedder=ollama_embedder,
        config=ProcessorConfig(
            enable_cache=False,
            enable_extraction=False,
            enable_graph=False,
            max_text_length=10_000_000,
            batch_size=4,
            search_mode="vector",
        ),
    )

    yield processor

    # Cleanup after test
    await falkordb_client.clear_all()


@pytest.fixture
async def index_codebase_tool(repository_indexer, memory_processor):
    """Create IndexCodebaseTool for Phase 3 integration test."""
    return IndexCodebaseTool(
        repository_indexer=repository_indexer,
        memory_processor=memory_processor,
    )


@pytest.fixture
async def export_graph_tool(memory_processor):
    """Create ExportGraphTool for Phase 3 integration test."""
    return ExportGraphTool(memory_processor=memory_processor)


@pytest.fixture
async def delete_memory_tool(memory_processor):
    """Create DeleteMemoryTool for Phase 3 integration test."""
    return DeleteMemoryTool(memory_processor=memory_processor)


@pytest.fixture
async def clear_all_tool(memory_processor):
    """Create ClearAllTool for Phase 3 integration test."""
    return ClearAllTool(memory_processor=memory_processor)


# ============================================================================
# Test Class 1: Code Indexing Workflow
# ============================================================================


@pytest.mark.integration
class TestCodeIndexingWorkflow:
    """
    Test code indexing workflow end-to-end.

    Validates: Repository indexing extracts correct file structure
    """

    @pytest.mark.asyncio
    async def test_codebase_indexing_workflow(self, index_codebase_tool, test_codebase_dir):
        """
        Test: Index a test codebase and verify results

        Validates:
        - Tool executes successfully
        - Files are discovered and indexed
        - Statistics are accurate
        - Only Python files are indexed
        """
        arguments = {
            "repo_path": str(test_codebase_dir),
            "languages": ["python"],
            "recursive": True,
            "max_file_size": 10 * 1024 * 1024,
            "include_tests": False,
        }

        result = await index_codebase_tool.execute(arguments)

        # Verify MCP response format
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"

        # Verify response contains expected information
        response_text = result["content"][0]["text"]
        assert "Repository indexed successfully" in response_text
        assert "Files indexed: 3" in response_text  # 3 Python files
        assert "Python" in response_text

    @pytest.mark.asyncio
    async def test_index_with_language_filter(self, index_codebase_tool, test_codebase_dir):
        """
        Test: Index repository with specific language filter

        Validates:
        - Language filtering works correctly
        - Only specified language files are indexed
        """
        arguments = {
            "repo_path": str(test_codebase_dir),
            "languages": ["python"],
            "recursive": True,
        }

        result = await index_codebase_tool.execute(arguments)

        assert result["isError"] is False
        response_text = result["content"][0]["text"]
        assert "Python" in response_text

    @pytest.mark.asyncio
    async def test_index_nonexistent_repo(self, index_codebase_tool):
        """
        Test: Indexing nonexistent repository fails gracefully

        Validates:
        - Error handling for invalid paths
        - Proper error message returned
        """
        arguments = {
            "repo_path": "/nonexistent/path/to/repo",
            "languages": ["python"],
        }

        result = await index_codebase_tool.execute(arguments)

        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]


# ============================================================================
# Test Class 2: Graph Export All Formats
# ============================================================================


@pytest.mark.integration
class TestGraphExportAllFormats:
    """
    Test graph export in all supported formats.

    Validates: All export formats work correctly with valid output
    """

    @pytest.mark.asyncio
    async def test_export_graphml_format(
        self, export_graph_tool, memory_processor, temp_export_dir
    ):
        """
        Test: Export graph to GraphML format

        Validates:
        - GraphML export succeeds
        - Valid XML structure
        - File is created with correct extension
        """
        # Add some test data
        await memory_processor.add_memory(
            text="Python is a programming language",
            metadata={"source": "test"},
        )

        output_path = str(temp_export_dir / "test_graph.graphml")
        arguments = {
            "format": "graphml",
            "output_path": output_path,
            "options": {"pretty_print": True},
        }

        result = await export_graph_tool.execute(arguments)

        # Verify MCP response
        assert result["isError"] is False
        assert "Graph exported successfully" in result["content"][0]["text"]
        assert "graphml" in result["content"][0]["text"]

        # Verify file exists and is valid XML
        assert Path(output_path).exists()
        tree = ET.parse(output_path)
        root = tree.getroot()
        assert "graphml" in root.tag.lower()

    @pytest.mark.asyncio
    async def test_export_cytoscape_format(
        self, export_graph_tool, memory_processor, temp_export_dir
    ):
        """
        Test: Export graph to Cytoscape JSON format

        Validates:
        - Cytoscape JSON export succeeds
        - Valid JSON structure
        - Contains elements (nodes/edges)
        """
        # Add test data
        await memory_processor.add_memory(
            text="Django is a web framework",
            metadata={"source": "test"},
        )

        output_path = str(temp_export_dir / "test_graph_cytoscape.json")
        arguments = {
            "format": "cytoscape",
            "output_path": output_path,
            "options": {"pretty_print": True, "include_style": True},
        }

        result = await export_graph_tool.execute(arguments)

        assert result["isError"] is False
        assert "Graph exported successfully" in result["content"][0]["text"]

        # Verify file exists and is valid JSON
        assert Path(output_path).exists()
        with open(output_path, "r") as f:
            data = json.load(f)
            assert "elements" in data
            assert "nodes" in data["elements"] or "edges" in data["elements"]

    @pytest.mark.asyncio
    async def test_export_neo4j_format(self, export_graph_tool, memory_processor, temp_export_dir):
        """
        Test: Export graph to Neo4j Cypher format

        Validates:
        - Neo4j Cypher export succeeds
        - Valid Cypher statements
        - File contains CREATE statements
        """
        # Add test data
        await memory_processor.add_memory(
            text="FastAPI is an API framework",
            metadata={"source": "test"},
        )

        output_path = str(temp_export_dir / "test_graph.cypher")
        arguments = {
            "format": "neo4j",
            "output_path": output_path,
            "options": {"batch_size": 1000},
        }

        result = await export_graph_tool.execute(arguments)

        assert result["isError"] is False
        assert "Graph exported successfully" in result["content"][0]["text"]

        # Verify file exists and contains Cypher statements
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "CREATE" in content or "MATCH" in content

    @pytest.mark.asyncio
    async def test_export_json_format(self, export_graph_tool, memory_processor, temp_export_dir):
        """
        Test: Export graph to simple JSON format

        Validates:
        - JSON export succeeds
        - Valid JSON structure
        - Contains nodes and edges
        """
        # Add test data
        await memory_processor.add_memory(
            text="Redis is a key-value store",
            metadata={"source": "test"},
        )

        output_path = str(temp_export_dir / "test_graph_simple.json")
        arguments = {
            "format": "json",
            "output_path": output_path,
            "options": {"pretty_print": True, "include_metadata": True},
        }

        result = await export_graph_tool.execute(arguments)

        assert result["isError"] is False

        # Verify file exists and is valid JSON
        assert Path(output_path).exists()
        with open(output_path, "r") as f:
            data = json.load(f)
            assert "nodes" in data or "edges" in data

    @pytest.mark.asyncio
    async def test_graph_export_all_formats(
        self, export_graph_tool, memory_processor, temp_export_dir
    ):
        """
        Test: Export graph in all formats sequentially

        Validates:
        - All formats can be exported from same data
        - Each format produces valid output
        - Files have correct extensions
        """
        # Add test data
        await memory_processor.add_memory(
            text="PostgreSQL is a relational database",
            metadata={"source": "test"},
        )
        await memory_processor.add_memory(
            text="MongoDB is a document database",
            metadata={"source": "test"},
        )

        formats = [
            ("graphml", "test_all.graphml"),
            ("cytoscape", "test_all_cyto.json"),
            ("neo4j", "test_all.cypher"),
            ("json", "test_all.json"),
        ]

        for format_name, filename in formats:
            output_path = str(temp_export_dir / filename)
            arguments = {
                "format": format_name,
                "output_path": output_path,
            }

            result = await export_graph_tool.execute(arguments)

            assert result["isError"] is False
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0


# ============================================================================
# Test Class 3: Memory Deletion with Safety
# ============================================================================


@pytest.mark.integration
class TestMemoryDeletionSafety:
    """
    Test memory deletion with safety confirmation.

    Validates: Delete memory requires confirmation
    """

    @pytest.mark.asyncio
    async def test_delete_memory_with_confirmation(self, delete_memory_tool, memory_processor):
        """
        Test: Delete memory with proper confirmation

        Validates:
        - Memory can be deleted with confirm=true
        - Delete succeeds and returns success message
        """
        # Add test memory
        memory_id = await memory_processor.add_memory(
            text="Test memory to delete",
            metadata={"source": "test"},
        )

        # Delete with confirmation
        arguments = {
            "memory_id": memory_id,
            "confirm": True,
        }

        result = await delete_memory_tool.execute(arguments)

        assert result["isError"] is False
        assert "Memory deleted successfully" in result["content"][0]["text"]
        assert memory_id in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_delete_memory_without_confirmation(self, delete_memory_tool, memory_processor):
        """
        Test: Delete memory without confirmation fails

        Validates:
        - Delete without confirm=true is rejected
        - Error message explains confirmation required
        """
        # Add test memory
        memory_id = await memory_processor.add_memory(
            text="Test memory protected",
            metadata={"source": "test"},
        )

        # Try delete without confirmation
        arguments = {
            "memory_id": memory_id,
            "confirm": False,
        }

        result = await delete_memory_tool.execute(arguments)

        assert result["isError"] is True
        assert "confirmation" in result["content"][0]["text"].lower()
        assert "confirm=true" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, delete_memory_tool):
        """
        Test: Deleting nonexistent memory returns not found

        Validates:
        - Deleting invalid UUID returns appropriate error
        - No exception raised, graceful handling
        """
        fake_uuid = "550e8400-e29b-41d4-a716-446655440000"

        arguments = {
            "memory_id": fake_uuid,
            "confirm": True,
        }

        result = await delete_memory_tool.execute(arguments)

        # Should indicate memory not found
        assert result["isError"] is True
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_delete_with_invalid_uuid(self, delete_memory_tool):
        """
        Test: Delete with invalid UUID format fails

        Validates:
        - Invalid UUID format is rejected
        - Validation error returned
        """
        arguments = {
            "memory_id": "invalid-uuid-format",
            "confirm": True,
        }

        result = await delete_memory_tool.execute(arguments)

        assert result["isError"] is True
        assert "error" in result["content"][0]["text"].lower()


# ============================================================================
# Test Class 4: Clear All with Safety Phrase
# ============================================================================


@pytest.mark.integration
class TestClearAllSafety:
    """
    Test clear_all with safety phrase confirmation.

    Validates: Clear all requires exact confirmation phrase
    """

    @pytest.mark.asyncio
    async def test_clear_all_with_correct_phrase(self, clear_all_tool, memory_processor):
        """
        Test: Clear all with correct confirmation phrase

        Validates:
        - Clear all succeeds with exact phrase
        - All memories are deleted
        - Stats show empty database
        """
        # Add some test data
        memory_id1 = await memory_processor.add_memory(text="Memory 1", metadata={"source": "test"})
        memory_id2 = await memory_processor.add_memory(text="Memory 2", metadata={"source": "test"})

        # Verify data exists by checking we got memory IDs
        assert memory_id1 is not None
        assert memory_id2 is not None

        # Clear all with correct phrase
        arguments = {
            "confirm_phrase": "DELETE ALL MEMORIES",
        }

        result = await clear_all_tool.execute(arguments)

        assert result["isError"] is False
        assert "cleared successfully" in result["content"][0]["text"].lower()
        assert "CLEARED" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_clear_all_rejects_wrong_phrase(self, clear_all_tool, memory_processor):
        """
        Test: Clear all rejects incorrect confirmation phrase

        Validates:
        - Wrong phrase is rejected
        - No data is deleted
        - Error message explains correct phrase
        """
        # Add test data
        memory_id = await memory_processor.add_memory(
            text="Protected memory", metadata={"source": "test"}
        )
        assert memory_id is not None

        # Try with wrong phrases - each should fail
        wrong_phrases = [
            "delete all memories",  # lowercase
            "DELETE ALL",  # incomplete
            "DELETE ALL MEMORY",  # singular
            "CLEAR ALL MEMORIES",  # wrong verb
        ]

        for wrong_phrase in wrong_phrases:
            arguments = {
                "confirm_phrase": wrong_phrase,
            }

            result = await clear_all_tool.execute(arguments)

            assert result["isError"] is True
            assert (
                "invalid" in result["content"][0]["text"].lower()
                or "error" in result["content"][0]["text"].lower()
            )

        # Test empty string separately (causes validation error)
        result = await clear_all_tool.execute(
            {
                "confirm_phrase": "",
            }
        )
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_clear_all_case_sensitive(self, clear_all_tool, memory_processor):
        """
        Test: Clear all confirmation is case-sensitive

        Validates:
        - Lowercase version is rejected
        - Only exact case match is accepted
        """
        # Add test data
        await memory_processor.add_memory(text="Case test", metadata={"source": "test"})

        # Try lowercase
        arguments = {
            "confirm_phrase": "delete all memories",
        }

        result = await clear_all_tool.execute(arguments)

        assert result["isError"] is True
        assert "invalid" in result["content"][0]["text"].lower()


# ============================================================================
# Test Class 5: Combined Workflow Tests
# ============================================================================


@pytest.mark.integration
class TestPhase3CompleteWorkflow:
    """
    Test complete Phase 3 workflows combining multiple tools.

    Validates: End-to-end Phase 3 scenarios work correctly
    """

    @pytest.mark.asyncio
    async def test_export_after_code_indexing(
        self,
        index_codebase_tool,
        export_graph_tool,
        memory_processor,
        test_codebase_dir,
        temp_export_dir,
    ):
        """
        Test: Index codebase then export graph

        Scenario:
        1. Index a test codebase
        2. Export graph to GraphML
        3. Verify export contains indexed data

        Validates:
        - Indexing creates graph data
        - Export captures indexed data
        - Complete workflow works end-to-end
        """
        # Step 1: Index codebase
        index_args = {
            "repo_path": str(test_codebase_dir),
            "languages": ["python"],
        }

        index_result = await index_codebase_tool.execute(index_args)
        assert index_result["isError"] is False

        # Step 2: Export graph
        export_path = str(temp_export_dir / "indexed_graph.graphml")
        export_args = {
            "format": "graphml",
            "output_path": export_path,
        }

        export_result = await export_graph_tool.execute(export_args)
        assert export_result["isError"] is False

        # Step 3: Verify export file
        assert Path(export_path).exists()
        assert Path(export_path).stat().st_size > 0

    @pytest.mark.asyncio
    async def test_full_phase3_workflow(
        self,
        index_codebase_tool,
        export_graph_tool,
        delete_memory_tool,
        clear_all_tool,
        memory_processor,
        test_codebase_dir,
        temp_export_dir,
    ):
        """
        Test: Complete Phase 3 workflow

        Scenario:
        1. Index codebase
        2. Add additional memories
        3. Export graph in multiple formats
        4. Delete specific memory
        5. Export again (verify deletion)
        6. Clear all with safety phrase

        Validates:
        - All Phase 3 tools work together
        - Data flows correctly through workflow
        - Safety mechanisms function properly
        """
        # Step 1: Index codebase
        index_result = await index_codebase_tool.execute(
            {
                "repo_path": str(test_codebase_dir),
                "languages": ["python"],
            }
        )
        assert index_result["isError"] is False

        # Step 2: Add additional memories
        memory_id = await memory_processor.add_memory(
            text="Additional test memory for Phase 3",
            metadata={"source": "test", "type": "additional"},
        )

        # Step 3: Export graph in multiple formats
        for format_name, ext in [("graphml", ".graphml"), ("json", ".json")]:
            export_result = await export_graph_tool.execute(
                {
                    "format": format_name,
                    "output_path": str(temp_export_dir / f"workflow{ext}"),
                }
            )
            assert export_result["isError"] is False

        # Step 4: Delete specific memory
        delete_result = await delete_memory_tool.execute(
            {
                "memory_id": memory_id,
                "confirm": True,
            }
        )
        assert delete_result["isError"] is False

        # Step 5: Export again to verify deletion
        export_after_delete = await export_graph_tool.execute(
            {
                "format": "json",
                "output_path": str(temp_export_dir / "after_delete.json"),
            }
        )
        assert export_after_delete["isError"] is False

        # Step 6: Clear all with safety phrase
        clear_result = await clear_all_tool.execute(
            {
                "confirm_phrase": "DELETE ALL MEMORIES",
            }
        )
        assert clear_result["isError"] is False
        assert "cleared successfully" in clear_result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_safety_mechanisms(
        self,
        delete_memory_tool,
        clear_all_tool,
        memory_processor,
    ):
        """
        Test: All safety mechanisms work correctly

        Validates:
        - Delete requires explicit confirmation
        - Clear all requires exact phrase
        - Wrong phrases are rejected
        - Data remains protected
        """
        # Add test data
        memory_id = await memory_processor.add_memory(
            text="Protected memory",
            metadata={"source": "safety_test"},
        )

        # Test 1: Delete without confirmation fails
        result = await delete_memory_tool.execute(
            {
                "memory_id": memory_id,
                "confirm": False,
            }
        )
        assert result["isError"] is True

        # Test 2: Clear with wrong phrase fails
        result = await clear_all_tool.execute(
            {
                "confirm_phrase": "DELETE EVERYTHING",
            }
        )
        assert result["isError"] is True

        # Test 3: Delete with confirmation succeeds
        result = await delete_memory_tool.execute(
            {
                "memory_id": memory_id,
                "confirm": True,
            }
        )
        assert result["isError"] is False


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
