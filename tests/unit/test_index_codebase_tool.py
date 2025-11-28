"""
Unit tests for IndexCodebaseTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful repository indexing
- Input validation and error handling
- Language filtering
- Test file exclusion
- Statistics calculation
- Response formatting

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_core.exceptions import (
    DatabaseError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_mcp.tools.index_codebase import (
    LANGUAGE_EXTENSIONS,
    VALID_LANGUAGES,
    IndexCodebaseRequest,
    IndexCodebaseTool,
)


class TestIndexCodebaseToolInit:
    """Test IndexCodebaseTool initialization."""

    def test_init_success_with_valid_dependencies(self):
        """Test successful initialization with valid dependencies."""
        # Setup
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

        # Verify
        assert tool.repository_indexer == mock_indexer
        assert tool.memory_processor == mock_processor
        assert tool.name == "index_codebase"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_indexer_type(self):
        """Test initialization fails with non-CodeRepositoryIndexer type."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            IndexCodebaseTool(
                repository_indexer="not an indexer",
                memory_processor=mock_processor,
            )

        assert "CodeRepositoryIndexer" in str(exc_info.value)

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Setup
        mock_indexer = Mock(spec=CodeRepositoryIndexer)

        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            IndexCodebaseTool(
                repository_indexer=mock_indexer,
                memory_processor="not a processor",
            )

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_indexer(self):
        """Test initialization fails with None indexer."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute & Verify
        with pytest.raises(TypeError):
            IndexCodebaseTool(
                repository_indexer=None,
                memory_processor=mock_processor,
            )

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Setup
        mock_indexer = Mock(spec=CodeRepositoryIndexer)

        # Execute & Verify
        with pytest.raises(TypeError):
            IndexCodebaseTool(
                repository_indexer=mock_indexer,
                memory_processor=None,
            )


class TestIndexCodebaseToolExecute:
    """Test IndexCodebaseTool.execute() method."""

    @pytest.fixture
    def temp_repo_dir(self):
        """Create a temporary repository directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_indexer(self):
        """Create a mock CodeRepositoryIndexer."""
        indexer = Mock(spec=CodeRepositoryIndexer)
        indexer.index_repository = Mock(
            return_value={
                "repository": {
                    "path": "/test/repo",
                    "name": "test_repo",
                    "git_url": None,
                    "default_branch": "main",
                },
                "files": [
                    {
                        "path": "/test/repo/module.py",
                        "relative_path": "module.py",
                        "extension": ".py",
                        "size_bytes": 1000,
                        "lines": 50,
                        "encoding": "utf-8",
                        "last_modified": 1234567890.0,
                        "git_info": {},
                    },
                    {
                        "path": "/test/repo/script.js",
                        "relative_path": "script.js",
                        "extension": ".js",
                        "size_bytes": 500,
                        "lines": 25,
                        "encoding": "utf-8",
                        "last_modified": 1234567890.0,
                        "git_info": {},
                    },
                ],
                "statistics": {
                    "total_files": 2,
                    "total_lines": 75,
                    "total_size": 1500,
                    "by_extension": {".py": 1, ".js": 1},
                },
            }
        )
        return indexer

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.add_memory = AsyncMock(return_value="test-memory-id")
        return processor

    @pytest.fixture
    def tool(self, mock_indexer, mock_processor):
        """Create IndexCodebaseTool with mocks."""
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    @pytest.mark.asyncio
    async def test_execute_success_minimal(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution with minimal input."""
        # Setup
        arguments = {"repo_path": temp_repo_dir}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "successfully" in result["content"][0]["text"].lower()
        mock_indexer.index_repository.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_with_language_filter(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution with language filtering."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "languages": ["python"],
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Files indexed: 1" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_with_multiple_languages(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution with multiple language filters."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "languages": ["python", "javascript"],
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Files indexed: 2" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_with_max_file_size(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution with max file size filter."""
        # Setup
        # Mock indexer returns 1000 and 500 bytes files
        # min_file_size validator requires >= 1024
        arguments = {
            "repo_path": temp_repo_dir,
            "max_file_size": 2000,  # Includes both files
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        # Both files pass the filter, just verifying successful indexing
        assert "successfully" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_success_exclude_tests(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution excluding test files."""
        # Setup
        mock_indexer.index_repository.return_value = {
            "repository": {
                "path": temp_repo_dir,
                "name": "test_repo",
                "git_url": None,
                "default_branch": "main",
            },
            "files": [
                {
                    "path": f"{temp_repo_dir}/test_module.py",
                    "relative_path": "test_module.py",
                    "extension": ".py",
                    "size_bytes": 1000,
                    "lines": 50,
                    "encoding": "utf-8",
                    "last_modified": 1234567890.0,
                    "git_info": {},
                },
                {
                    "path": f"{temp_repo_dir}/module.py",
                    "relative_path": "module.py",
                    "extension": ".py",
                    "size_bytes": 1000,
                    "lines": 50,
                    "encoding": "utf-8",
                    "last_modified": 1234567890.0,
                    "git_info": {},
                },
            ],
            "statistics": {
                "total_files": 2,
                "total_lines": 100,
                "total_size": 2000,
                "by_extension": {".py": 2},
            },
        }

        arguments = {
            "repo_path": temp_repo_dir,
            "include_tests": False,  # Exclude test files
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Files indexed: 1" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_include_tests(self, tool, mock_indexer, temp_repo_dir):
        """Test successful execution including test files."""
        # Setup
        mock_indexer.index_repository.return_value = {
            "repository": {
                "path": temp_repo_dir,
                "name": "test_repo",
                "git_url": None,
                "default_branch": "main",
            },
            "files": [
                {
                    "path": f"{temp_repo_dir}/test_module.py",
                    "relative_path": "test_module.py",
                    "extension": ".py",
                    "size_bytes": 1000,
                    "lines": 50,
                    "encoding": "utf-8",
                    "last_modified": 1234567890.0,
                    "git_info": {},
                },
                {
                    "path": f"{temp_repo_dir}/module.py",
                    "relative_path": "module.py",
                    "extension": ".py",
                    "size_bytes": 1000,
                    "lines": 50,
                    "encoding": "utf-8",
                    "last_modified": 1234567890.0,
                    "git_info": {},
                },
            ],
            "statistics": {
                "total_files": 2,
                "total_lines": 100,
                "total_size": 2000,
                "by_extension": {".py": 2},
            },
        }

        arguments = {
            "repo_path": temp_repo_dir,
            "include_tests": True,  # Include test files
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Files indexed: 2" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_repo_path(self, tool):
        """Test execution with non-existent repository path."""
        # Setup
        arguments = {"repo_path": "/nonexistent/path/to/repo"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_repo_path_not_directory(self, tool):
        """Test execution with file path instead of directory."""
        # Setup
        with tempfile.NamedTemporaryFile() as tmp_file:
            arguments = {"repo_path": tmp_file.name}

            # Execute
            result = await tool.execute(arguments)

            # Verify
            assert result["isError"] is True
            assert "not a directory" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_empty_repo_path(self, tool):
        """Test execution with empty repo path."""
        # Setup
        arguments = {"repo_path": ""}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_language(self, tool, temp_repo_dir):
        """Test execution with unsupported language."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "languages": ["invalid_language"],
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_max_file_size(self, tool, temp_repo_dir):
        """Test execution with invalid max_file_size."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "max_file_size": 100,  # Too small
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_indexer_processing_error(self, tool, mock_indexer, temp_repo_dir):
        """Test execution when indexer raises ProcessingError."""
        # Setup
        mock_indexer.index_repository.side_effect = ProcessingError(
            message="Indexing failed",
            error_code="PROC_001",
        )
        arguments = {"repo_path": temp_repo_dir}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_indexer, temp_repo_dir):
        """Test execution when indexer raises unexpected exception."""
        # Setup
        mock_indexer.index_repository.side_effect = RuntimeError("Boom")
        arguments = {"repo_path": temp_repo_dir}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]


class TestIndexCodebaseToolValidation:
    """Test IndexCodebaseTool._validate_arguments() method."""

    @pytest.fixture
    def temp_repo_dir(self):
        """Create a temporary repository directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def tool(self):
        """Create IndexCodebaseTool with mocks."""
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    def test_validate_arguments_success_minimal(self, tool, temp_repo_dir):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"repo_path": temp_repo_dir}

        # Execute
        repo_path, languages, recursive, max_file_size, include_tests = tool._validate_arguments(
            arguments
        )

        # Verify
        assert repo_path == temp_repo_dir
        assert languages == []
        assert recursive is True
        assert max_file_size == 10 * 1024 * 1024
        assert include_tests is False

    def test_validate_arguments_success_with_languages(self, tool, temp_repo_dir):
        """Test successful validation with language filter."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "languages": ["python", "javascript"],
        }

        # Execute
        repo_path, languages, recursive, max_file_size, include_tests = tool._validate_arguments(
            arguments
        )

        # Verify
        assert languages == ["python", "javascript"]

    def test_validate_arguments_success_with_all_params(self, tool, temp_repo_dir):
        """Test successful validation with all parameters."""
        # Setup
        arguments = {
            "repo_path": temp_repo_dir,
            "languages": ["python"],
            "recursive": False,
            "max_file_size": 5242880,
            "include_tests": True,
        }

        # Execute
        repo_path, languages, recursive, max_file_size, include_tests = tool._validate_arguments(
            arguments
        )

        # Verify
        assert languages == ["python"]
        assert recursive is False
        assert max_file_size == 5242880
        assert include_tests is True

    def test_validate_arguments_nonexistent_path_raises_error(self, tool):
        """Test that nonexistent path raises CoreValidationError."""
        # Setup
        arguments = {"repo_path": "/nonexistent/path"}

        # Execute & Verify
        with pytest.raises(CoreValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_file_path_raises_error(self, tool):
        """Test that file path (not directory) raises error."""
        # Setup
        with tempfile.NamedTemporaryFile() as tmp_file:
            arguments = {"repo_path": tmp_file.name}

            # Execute & Verify
            with pytest.raises(CoreValidationError):
                tool._validate_arguments(arguments)

    def test_validate_arguments_empty_repo_path_raises_error(self, tool):
        """Test that empty repo_path raises ValidationError."""
        # Setup
        arguments = {"repo_path": ""}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_missing_repo_path_raises_error(self, tool):
        """Test that missing repo_path raises ValidationError."""
        # Setup
        arguments = {"languages": ["python"]}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)


class TestIndexCodebaseToolFiltering:
    """Test IndexCodebaseTool filtering methods."""

    @pytest.fixture
    def tool(self):
        """Create IndexCodebaseTool with mocks."""
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    def test_filter_files_by_language_python(self, tool):
        """Test filtering files by Python language."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/script.js",
                "relative_path": "script.js",
                "extension": ".js",
                "size_bytes": 500,
                "lines": 25,
            },
        ]

        # Execute
        filtered = tool._filter_files(
            files, languages=["python"], include_tests=False, max_file_size=10_000_000
        )

        # Verify
        assert len(filtered) == 1
        assert filtered[0]["extension"] == ".py"

    def test_filter_files_by_multiple_languages(self, tool):
        """Test filtering files by multiple languages."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/script.js",
                "relative_path": "script.js",
                "extension": ".js",
                "size_bytes": 500,
                "lines": 25,
            },
            {
                "path": "/repo/style.rs",
                "relative_path": "style.rs",
                "extension": ".rs",
                "size_bytes": 800,
                "lines": 40,
            },
        ]

        # Execute
        filtered = tool._filter_files(
            files,
            languages=["python", "javascript"],
            include_tests=False,
            max_file_size=10_000_000,
        )

        # Verify
        assert len(filtered) == 2
        assert all(f["extension"] in [".py", ".js"] for f in filtered)

    def test_filter_files_no_language_filter(self, tool):
        """Test filtering with no language filter (includes all)."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/script.js",
                "relative_path": "script.js",
                "extension": ".js",
                "size_bytes": 500,
                "lines": 25,
            },
        ]

        # Execute
        filtered = tool._filter_files(
            files, languages=[], include_tests=False, max_file_size=10_000_000
        )

        # Verify
        assert len(filtered) == 2

    def test_filter_files_by_max_size(self, tool):
        """Test filtering files by maximum size."""
        # Setup
        files = [
            {
                "path": "/repo/small.py",
                "relative_path": "small.py",
                "extension": ".py",
                "size_bytes": 500,
                "lines": 25,
            },
            {
                "path": "/repo/large.py",
                "relative_path": "large.py",
                "extension": ".py",
                "size_bytes": 2000,
                "lines": 100,
            },
        ]

        # Execute
        filtered = tool._filter_files(files, languages=[], include_tests=False, max_file_size=1000)

        # Verify
        assert len(filtered) == 1
        assert filtered[0]["size_bytes"] == 500

    def test_filter_files_exclude_test_files(self, tool):
        """Test filtering that excludes test files."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/test_module.py",
                "relative_path": "test_module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
        ]

        # Execute
        filtered = tool._filter_files(
            files, languages=[], include_tests=False, max_file_size=10_000_000
        )

        # Verify
        assert len(filtered) == 1
        assert "test" not in filtered[0]["relative_path"]

    def test_filter_files_include_test_files(self, tool):
        """Test filtering that includes test files."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/test_module.py",
                "relative_path": "test_module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
        ]

        # Execute
        filtered = tool._filter_files(
            files, languages=[], include_tests=True, max_file_size=10_000_000
        )

        # Verify
        assert len(filtered) == 2

    def test_is_test_file_with_test_prefix(self, tool):
        """Test detection of test files with test_ prefix."""
        # Verify
        assert tool._is_test_file("test_module.py") is True
        assert tool._is_test_file("test_helper.py") is True

    def test_is_test_file_with_test_suffix(self, tool):
        """Test detection of test files with _test suffix."""
        # Verify
        assert tool._is_test_file("module_test.py") is True
        assert tool._is_test_file("module.test.js") is True

    def test_is_test_file_with_spec_suffix(self, tool):
        """Test detection of spec files."""
        # Verify
        assert tool._is_test_file("module.spec.js") is True
        assert tool._is_test_file("component.test.ts") is True

    def test_is_test_file_with_test_directory(self, tool):
        """Test detection of files in test directories."""
        # Verify
        assert tool._is_test_file("tests/utils.py") is True
        assert tool._is_test_file("test/helpers.js") is True
        assert tool._is_test_file("__tests__/module.ts") is True

    def test_is_test_file_regular_file(self, tool):
        """Test that regular files are not detected as tests."""
        # Verify
        assert tool._is_test_file("module.py") is False
        assert tool._is_test_file("script.js") is False
        assert tool._is_test_file("README.md") is False


class TestIndexCodebaseToolStatistics:
    """Test IndexCodebaseTool statistics calculation."""

    @pytest.fixture
    def tool(self):
        """Create IndexCodebaseTool with mocks."""
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    def test_calculate_statistics_single_file(self, tool):
        """Test statistics calculation with single file."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
        ]
        full_result = {"statistics": {"by_extension": {".py": 1}}}

        # Execute
        stats = tool._calculate_statistics(files, full_result)

        # Verify
        assert stats["files_indexed"] == 1
        assert stats["total_lines"] == 50
        assert stats["languages"]["python"] == 1

    def test_calculate_statistics_multiple_files(self, tool):
        """Test statistics calculation with multiple files."""
        # Setup
        files = [
            {
                "path": "/repo/module.py",
                "relative_path": "module.py",
                "extension": ".py",
                "size_bytes": 1000,
                "lines": 50,
            },
            {
                "path": "/repo/script.js",
                "relative_path": "script.js",
                "extension": ".js",
                "size_bytes": 500,
                "lines": 25,
            },
            {
                "path": "/repo/data.ts",
                "relative_path": "data.ts",
                "extension": ".ts",
                "size_bytes": 300,
                "lines": 15,
            },
        ]
        full_result = {"statistics": {}}

        # Execute
        stats = tool._calculate_statistics(files, full_result)

        # Verify
        assert stats["files_indexed"] == 3
        assert stats["total_lines"] == 90
        assert stats["languages"]["python"] == 1
        assert stats["languages"]["javascript"] == 1
        assert stats["languages"]["typescript"] == 1

    def test_calculate_statistics_no_files(self, tool):
        """Test statistics calculation with no files."""
        # Setup
        files = []
        full_result = {"statistics": {}}

        # Execute
        stats = tool._calculate_statistics(files, full_result)

        # Verify
        assert stats["files_indexed"] == 0
        assert stats["total_lines"] == 0
        assert len(stats["languages"]) == 0


class TestIndexCodebaseToolInputSchema:
    """Test IndexCodebaseTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = IndexCodebaseTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "repo_path" in schema["properties"]
        assert "languages" in schema["properties"]
        assert "recursive" in schema["properties"]
        assert "repo_path" in schema["required"]
        assert "languages" not in schema["required"]

    def test_input_schema_repo_path_property(self):
        """Test that repo_path property is correctly defined."""
        schema = IndexCodebaseTool.input_schema
        repo_path_schema = schema["properties"]["repo_path"]

        # Verify
        assert repo_path_schema["type"] == "string"
        assert repo_path_schema["minLength"] == 1

    def test_input_schema_languages_property(self):
        """Test that languages property is correctly defined."""
        schema = IndexCodebaseTool.input_schema
        languages_schema = schema["properties"]["languages"]

        # Verify
        assert languages_schema["type"] == "array"
        assert "enum" in languages_schema["items"]
        assert set(languages_schema["items"]["enum"]) == VALID_LANGUAGES

    def test_input_schema_valid_languages_enum(self):
        """Test that all valid languages are in schema enum."""
        schema = IndexCodebaseTool.input_schema
        enum_languages = set(schema["properties"]["languages"]["items"]["enum"])

        # Verify all languages are present
        assert enum_languages == VALID_LANGUAGES


class TestIndexCodebaseRequestPydantic:
    """Test IndexCodebaseRequest Pydantic model."""

    def test_request_minimal(self):
        """Test creating request with minimal parameters."""
        # Execute
        request = IndexCodebaseRequest(repo_path="/test/repo")

        # Verify
        assert request.repo_path == "/test/repo"
        assert request.languages == []
        assert request.recursive is True
        assert request.include_tests is False

    def test_request_with_all_params(self):
        """Test creating request with all parameters."""
        # Execute
        request = IndexCodebaseRequest(
            repo_path="/test/repo",
            languages=["python", "javascript"],
            recursive=False,
            max_file_size=5000000,
            include_tests=True,
        )

        # Verify
        assert request.repo_path == "/test/repo"
        assert request.languages == ["python", "javascript"]
        assert request.recursive is False
        assert request.max_file_size == 5000000
        assert request.include_tests is True

    def test_request_invalid_language(self):
        """Test that invalid language raises ValidationError."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            IndexCodebaseRequest(
                repo_path="/test/repo",
                languages=["invalid_language"],
            )

    def test_request_empty_repo_path(self):
        """Test that empty repo_path raises ValidationError."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            IndexCodebaseRequest(repo_path="")

    def test_request_whitespace_only_repo_path(self):
        """Test that whitespace-only repo_path raises ValidationError."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            IndexCodebaseRequest(repo_path="   ")

    def test_request_invalid_max_file_size_too_small(self):
        """Test that max_file_size too small raises ValidationError."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            IndexCodebaseRequest(
                repo_path="/test/repo",
                max_file_size=512,  # Less than 1024
            )

    def test_request_invalid_max_file_size_too_large(self):
        """Test that max_file_size too large raises ValidationError."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            IndexCodebaseRequest(
                repo_path="/test/repo",
                max_file_size=200 * 1024 * 1024,  # More than 100MB
            )


class TestIndexCodebaseToolExtensionToLanguage:
    """Test IndexCodebaseTool._extension_to_language() method."""

    @pytest.fixture
    def tool(self):
        """Create IndexCodebaseTool with mocks."""
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    def test_extension_to_language_python(self, tool):
        """Test Python extension detection."""
        assert tool._extension_to_language(".py") == "python"

    def test_extension_to_language_javascript(self, tool):
        """Test JavaScript extension detection."""
        assert tool._extension_to_language(".js") == "javascript"
        assert tool._extension_to_language(".jsx") == "javascript"

    def test_extension_to_language_typescript(self, tool):
        """Test TypeScript extension detection."""
        assert tool._extension_to_language(".ts") == "typescript"
        assert tool._extension_to_language(".tsx") == "typescript"

    def test_extension_to_language_rust(self, tool):
        """Test Rust extension detection."""
        assert tool._extension_to_language(".rs") == "rust"

    def test_extension_to_language_go(self, tool):
        """Test Go extension detection."""
        assert tool._extension_to_language(".go") == "go"

    def test_extension_to_language_java(self, tool):
        """Test Java extension detection."""
        assert tool._extension_to_language(".java") == "java"

    def test_extension_to_language_cpp(self, tool):
        """Test C++ extension detection."""
        assert tool._extension_to_language(".cpp") == "cpp"
        assert tool._extension_to_language(".cc") == "cpp"

    def test_extension_to_language_c(self, tool):
        """Test C extension detection."""
        assert tool._extension_to_language(".c") == "c"
        assert tool._extension_to_language(".h") == "c"

    def test_extension_to_language_unknown(self, tool):
        """Test unknown extension returns 'unknown'."""
        assert tool._extension_to_language(".xyz") == "unknown"
        assert tool._extension_to_language(".md") == "unknown"
        assert tool._extension_to_language("") == "unknown"

    def test_extension_to_language_case_insensitive(self, tool):
        """Test extension detection is case-insensitive."""
        assert tool._extension_to_language(".PY") == "python"
        assert tool._extension_to_language(".Js") == "javascript"


class TestIndexCodebaseToolStoresFileContent:
    """Test that IndexCodebaseTool stores actual file content (Issue #2 fix)."""

    @pytest.fixture
    def temp_repo_dir(self):
        """Create a temporary repository with code files."""
        temp_dir = tempfile.mkdtemp()

        # Create a Python file with actual content
        python_file = Path(temp_dir) / "example.py"
        python_file.write_text('''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
''')

        # Create a TypeScript file
        ts_file = Path(temp_dir) / "utils.ts"
        ts_file.write_text('''
export function formatDate(date: Date): string {
    return date.toISOString();
}

export interface User {
    id: number;
    name: string;
    email: string;
}
''')

        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_db_client(self):
        """Create mock database client."""
        db_client = AsyncMock()
        db_client.mark_code_memories_stale = AsyncMock(return_value=0)
        db_client.mark_memory_fresh = AsyncMock(return_value=None)
        db_client.count_stale_memories = AsyncMock(return_value=0)
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create mock MemoryProcessor with db_client."""
        processor = Mock(spec=MemoryProcessor)
        processor.add_memory = AsyncMock(return_value="test-memory-id")
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def mock_indexer(self, temp_repo_dir):
        """Create mock indexer that returns real file info."""
        indexer = Mock(spec=CodeRepositoryIndexer)
        indexer.index_repository = Mock(
            return_value={
                "repository": {
                    "path": temp_repo_dir,
                    "name": "test_repo",
                    "git_url": None,
                    "default_branch": "main",
                },
                "files": [
                    {
                        "path": str(Path(temp_repo_dir) / "example.py"),
                        "relative_path": "example.py",
                        "extension": ".py",
                        "size_bytes": 300,
                        "lines": 15,
                        "encoding": "utf-8",
                        "last_modified": 1234567890.0,
                        "git_info": {},
                    },
                    {
                        "path": str(Path(temp_repo_dir) / "utils.ts"),
                        "relative_path": "utils.ts",
                        "extension": ".ts",
                        "size_bytes": 200,
                        "lines": 10,
                        "encoding": "utf-8",
                        "last_modified": 1234567890.0,
                        "git_info": {},
                    },
                ],
                "statistics": {
                    "total_files": 2,
                    "total_lines": 25,
                    "total_size": 500,
                    "by_extension": {".py": 1, ".ts": 1},
                },
            }
        )
        return indexer

    @pytest.fixture
    def tool(self, mock_indexer, mock_processor):
        """Create IndexCodebaseTool with mocks."""
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    @pytest.mark.asyncio
    async def test_execute_stores_file_content_not_just_metadata(
        self, tool, mock_processor, temp_repo_dir
    ):
        """Test that execute stores actual file content, not just metadata."""
        # Execute
        result = await tool.execute({"repo_path": temp_repo_dir})

        # Verify success
        assert result["isError"] is False

        # Verify add_memory was called with actual content
        assert mock_processor.add_memory.called

        # Get the text that was stored
        calls = mock_processor.add_memory.call_args_list
        stored_texts = [call[0][0] for call in calls]

        # Verify actual code content is in stored text (not just metadata)
        python_stored = any("def hello_world" in text for text in stored_texts)
        ts_stored = any("export function formatDate" in text for text in stored_texts)

        assert python_stored, "Python function content should be stored"
        assert ts_stored, "TypeScript function content should be stored"

    @pytest.mark.asyncio
    async def test_execute_includes_file_header_in_content(
        self, tool, mock_processor, temp_repo_dir
    ):
        """Test that stored content includes file metadata header."""
        # Execute
        await tool.execute({"repo_path": temp_repo_dir})

        # Get the text that was stored
        calls = mock_processor.add_memory.call_args_list
        stored_texts = [call[0][0] for call in calls]

        # Verify header format
        assert any("# File: example.py" in text for text in stored_texts)
        assert any("# Language: .py" in text for text in stored_texts)

    @pytest.mark.asyncio
    async def test_execute_metadata_includes_language(
        self, tool, mock_processor, temp_repo_dir
    ):
        """Test that metadata includes language field."""
        # Execute
        await tool.execute({"repo_path": temp_repo_dir})

        # Get the metadata that was stored
        calls = mock_processor.add_memory.call_args_list
        stored_metadata = [call[0][1] for call in calls]

        # Verify language is in metadata
        assert any(meta.get("language") == "python" for meta in stored_metadata)
        assert any(meta.get("language") == "typescript" for meta in stored_metadata)

    @pytest.mark.asyncio
    async def test_execute_metadata_includes_indexed_at(
        self, tool, mock_processor, temp_repo_dir
    ):
        """Test that metadata includes indexed_at timestamp."""
        # Execute
        await tool.execute({"repo_path": temp_repo_dir})

        # Get the metadata that was stored
        calls = mock_processor.add_memory.call_args_list
        stored_metadata = [call[0][1] for call in calls]

        # Verify indexed_at is in metadata
        assert all("indexed_at" in meta for meta in stored_metadata)

    @pytest.mark.asyncio
    async def test_execute_skips_empty_files(self, tool, mock_processor, mock_indexer):
        """Test that empty files are skipped."""
        # Create a temp dir with an empty file
        temp_dir = tempfile.mkdtemp()
        try:
            empty_file = Path(temp_dir) / "empty.py"
            empty_file.write_text("")

            mock_indexer.index_repository.return_value = {
                "repository": {"path": temp_dir, "name": "test", "git_url": None, "default_branch": "main"},
                "files": [
                    {
                        "path": str(empty_file),
                        "relative_path": "empty.py",
                        "extension": ".py",
                        "size_bytes": 0,
                        "lines": 0,
                        "encoding": "utf-8",
                        "last_modified": 1234567890.0,
                        "git_info": {},
                    },
                ],
                "statistics": {"total_files": 1, "total_lines": 0, "total_size": 0, "by_extension": {".py": 1}},
            }

            # Execute
            await tool.execute({"repo_path": temp_dir})

            # Verify add_memory was NOT called for empty file
            assert not mock_processor.add_memory.called
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
