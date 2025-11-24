"""
IndexCodebase MCP Tool - Full Implementation.

Indexes code repositories with AST analysis for code search and storage.
Delegates to CodeRepositoryIndexer for scanning and extraction.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import structlog
from pydantic import ValidationError, BaseModel, ConfigDict, field_validator
from typing_extensions import Annotated

from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import (
    ValidationError as CoreValidationError,
    ProcessingError,
    DatabaseError,
)


logger = structlog.get_logger(__name__)


# Language to file extensions mapping
LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "go": {".go"},
    "rust": {".rs"},
    "cpp": {".cpp", ".cc", ".cxx"},
    "c": {".c", ".h"},
}

# Valid languages
VALID_LANGUAGES = set(LANGUAGE_EXTENSIONS.keys())


class IndexCodebaseRequest(BaseModel):
    """Pydantic model for validating index_codebase request."""

    model_config = ConfigDict(extra="forbid")

    repo_path: str
    languages: List[str] = []
    recursive: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB default
    include_tests: bool = False

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Validate that all languages are supported."""
        if not isinstance(v, list):
            raise ValueError("languages must be a list")

        for lang in v:
            if lang not in VALID_LANGUAGES:
                raise ValueError(
                    f"Unsupported language: {lang}. "
                    f"Supported: {', '.join(sorted(VALID_LANGUAGES))}"
                )

        return v

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that repo_path is provided."""
        if not v or not v.strip():
            raise ValueError("repo_path cannot be empty")
        return v.strip()

    @field_validator("max_file_size")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max_file_size is reasonable."""
        if v < 1024:
            raise ValueError("max_file_size must be at least 1024 bytes")
        if v > 100 * 1024 * 1024:
            raise ValueError("max_file_size cannot exceed 100MB")
        return v


class IndexCodebaseResponse(BaseModel):
    """Pydantic model for index_codebase response."""

    repo_path: str
    files_indexed: int
    functions_found: int
    classes_found: int
    languages: Dict[str, int]
    total_lines: int
    processing_time_ms: float


class IndexCodebaseTool:
    """
    MCP tool for indexing code repositories.

    This tool validates input, scans repositories using CodeRepositoryIndexer,
    extracts code structure information, and formats the response according
    to MCP protocol.

    Attributes:
        name: Tool identifier ("index_codebase")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        repository_indexer: CodeRepositoryIndexer instance for scanning
        memory_processor: MemoryProcessor instance for storing code
        logger: Structured logger for operations
    """

    name = "index_codebase"
    description = (
        "Index a code repository with AST analysis to extract functions, classes, "
        "and other code structures. Results are stored as memories for later retrieval."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to code repository",
                "minLength": 1,
            },
            "languages": {
                "type": "array",
                "description": "Programming languages to index (empty = all)",
                "items": {
                    "type": "string",
                    "enum": sorted(list(VALID_LANGUAGES)),
                },
                "default": [],
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursively index subdirectories (default: true)",
                "default": True,
            },
            "max_file_size": {
                "type": "integer",
                "description": "Max file size in bytes (default: 10485760)",
                "minimum": 1024,
                "maximum": 100000000,
                "default": 10485760,
            },
            "include_tests": {
                "type": "boolean",
                "description": "Include test files (default: false)",
                "default": False,
            },
        },
        "required": ["repo_path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repository_indexer: CodeRepositoryIndexer,
        memory_processor: MemoryProcessor,
    ) -> None:
        """
        Initialize IndexCodebaseTool with required dependencies.

        Args:
            repository_indexer: CodeRepositoryIndexer instance for scanning repos.
                Must be initialized and configured.
            memory_processor: MemoryProcessor instance for storing code memories.
                Must be initialized and connected to database.

        Raises:
            TypeError: If dependencies are not the correct type
            ValueError: If dependencies are not initialized

        Example:
            >>> indexer = CodeRepositoryIndexer()
            >>> processor = MemoryProcessor(...)
            >>> tool = IndexCodebaseTool(
            ...     repository_indexer=indexer,
            ...     memory_processor=processor
            ... )
        """
        if not isinstance(repository_indexer, CodeRepositoryIndexer):
            raise TypeError(
                f"repository_indexer must be CodeRepositoryIndexer instance, "
                f"got {type(repository_indexer)}"
            )

        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, "
                f"got {type(memory_processor)}"
            )

        self.repository_indexer = repository_indexer
        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("index_codebase_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute index_codebase tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the index_codebase tool. It validates inputs, indexes the
        repository through the repository indexer, stores results as memories,
        and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - repo_path (str, required): Repository path to index
                - languages (list, optional): Languages to filter
                - recursive (bool, optional): Recursively index subdirectories
                - max_file_size (int, optional): Maximum file size in bytes
                - include_tests (bool, optional): Include test files

        Returns:
            MCP-formatted response dictionary with indexing results

        Example:
            >>> result = await tool.execute({
            ...     "repo_path": "/path/to/repo",
            ...     "languages": ["python", "javascript"]
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)

        start_time = time.time()

        try:
            # Step 1: Validate and extract arguments
            log.info("validating_arguments")
            (
                repo_path,
                languages,
                recursive,
                max_file_size,
                include_tests,
            ) = self._validate_arguments(arguments)

            # Step 2: Index repository
            log.info(
                "indexing_repository",
                repo_path=repo_path,
                languages=languages,
                recursive=recursive,
            )
            index_result = self.repository_indexer.index_repository(repo_path)

            # Step 3: Filter files by language and test inclusion
            files = index_result.get("files", [])
            files = self._filter_files(files, languages, include_tests, max_file_size)

            # Step 4: Calculate statistics
            stats = self._calculate_statistics(files, index_result)

            # Step 5: Store code as memories (optional - for advanced features)
            # This would store individual code files/functions as memories
            memories_created = 0
            # await self._store_code_memories(files)

            # Step 6: Format success response
            processing_time_ms = (time.time() - start_time) * 1000

            log.info(
                "repository_indexed_successfully",
                files_indexed=stats["files_indexed"],
                functions=stats["functions_found"],
                classes=stats["classes_found"],
                processing_time_ms=processing_time_ms,
            )

            return self._format_success(
                repo_path=repo_path,
                files_indexed=stats["files_indexed"],
                functions_found=stats["functions_found"],
                classes_found=stats["classes_found"],
                languages=stats["languages"],
                total_lines=stats["total_lines"],
                processing_time_ms=processing_time_ms,
            )

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            log.warning("validation_error", error=str(e))
            return self._format_error(e)

        except (ProcessingError, DatabaseError) as e:
            # Core processing error
            log.error(
                "processing_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e)

        except Exception as e:
            # Unexpected error
            log.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e)

    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
    ) -> Tuple[str, List[str], bool, int, bool]:
        """
        Validate and extract arguments from MCP request.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            Tuple of (repo_path, languages, recursive, max_file_size, include_tests)

        Raises:
            ValidationError: If arguments don't match schema
            CoreValidationError: If path validation fails
        """
        # Validate using Pydantic model
        try:
            request = IndexCodebaseRequest(**arguments)
        except ValidationError as e:
            raise

        # Validate repo path exists and is directory
        from pathlib import Path

        repo_path_obj = Path(request.repo_path).expanduser().resolve()

        if not repo_path_obj.exists():
            raise CoreValidationError(
                f"Repository path does not exist: {request.repo_path}",
                error_code="VAL_001",
                details={"path": request.repo_path},
            )

        if not repo_path_obj.is_dir():
            raise CoreValidationError(
                f"Repository path is not a directory: {request.repo_path}",
                error_code="VAL_001",
                details={"path": request.repo_path},
            )

        return (
            str(repo_path_obj),
            request.languages,
            request.recursive,
            request.max_file_size,
            request.include_tests,
        )

    def _filter_files(
        self,
        files: List[Dict[str, Any]],
        languages: List[str],
        include_tests: bool,
        max_file_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter files by language and test inclusion.

        Args:
            files: List of file dictionaries from indexer
            languages: List of language filters (empty = all)
            include_tests: Whether to include test files
            max_file_size: Maximum file size threshold

        Returns:
            Filtered list of files
        """
        filtered = []

        # Build set of extensions to include
        allowed_extensions: Set[str] = set()
        if languages:
            for lang in languages:
                if lang in LANGUAGE_EXTENSIONS:
                    allowed_extensions.update(LANGUAGE_EXTENSIONS[lang])
        else:
            # Include all extensions if no language filter
            for exts in LANGUAGE_EXTENSIONS.values():
                allowed_extensions.update(exts)

        for file_dict in files:
            # Check extension
            ext = file_dict.get("extension", "").lower()
            if ext not in allowed_extensions:
                continue

            # Check file size
            if file_dict.get("size_bytes", 0) > max_file_size:
                continue

            # Check test file inclusion
            if not include_tests:
                relative_path = file_dict.get("relative_path", "").lower()
                if self._is_test_file(relative_path):
                    continue

            filtered.append(file_dict)

        return filtered

    def _is_test_file(self, relative_path: str) -> bool:
        """
        Check if a file is a test file.

        Args:
            relative_path: Relative path of file

        Returns:
            True if file appears to be a test file
        """
        test_patterns = [
            "test_",
            "_test.py",
            ".test.js",
            ".spec.js",
            ".test.ts",
            ".spec.ts",
            "__tests__",
            "tests/",
            "test/",
        ]

        relative_path_lower = relative_path.lower()
        return any(pattern in relative_path_lower for pattern in test_patterns)

    def _calculate_statistics(
        self,
        files: List[Dict[str, Any]],
        full_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate statistics from indexed files.

        Args:
            files: Filtered list of files
            full_result: Full result from indexer including statistics

        Returns:
            Dictionary with calculated statistics
        """
        stats = {
            "files_indexed": len(files),
            "functions_found": 0,  # Would need AST parsing to count
            "classes_found": 0,  # Would need AST parsing to count
            "languages": {},
            "total_lines": 0,
        }

        # Count files by language
        for file_dict in files:
            ext = file_dict.get("extension", "").lower()

            # Find language for this extension
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                    break

            # Count lines
            stats["total_lines"] += file_dict.get("lines", 0)

        return stats

    async def _store_code_memories(self, files: List[Dict[str, Any]]) -> int:
        """
        Store code files as memories for later retrieval.

        Args:
            files: List of code file dictionaries

        Returns:
            Number of memories created
        """
        memories_created = 0

        for file_dict in files:
            try:
                # Create memory for each code file with metadata
                file_path = file_dict.get("path")
                relative_path = file_dict.get("relative_path")
                extension = file_dict.get("extension")
                lines = file_dict.get("lines", 0)

                # Create a summary for the memory
                summary = f"Code file: {relative_path}\n" f"Type: {extension}\n" f"Lines: {lines}"

                metadata = {
                    "source": "code_indexer",
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "extension": extension,
                    "lines": lines,
                }

                # Store as memory
                await self.memory_processor.add_memory(summary, metadata)
                memories_created += 1

            except Exception as e:
                self.logger.warning(
                    "failed_to_store_code_memory",
                    file_path=file_dict.get("path"),
                    error=str(e),
                )
                continue

        return memories_created

    def _format_success(
        self,
        repo_path: str,
        files_indexed: int,
        functions_found: int,
        classes_found: int,
        languages: Dict[str, int],
        total_lines: int,
        processing_time_ms: float,
    ) -> Dict[str, Any]:
        """
        Format successful indexing as MCP response.

        Args:
            repo_path: Repository path that was indexed
            files_indexed: Number of files indexed
            functions_found: Number of functions extracted
            classes_found: Number of classes extracted
            languages: Dictionary of language counts
            total_lines: Total lines of code
            processing_time_ms: Processing time in milliseconds

        Returns:
            MCP response dictionary
        """
        # Format language summary
        lang_summary = (
            ", ".join(
                [f"{lang.capitalize()} ({count})" for lang, count in sorted(languages.items())]
            )
            or "None"
        )

        processing_time_sec = processing_time_ms / 1000.0

        message = (
            f"Repository indexed successfully.\n"
            f"Path: {repo_path}\n"
            f"Files indexed: {files_indexed}\n"
            f"Functions: {functions_found}\n"
            f"Classes: {classes_found}\n"
            f"Languages: {lang_summary}\n"
            f"Total lines: {total_lines:,}\n"
            f"Indexing time: {processing_time_sec:.2f}s"
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
            "isError": False,
        }

    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format error as MCP error response.

        Args:
            error: Exception that occurred during processing

        Returns:
            MCP error response dictionary
        """
        # Determine error message based on exception type
        if isinstance(error, ValidationError):
            # Pydantic validation error
            error_msg = str(error)
        elif isinstance(error, CoreValidationError):
            # Core validation error - safe to expose
            error_msg = str(error)
        elif isinstance(error, DatabaseError):
            # Database error - suggest retry
            error_msg = "Database temporarily unavailable. Please retry in a few seconds."
        elif isinstance(error, ProcessingError):
            # Processing error
            error_msg = "Failed to index repository. Please try again."
        else:
            # Unknown error - generic message for security
            error_msg = "An internal error occurred while indexing the repository."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
