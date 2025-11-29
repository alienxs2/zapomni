"""
IndexCodebase MCP Tool - Full Implementation.

Indexes code repositories with AST analysis for code search and storage.
Delegates to CodeRepositoryIndexer for scanning and extraction.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import time
from typing import Any, Dict, List, Set, Tuple

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_core.exceptions import (
    DatabaseError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_db.models import DEFAULT_WORKSPACE_ID

logger = structlog.get_logger(__name__)

# Tree-sitter imports for AST parsing
try:
    from zapomni_core.treesitter.analyzers.call_graph import CallGraphAnalyzer
    from zapomni_core.treesitter.extractors import GenericExtractor
    from zapomni_core.treesitter.models import ExtractedCode
    from zapomni_core.treesitter.parser.factory import ParserFactory
    from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False
    ParserFactory = None  # type: ignore
    GenericExtractor = None  # type: ignore
    ExtractedCode = None  # type: ignore
    CallGraphAnalyzer = None  # type: ignore


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
    stale_count: int = 0  # Count of stale memories (deleted files)


class IndexCodebaseTool:
    """
    MCP tool for indexing code repositories.

    This tool validates input, scans repositories using CodeRepositoryIndexer,
    extracts code structure information, and formats the response according
    to MCP protocol.

    Delta Indexing Support:
        Before indexing, marks existing code memories as stale.
        After processing each file, marks it fresh.
        Remaining stale memories indicate deleted files.
        Use prune_memory tool to clean up stale entries.

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

        # Initialize call graph analyzer if tree-sitter is available
        self._call_graph_analyzer = (
            CallGraphAnalyzer() if TREESITTER_AVAILABLE and CallGraphAnalyzer is not None else None
        )

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

            # Step 1.5: Delta indexing - Mark existing memories as stale
            workspace_id = DEFAULT_WORKSPACE_ID
            marked_stale = 0
            if hasattr(self.memory_processor, "db_client") and self.memory_processor.db_client:
                try:
                    marked_stale = await self.memory_processor.db_client.mark_code_memories_stale(
                        workspace_id
                    )
                    log.info(
                        "memories_marked_stale_for_delta",
                        marked_stale=marked_stale,
                        workspace_id=workspace_id,
                    )
                except Exception as e:
                    log.warning(
                        "failed_to_mark_stale",
                        error=str(e),
                    )

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

            # Step 4: Store code as memories with AST parsing (delta indexing)
            memories_created = 0
            functions_found = 0
            classes_found = 0
            calls_found = 0
            if hasattr(self.memory_processor, "db_client") and self.memory_processor.db_client:
                memories_created, functions_found, classes_found, calls_found = (
                    await self._store_code_memories_with_delta(files, workspace_id)
                )

            # Step 5: Calculate statistics with actual AST counts
            stats = self._calculate_statistics(
                files, index_result, functions_found, classes_found, calls_found
            )

            # Step 5.5: Count remaining stale memories
            stale_count = 0
            if hasattr(self.memory_processor, "db_client") and self.memory_processor.db_client:
                try:
                    stale_count = await self.memory_processor.db_client.count_stale_memories(
                        workspace_id
                    )
                    if stale_count > 0:
                        log.info(
                            "stale_memories_detected",
                            stale_count=stale_count,
                            hint="Use prune_memory tool to clean up",
                        )
                except Exception as e:
                    log.warning(
                        "failed_to_count_stale",
                        error=str(e),
                    )

            # Step 6: Format success response
            processing_time_ms = (time.time() - start_time) * 1000

            log.info(
                "repository_indexed_successfully",
                files_indexed=stats["files_indexed"],
                functions=stats["functions_found"],
                classes=stats["classes_found"],
                calls=stats["calls_found"],
                memories_created=memories_created,
                stale_count=stale_count,
                processing_time_ms=processing_time_ms,
            )

            return self._format_success(
                repo_path=repo_path,
                files_indexed=stats["files_indexed"],
                functions_found=stats["functions_found"],
                classes_found=stats["classes_found"],
                calls_found=stats["calls_found"],
                languages=stats["languages"],
                total_lines=stats["total_lines"],
                processing_time_ms=processing_time_ms,
                stale_count=stale_count,
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
        except ValidationError:
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

    def _extension_to_language(self, extension: str) -> str:
        """
        Convert file extension to language name.

        Args:
            extension: File extension (e.g., ".py", ".ts")

        Returns:
            Language name (e.g., "python", "typescript")
        """
        ext = extension.lower()
        for lang, exts in LANGUAGE_EXTENSIONS.items():
            if ext in exts:
                return lang
        return "unknown"

    def _parse_file_ast(
        self,
        file_path: str,
        content: bytes,
    ) -> Dict[str, Any]:
        """
        Parse file with Tree-sitter and extract functions/classes.

        Uses Tree-sitter's AST parsing to extract structured code elements
        (functions, methods, classes) from source files. Falls back gracefully
        if Tree-sitter is not available.

        Args:
            file_path: Path to the source file.
            content: File content as bytes.

        Returns:
            Dictionary containing:
                - functions: List of ExtractedCode objects for functions/methods
                - classes: List of ExtractedCode objects for classes
                - tree: Parsed tree-sitter Tree (for call graph analysis)
                - language: Detected language name
                - errors: List of error messages (if any)
        """
        if not TREESITTER_AVAILABLE:
            return {
                "functions": [],
                "classes": [],
                "tree": None,
                "language": None,
                "errors": ["Tree-sitter not available"],
            }

        try:
            # Get parser for this file type
            parser_wrapper = ParserFactory.get_parser_for_file(file_path)
            if parser_wrapper is None:
                return {
                    "functions": [],
                    "classes": [],
                    "tree": None,
                    "language": None,
                    "errors": [f"No parser available for {file_path}"],
                }

            # Get the actual tree-sitter parser and parse the content
            ts_parser = parser_wrapper.get_parser()
            tree = ts_parser.parse(content)

            if tree is None or tree.root_node is None:
                return {
                    "functions": [],
                    "classes": [],
                    "tree": None,
                    "language": None,
                    "errors": [f"Parse failed for {file_path}"],
                }

            # Get language-specific extractor with fallback to generic
            from pathlib import Path as PathLib

            extension = PathLib(file_path).suffix
            language = self._extension_to_language(extension)
            registry = LanguageParserRegistry()
            extractor = registry.get_extractor(language)
            if extractor is None:
                extractor = GenericExtractor()

            self.logger.debug(
                "using_extractor",
                file=file_path,
                language=language,
                extractor_type=type(extractor).__name__,
            )

            functions = extractor.extract_functions(tree, content, file_path)
            classes = extractor.extract_classes(tree, content, file_path)

            self.logger.debug(
                "ast_parse_complete",
                file=file_path,
                functions=len(functions),
                classes=len(classes),
            )

            return {
                "functions": functions,
                "classes": classes,
                "tree": tree,
                "language": language,
                "errors": [],
            }

        except Exception as e:
            self.logger.warning(
                "ast_parse_error",
                file=file_path,
                error=str(e),
            )
            return {
                "functions": [],
                "classes": [],
                "tree": None,
                "language": None,
                "errors": [str(e)],
            }

    def _calculate_statistics(
        self,
        files: List[Dict[str, Any]],
        full_result: Dict[str, Any],
        functions_found: int = 0,
        classes_found: int = 0,
        calls_found: int = 0,
    ) -> Dict[str, Any]:
        """
        Calculate statistics from indexed files.

        Args:
            files: Filtered list of files
            full_result: Full result from indexer including statistics
            functions_found: Number of functions extracted via AST parsing
            classes_found: Number of classes extracted via AST parsing
            calls_found: Number of call relationships extracted via call graph analysis

        Returns:
            Dictionary with calculated statistics
        """
        languages: Dict[str, int] = {}
        stats: Dict[str, Any] = {
            "files_indexed": len(files),
            "functions_found": functions_found,
            "classes_found": classes_found,
            "calls_found": calls_found,
            "languages": languages,
            "total_lines": 0,
        }

        # Count files by language
        for file_dict in files:
            ext = file_dict.get("extension", "").lower()

            # Find language for this extension
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    languages[lang] = languages.get(lang, 0) + 1
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

    async def _store_code_memories_with_delta(
        self,
        files: List[Dict[str, Any]],
        workspace_id: str,
    ) -> Tuple[int, int, int, int]:
        """
        Store code files as memories with delta indexing support.

        For each file:
        1. Parse AST to extract functions and classes
        2. Store each function/class as separate memory with rich metadata
        3. Analyze call graph and store CALLS relationships
        4. Fall back to whole-file storage if AST parsing not available
        5. Track metrics

        Args:
            files: List of code file dictionaries
            workspace_id: Workspace ID for operations

        Returns:
            Tuple of (memories_created, functions_found, classes_found, calls_found)
        """
        from datetime import datetime, timezone
        from pathlib import Path

        memories_created = 0
        memories_refreshed = 0
        total_functions = 0
        total_classes = 0
        total_calls = 0

        db_client = self.memory_processor.db_client

        for file_dict in files:
            try:
                file_path = file_dict.get("path")
                relative_path = file_dict.get("relative_path")
                extension = file_dict.get("extension")
                lines = file_dict.get("lines", 0)

                if not file_path:
                    continue

                # Read file content
                try:
                    content_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    content_bytes = content_text.encode("utf-8")
                except Exception as read_error:
                    self.logger.warning(
                        "failed_to_read_file_content",
                        file_path=file_path,
                        error=str(read_error),
                    )
                    continue

                if not content_text.strip():
                    self.logger.debug(
                        "empty_file_skipped",
                        file_path=file_path,
                    )
                    continue

                # Determine language from extension
                language = self._extension_to_language(extension if extension else "")

                # Try AST parsing for granular extraction
                ast_result = self._parse_file_ast(file_path, content_bytes)

                functions = ast_result.get("functions", [])
                classes = ast_result.get("classes", [])

                total_functions += len(functions)
                total_classes += len(classes)

                # Store each function as separate memory
                for func in functions:
                    try:
                        # Format function text with context
                        start_l = func.location.start_line + 1
                        end_l = func.location.end_line + 1
                        text_to_store = (
                            f"# Function: {func.name}\n"
                            f"# File: {relative_path}\n"
                            f"# Language: {language}\n"
                            f"# Lines: {start_l}-{end_l}\n"
                        )
                        if func.parent_class:
                            text_to_store += f"# Class: {func.parent_class}\n"
                        text_to_store += f"\n{func.source_code}"

                        metadata = {
                            "source": "code_indexer",
                            "element_type": func.element_type.value,
                            "element_name": func.name,
                            "qualified_name": func.qualified_name,
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "extension": extension,
                            "language": language,
                            "start_line": func.location.start_line + 1,
                            "end_line": func.location.end_line + 1,
                            "line_count": func.line_count,
                            "is_async": func.is_async,
                            "is_private": func.is_private,
                            "indexed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        if func.parent_class:
                            metadata["parent_class"] = func.parent_class

                        await self.memory_processor.add_memory(text_to_store, metadata)
                        memories_created += 1

                        self.logger.debug(
                            "function_memory_created",
                            name=func.name,
                            file=relative_path,
                        )

                    except Exception as func_error:
                        self.logger.warning(
                            "failed_to_store_function",
                            name=func.name,
                            file=file_path,
                            error=str(func_error),
                        )

                # Store each class as separate memory
                for cls in classes:
                    try:
                        # Format class text with context
                        text_to_store = (
                            f"# Class: {cls.name}\n"
                            f"# File: {relative_path}\n"
                            f"# Language: {language}\n"
                            f"# Lines: {cls.location.start_line + 1}-{cls.location.end_line + 1}\n"
                        )
                        if cls.methods:
                            text_to_store += f"# Methods: {', '.join(cls.methods)}\n"
                        if cls.bases:
                            text_to_store += f"# Bases: {', '.join(cls.bases)}\n"
                        text_to_store += f"\n{cls.source_code}"

                        metadata = {
                            "source": "code_indexer",
                            "element_type": cls.element_type.value,
                            "element_name": cls.name,
                            "qualified_name": cls.qualified_name,
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "extension": extension,
                            "language": language,
                            "start_line": cls.location.start_line + 1,
                            "end_line": cls.location.end_line + 1,
                            "line_count": cls.line_count,
                            "is_private": cls.is_private,
                            "indexed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        if cls.methods:
                            metadata["methods"] = cls.methods
                        if cls.bases:
                            metadata["bases"] = cls.bases

                        await self.memory_processor.add_memory(text_to_store, metadata)
                        memories_created += 1

                        self.logger.debug(
                            "class_memory_created",
                            name=cls.name,
                            file=relative_path,
                        )

                    except Exception as cls_error:
                        self.logger.warning(
                            "failed_to_store_class",
                            name=cls.name,
                            file=file_path,
                            error=str(cls_error),
                        )

                # Analyze and store call graph relationships
                tree = ast_result.get("tree")
                ast_language = ast_result.get("language")
                if (
                    self._call_graph_analyzer is not None
                    and tree is not None
                    and ast_language is not None
                    and ast_language in self._call_graph_analyzer.CALL_NODE_TYPES
                    and functions
                ):
                    try:
                        call_graph = self._call_graph_analyzer.analyze_file(
                            tree, content_bytes, file_path, ast_language, functions
                        )
                        if call_graph.calls:
                            # Prepare batch for storing
                            calls_batch = [
                                {
                                    "caller": call.caller_qualified_name,
                                    "callee": call.callee_name,
                                    "line": call.location.start_line,
                                    "type": call.call_type.value,
                                    "args": call.arguments_count,
                                }
                                for call in call_graph.calls
                            ]
                            stored_count = await db_client.add_calls_batch(
                                calls_batch, workspace_id
                            )
                            total_calls += stored_count
                            self.logger.debug(
                                "call_graph_stored",
                                file=relative_path,
                                calls_found=len(call_graph.calls),
                                calls_stored=stored_count,
                            )
                    except Exception as cg_error:
                        self.logger.warning(
                            "call_graph_analysis_failed",
                            file=file_path,
                            error=str(cg_error),
                        )

                # If no AST elements extracted, fall back to whole-file storage
                if not functions and not classes:
                    # Try to mark existing memory as fresh
                    existing_id = await db_client.mark_memory_fresh(
                        file_path=file_path,
                        workspace_id=workspace_id,
                    )

                    if existing_id:
                        memories_refreshed += 1
                        self.logger.debug(
                            "memory_refreshed",
                            file_path=file_path,
                            memory_id=existing_id,
                        )
                    else:
                        # Store whole file as fallback
                        text_to_store = (
                            f"# File: {relative_path}\n"
                            f"# Language: {language}\n"
                            f"# Lines: {lines}\n\n"
                            f"{content_text}"
                        )

                        metadata = {
                            "source": "code_indexer",
                            "element_type": "file",
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "extension": extension,
                            "language": language,
                            "lines": lines,
                            "indexed_at": datetime.now(timezone.utc).isoformat(),
                        }

                        await self.memory_processor.add_memory(text_to_store, metadata)
                        memories_created += 1
                        self.logger.debug(
                            "file_memory_created",
                            file_path=file_path,
                        )

            except Exception as e:
                self.logger.warning(
                    "failed_to_process_file_for_delta",
                    file_path=file_dict.get("path"),
                    error=str(e),
                )
                continue

        self.logger.info(
            "delta_indexing_complete",
            memories_created=memories_created,
            memories_refreshed=memories_refreshed,
            functions_found=total_functions,
            classes_found=total_classes,
            calls_found=total_calls,
            workspace_id=workspace_id,
        )

        return memories_created, total_functions, total_classes, total_calls

    def _format_success(
        self,
        repo_path: str,
        files_indexed: int,
        functions_found: int,
        classes_found: int,
        calls_found: int,
        languages: Dict[str, int],
        total_lines: int,
        processing_time_ms: float,
        stale_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Format successful indexing as MCP response.

        Args:
            repo_path: Repository path that was indexed
            files_indexed: Number of files indexed
            functions_found: Number of functions extracted
            classes_found: Number of classes extracted
            calls_found: Number of call relationships extracted
            languages: Dictionary of language counts
            total_lines: Total lines of code
            processing_time_ms: Processing time in milliseconds
            stale_count: Count of stale memories (deleted files)

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
            f"Call relationships: {calls_found}\n"
            f"Languages: {lang_summary}\n"
            f"Total lines: {total_lines:,}\n"
            f"Indexing time: {processing_time_sec:.2f}s"
        )

        # Add stale count warning if there are stale memories
        if stale_count > 0:
            message += (
                f"\n\n[!] {stale_count} stale memories detected (deleted files).\n"
                f"Run prune_memory(strategy='stale_code') to clean up."
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
