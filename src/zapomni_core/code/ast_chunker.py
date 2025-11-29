"""
ASTCodeChunker: AST-level code chunking for Zapomni Phase 3.

Chunks code files at the Abstract Syntax Tree level, extracting functions,
classes, and module-level definitions. Supports Python natively with fallback
to line-based chunking for unsupported languages.

Author: Zapomni Team
License: MIT
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_db.models import Chunk

logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Supported programming languages for AST chunking."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


@dataclass
class CodeMetadata:
    """Metadata extracted from a code node."""

    name: str
    node_type: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    decorators: List[str] = None
    is_private: bool = False
    is_async: bool = False

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.decorators is None:
            self.decorators = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "node_type": self.node_type,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "is_private": self.is_private,
            "is_async": self.is_async,
        }


class ASTCodeChunker:
    """
    Chunks code files at the Abstract Syntax Tree level.

    Extracts functions, classes, and module-level definitions from code,
    creating semantically meaningful chunks. Currently supports Python;
    other languages fall back to line-based chunking.

    Attributes:
        language: Programming language being chunked
        min_chunk_lines: Minimum lines for a chunk (default: 2)
    """

    def __init__(self, language: str = "python", min_chunk_lines: int = 2) -> None:
        """
        Initialize ASTCodeChunker for a specific language.

        Args:
            language: Programming language ("python", "javascript", "typescript")
            min_chunk_lines: Minimum number of lines for a chunk (default: 2)

        Raises:
            ValidationError: If language is not supported
            ValueError: If min_chunk_lines < 1
        """
        language_lower = language.lower().strip()

        # Validate language
        try:
            self.language = SupportedLanguage(language_lower)
        except ValueError:
            raise ValidationError(
                message=f"Unsupported language: {language}. "
                "Supported: python, javascript, typescript",
                error_code="VAL_002",
            )

        if min_chunk_lines < 1:
            raise ValueError("min_chunk_lines must be >= 1")

        self.min_chunk_lines = min_chunk_lines
        logger.debug(f"Initialized ASTCodeChunker for {self.language.value}")

    def chunk_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> List[Chunk]:
        """
        Parse and chunk a code file.

        Args:
            file_path: Path to the code file
            content: Optional file content (if None, reads from file_path)

        Returns:
            List of Chunk objects representing code segments

        Raises:
            ValidationError: If file_path or content is invalid
            ProcessingError: If AST parsing fails
        """
        file_path = Path(file_path)

        # Validate file_path
        if not isinstance(file_path, Path):
            raise ValidationError(
                message="file_path must be a string or Path object",
                error_code="VAL_002",
            )

        # Load content if not provided
        if content is None:
            if not file_path.exists():
                raise ValidationError(
                    message=f"File not found: {file_path}",
                    error_code="VAL_001",
                )
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                raise ProcessingError(
                    message=f"Failed to read file {file_path}: {exc}",
                    error_code="PROC_002",
                    original_exception=exc,
                )

        # Validate content
        if not isinstance(content, str):
            raise ValidationError(
                message="content must be a string",
                error_code="VAL_002",
            )

        if not content.strip():
            raise ValidationError(
                message="content cannot be empty",
                error_code="VAL_001",
            )

        logger.debug(f"Chunking file: {file_path} ({len(content)} chars)")

        # Route to appropriate chunker
        if self.language == SupportedLanguage.PYTHON:
            return self._chunk_python(content, file_path)
        else:
            # Fallback: line-based chunking for unsupported languages
            logger.info(f"Falling back to line-based chunking for {self.language.value}")
            return self._chunk_lines(content, file_path)

    def _chunk_python(self, content: str, file_path: Path) -> List[Chunk]:
        """
        Chunk Python code using AST parsing.

        Args:
            content: Python source code
            file_path: Path to the file (for reference)

        Returns:
            List of Chunk objects

        Raises:
            ProcessingError: If AST parsing fails
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ProcessingError(
                message=f"Python syntax error in {file_path}: {exc}",
                error_code="PROC_001",
                original_exception=exc,
            )
        except Exception as exc:
            raise ProcessingError(
                message=f"AST parsing failed for {file_path}: {exc}",
                error_code="PROC_001",
                original_exception=exc,
            )

        chunks: List[Chunk] = []
        lines = content.split("\n")

        # Module docstring extracted for future metadata use
        # (currently unused but kept for API expansion)
        ast.get_docstring(tree)

        # Extract all top-level definitions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function_chunk(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_class_chunk(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                chunk = self._extract_import_chunk(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)

        # If no chunks extracted, fall back to line-based chunking
        if not chunks:
            logger.debug("No AST nodes extracted; falling back to line-based chunking")
            return self._chunk_lines(content, file_path)

        # Reindex chunks sequentially
        for idx, chunk in enumerate(chunks):
            chunk.index = idx

        logger.info(f"Chunked Python file into {len(chunks)} AST-based chunks")
        return chunks

    def _extract_function_chunk(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], lines: List[str], file_path: Path
    ) -> Optional[Chunk]:
        """
        Extract a function definition as a chunk.

        Args:
            node: AST function node
            lines: Source code lines
            file_path: Path to the file

        Returns:
            Chunk object or None if too small
        """
        metadata = self.get_chunk_metadata(node)

        # Extract source lines
        start_line = node.lineno - 1  # 0-indexed
        end_line = node.end_lineno  # inclusive

        if end_line is None:
            # Fallback for older Python versions
            end_line = start_line + 5

        # Validate line range
        if start_line < 0 or end_line > len(lines):
            logger.warning(f"Invalid line range for {metadata.name}")
            return None

        # Extract text
        chunk_lines = lines[start_line:end_line]
        text = "\n".join(chunk_lines)

        # Check minimum size
        if len(chunk_lines) < self.min_chunk_lines:
            logger.debug(f"Skipping small function: {metadata.name} ({len(chunk_lines)} lines)")
            return None

        return Chunk(
            text=text,
            index=0,  # Reindexed later
            start_char=self._calculate_char_offset(lines, start_line),
            end_char=self._calculate_char_offset(lines, end_line),
            metadata={
                **metadata.to_dict(),
                "file_path": str(file_path),
                "chunk_type": "function",
            },
        )

    def _extract_class_chunk(
        self, node: ast.ClassDef, lines: List[str], file_path: Path
    ) -> Optional[Chunk]:
        """
        Extract a class definition as a chunk.

        Args:
            node: AST class node
            lines: Source code lines
            file_path: Path to the file

        Returns:
            Chunk object or None if too small
        """
        metadata = self.get_chunk_metadata(node)

        # Extract source lines
        start_line = node.lineno - 1  # 0-indexed
        end_line = node.end_lineno  # inclusive

        if end_line is None:
            # Fallback for older Python versions
            end_line = start_line + 10

        # Validate line range
        if start_line < 0 or end_line > len(lines):
            logger.warning(f"Invalid line range for {metadata.name}")
            return None

        # Extract text
        chunk_lines = lines[start_line:end_line]
        text = "\n".join(chunk_lines)

        # Check minimum size
        if len(chunk_lines) < self.min_chunk_lines:
            logger.debug(f"Skipping small class: {metadata.name} ({len(chunk_lines)} lines)")
            return None

        return Chunk(
            text=text,
            index=0,  # Reindexed later
            start_char=self._calculate_char_offset(lines, start_line),
            end_char=self._calculate_char_offset(lines, end_line),
            metadata={
                **metadata.to_dict(),
                "file_path": str(file_path),
                "chunk_type": "class",
                "methods_count": len(
                    [
                        n
                        for n in ast.walk(node)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                ),
            },
        )

    def _extract_import_chunk(
        self, node: Union[ast.Import, ast.ImportFrom], lines: List[str], file_path: Path
    ) -> Optional[Chunk]:
        """
        Extract an import statement as a chunk.

        Args:
            node: AST import node
            lines: Source code lines
            file_path: Path to the file

        Returns:
            Chunk object or None
        """
        start_line = node.lineno - 1  # 0-indexed
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        # Extract text
        chunk_lines = lines[start_line:end_line]
        text = "\n".join(chunk_lines)

        # Extract import names
        import_names = []
        if isinstance(node, ast.Import):
            import_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            import_names = [alias.name for alias in node.names]

        return Chunk(
            text=text,
            index=0,  # Reindexed later
            start_char=self._calculate_char_offset(lines, start_line),
            end_char=self._calculate_char_offset(lines, end_line),
            metadata={
                "name": "imports",
                "node_type": "import",
                "line_start": start_line + 1,
                "line_end": end_line,
                "file_path": str(file_path),
                "chunk_type": "import",
                "imports": import_names,
            },
        )

    def _chunk_lines(self, content: str, file_path: Path) -> List[Chunk]:
        """
        Fallback: chunk code by lines (for unsupported languages).

        Creates chunks of approximately 20-30 lines with 5-line overlap.

        Args:
            content: Source code
            file_path: Path to the file

        Returns:
            List of Chunk objects
        """
        lines = content.split("\n")
        chunks: List[Chunk] = []

        chunk_size = 20
        overlap = 5
        step = chunk_size - overlap

        current_offset = 0
        chunk_index = 0

        for start in range(0, len(lines), step):
            end = min(start + chunk_size, len(lines))

            chunk_lines = lines[start:end]
            text = "\n".join(chunk_lines)

            start_char = current_offset
            end_char = start_char + len(text)

            chunk = Chunk(
                text=text,
                index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "line_start": start + 1,
                    "line_end": end,
                    "file_path": str(file_path),
                    "chunk_type": "line_based",
                    "language": self.language.value,
                },
            )
            chunks.append(chunk)

            current_offset = end_char + 1  # +1 for newline
            chunk_index += 1

            if end >= len(lines):
                break

        logger.info(f"Chunked file into {len(chunks)} line-based chunks")
        return chunks

    def extract_functions(self, ast_tree: ast.Module) -> List[ast.FunctionDef]:
        """
        Extract all function definitions from an AST tree.

        Args:
            ast_tree: AST Module object

        Returns:
            List of FunctionDef and AsyncFunctionDef nodes
        """
        functions: List[ast.FunctionDef] = []

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)

        logger.debug(f"Extracted {len(functions)} functions from AST")
        return functions

    def extract_classes(self, ast_tree: ast.Module) -> List[ast.ClassDef]:
        """
        Extract all class definitions from an AST tree.

        Args:
            ast_tree: AST Module object

        Returns:
            List of ClassDef nodes
        """
        classes: List[ast.ClassDef] = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node)

        logger.debug(f"Extracted {len(classes)} classes from AST")
        return classes

    def get_chunk_metadata(self, node: ast.AST) -> CodeMetadata:
        """
        Extract metadata from an AST node.

        Args:
            node: AST node (FunctionDef, AsyncFunctionDef, or ClassDef)

        Returns:
            CodeMetadata object with name, type, line range, docstring, etc.

        Raises:
            ValueError: If node type is unsupported
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [
                ast.unparse(dec) if hasattr(ast, "unparse") else self._get_decorator_name(dec)
                for dec in node.decorator_list
            ]
            is_private = node.name.startswith("_")
            is_async = isinstance(node, ast.AsyncFunctionDef)

            return CodeMetadata(
                name=node.name,
                node_type="async_function" if is_async else "function",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno + 5,
                docstring=ast.get_docstring(node),
                decorators=decorators,
                is_private=is_private,
                is_async=is_async,
            )

        elif isinstance(node, ast.ClassDef):
            return CodeMetadata(
                name=node.name,
                node_type="class",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno + 10,
                docstring=ast.get_docstring(node),
                is_private=node.name.startswith("_"),
            )

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    @staticmethod
    def _get_decorator_name(decorator: ast.expr) -> str:
        """
        Extract decorator name from AST node (fallback for older Python versions).

        Args:
            decorator: AST decorator expression

        Returns:
            Decorator name as string
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown"

    @staticmethod
    def _calculate_char_offset(lines: List[str], line_index: int) -> int:
        """
        Calculate character offset for a line index.

        Args:
            lines: List of source code lines
            line_index: 0-based line index

        Returns:
            Character offset from start of file
        """
        if line_index < 0 or line_index > len(lines):
            return 0

        offset = 0
        for i in range(line_index):
            offset += len(lines[i]) + 1  # +1 for newline

        return offset
