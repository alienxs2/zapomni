"""Base classes for code element extraction from AST.

This module defines the abstract base class for code extractors that
analyze tree-sitter AST trees and extract structured code elements
like functions and classes.
"""

from abc import ABC, abstractmethod
from typing import List

import structlog
from tree_sitter import Node, Tree

from ..models import ASTNodeLocation, CodeElementType, ExtractedCode, ParameterInfo

logger = structlog.get_logger(__name__)


class BaseCodeExtractor(ABC):
    """Abstract base class for extracting code elements from AST trees.

    Subclasses must implement language-specific extraction logic for
    functions and classes. This base class provides common helper methods
    for text extraction, location creation, and name analysis.

    Attributes:
        language_name: Name of the programming language this extractor handles.
        supported_node_types: Tuple of AST node types this extractor processes.

    Example:
        >>> class PythonExtractor(BaseCodeExtractor):
        ...     @property
        ...     def language_name(self) -> str:
        ...         return "python"
        ...
        ...     @property
        ...     def supported_node_types(self) -> tuple[str, ...]:
        ...         return ("function_definition", "class_definition")
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the name of the programming language.

        Returns:
            Language name in lowercase (e.g., "python", "javascript").
        """
        ...

    @property
    @abstractmethod
    def supported_node_types(self) -> tuple[str, ...]:
        """Return the AST node types this extractor can process.

        These are tree-sitter node type names that identify extractable
        code elements (e.g., "function_definition", "class_definition").

        Returns:
            Tuple of supported node type strings.
        """
        ...

    @abstractmethod
    def extract_functions(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """Extract all function definitions from the AST tree.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Absolute path to the source file.

        Returns:
            List of ExtractedCode objects representing functions.
        """
        ...

    @abstractmethod
    def extract_classes(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """Extract all class definitions from the AST tree.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Absolute path to the source file.

        Returns:
            List of ExtractedCode objects representing classes.
        """
        ...

    def extract_all(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """Extract all code elements (functions and classes) from the AST tree.

        Combines results from extract_functions() and extract_classes(),
        then sorts them by their starting line number for consistent ordering.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Absolute path to the source file.

        Returns:
            List of ExtractedCode objects sorted by start_line.
        """
        logger.debug(
            "extracting_all_elements",
            language=self.language_name,
            file_path=file_path,
        )

        functions = self.extract_functions(tree, source, file_path)
        classes = self.extract_classes(tree, source, file_path)

        # Combine and sort by start line
        all_elements = functions + classes
        all_elements.sort(key=lambda x: x.location.start_line)

        logger.debug(
            "extraction_complete",
            language=self.language_name,
            file_path=file_path,
            functions_count=len(functions),
            classes_count=len(classes),
            total_count=len(all_elements),
        )

        return all_elements

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """Extract text content of a node from source bytes.

        Uses byte offsets from the node to extract the exact source
        text. Handles encoding errors gracefully by replacing
        invalid characters.

        Args:
            node: Tree-sitter AST node.
            source: Original source code as bytes.

        Returns:
            String content of the node.
        """
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _create_location(self, node: Node) -> ASTNodeLocation:
        """Create ASTNodeLocation from tree-sitter Node.

        Extracts position information from the tree-sitter node
        and creates an immutable location object.

        Args:
            node: Tree-sitter AST node.

        Returns:
            ASTNodeLocation with line, column, and byte positions.
        """
        return ASTNodeLocation(
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )

    def _is_private_name(self, name: str) -> bool:
        """Check if name indicates private member (starts with _).

        In many languages, a single underscore prefix indicates
        a private or internal member. Double underscore (dunder)
        methods are NOT considered private as they are special
        methods.

        Args:
            name: The name to check.

        Returns:
            True if name starts with single underscore (not double).
        """
        return name.startswith("_") and not name.startswith("__")
