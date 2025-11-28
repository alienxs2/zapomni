"""
Generic code extractor that works with any language.

This module provides a universal code extractor that can extract functions,
methods, classes, and other code elements from any programming language
supported by tree-sitter. It serves as a fallback when no language-specific
extractor is available.

The extractor works by searching for common AST node types that are shared
across many programming languages. This approach allows it to extract
meaningful code structures even for languages without dedicated extractors.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import List, Optional, Set

import structlog
from tree_sitter import Node, Tree

from .base import BaseCodeExtractor
from ..models import ASTNodeLocation, CodeElementType, ExtractedCode

logger = structlog.get_logger(__name__)

# =============================================================================
# Common AST Node Types
# =============================================================================

# Function-like node types found across various programming languages
FUNCTION_NODE_TYPES: Set[str] = {
    # Common function definitions
    "function_definition",      # Python, C, C++, etc.
    "function_declaration",     # JavaScript, Go, C, etc.
    "method_definition",        # Python, Ruby, JavaScript
    "method_declaration",       # Java, C#, Kotlin
    "function",                 # Lua, Haskell
    "func_literal",             # Go (anonymous functions)
    "arrow_function",           # JavaScript, TypeScript
    "lambda",                   # Python lambda
    "lambda_expression",        # Java, C#
    "function_item",            # Rust
    "fun_spec",                 # Erlang
    "function_expression",      # JavaScript
    "generator_function",       # JavaScript (function*)
    "generator_function_declaration",  # JavaScript
    "async_function",           # JavaScript (async function)
    "async_function_declaration",  # JavaScript
    "method",                   # Ruby, Smalltalk
    "def",                      # Some languages use 'def' node type
    "procedure_declaration",    # Pascal, Ada
    "subroutine",               # Fortran
    "fn_item",                  # Rust (alternative)
    "function_literal",         # Kotlin
    "anonymous_function",       # Various languages
    "closure_expression",       # Swift
    "block",                    # Ruby blocks (procs)
}

# Class-like node types found across various programming languages
CLASS_NODE_TYPES: Set[str] = {
    # Class definitions
    "class_definition",         # Python
    "class_declaration",        # Java, JavaScript, TypeScript, C#
    "class",                    # Ruby
    "class_specifier",          # C++
    # Struct definitions
    "struct_item",              # Rust
    "struct_definition",        # C
    "struct_declaration",       # Go
    "struct_specifier",         # C, C++
    "struct",                   # Various languages
    # Interface definitions
    "interface_declaration",    # Java, TypeScript, Go
    "interface_definition",     # Various
    "interface",                # Ruby (module as interface)
    "protocol",                 # Swift, Objective-C
    "trait_item",               # Rust
    "trait_definition",         # Scala
    # Implementation blocks
    "impl_item",                # Rust
    "impl_block",               # Rust (alternative)
    # Type definitions
    "type_declaration",         # Go (type X struct{})
    "type_definition",          # C (typedef)
    "type_alias",               # TypeScript, Rust
    "type_spec",                # Go
    # Module definitions (often class-like)
    "module",                   # Elixir, Ruby, Erlang
    "module_definition",        # Various
    "namespace_definition",     # C++
    "package_declaration",      # Java
    # Enum definitions
    "enum_declaration",         # Java, TypeScript
    "enum_definition",          # Rust
    "enum_specifier",           # C, C++
    "enum_item",                # Rust
    # Object definitions
    "object_declaration",       # Kotlin, Scala
    "singleton_class",          # Ruby
}

# Node types that indicate the name/identifier of a definition
NAME_NODE_TYPES: Set[str] = {
    "identifier",
    "name",
    "property_identifier",
    "type_identifier",
    "constant",                 # Ruby constants as class names
    "simple_identifier",        # Kotlin
    "word",                     # Some languages
}


class GenericExtractor(BaseCodeExtractor):
    """
    Universal code extractor that works with any programming language.

    This extractor searches the AST for common node types that represent
    functions, methods, classes, and other code structures. It serves as
    a fallback when no language-specific extractor is available.

    The extraction strategy is based on pattern matching against known
    AST node types from various tree-sitter grammars. This approach
    allows it to work with 165+ languages supported by tree-sitter.

    Attributes:
        _log: Structured logger for this extractor instance.

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("python")
        >>> tree = parser.parse(b"def hello(): pass")
        >>> extractor = GenericExtractor()
        >>> functions = extractor.extract_functions(tree, b"def hello(): pass", "test.py")
        >>> len(functions)
        1
        >>> functions[0].name
        'hello'
    """

    def __init__(self) -> None:
        """Initialize the generic extractor with logging."""
        super().__init__()
        self._log = logger.bind(extractor="GenericExtractor")
        self._log.debug("GenericExtractor initialized")

    @property
    def language_name(self) -> str:
        """
        Get the language name for this extractor.

        Returns:
            "generic" - indicates this is a universal fallback extractor.
        """
        return "generic"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        """
        Get all AST node types supported by this extractor.

        Returns:
            Tuple of all function and class node types that this extractor
            can recognize and process.
        """
        return tuple(FUNCTION_NODE_TYPES | CLASS_NODE_TYPES)

    def extract_functions(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract function-like definitions from the AST.

        Traverses the entire AST tree searching for nodes that match
        known function definition patterns. Each found function is
        converted to an ExtractedCode object with full metadata.

        Args:
            tree: Parsed AST tree from tree-sitter.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects representing functions/methods.

        Example:
            >>> tree = parser.parse(b"def foo(): pass\\ndef bar(): pass")
            >>> funcs = extractor.extract_functions(tree, source, "test.py")
            >>> [f.name for f in funcs]
            ['foo', 'bar']
        """
        results: List[ExtractedCode] = []

        def visit(node: Node) -> None:
            """Recursively visit AST nodes to find functions."""
            if node.type in FUNCTION_NODE_TYPES:
                extracted = self._extract_function_node(node, source, file_path)
                if extracted:
                    results.append(extracted)
                    self._log.debug(
                        "function_extracted",
                        name=extracted.name,
                        node_type=node.type,
                        file=file_path,
                    )

            # Continue traversing children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "function_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def extract_classes(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract class-like definitions from the AST.

        Traverses the entire AST tree searching for nodes that match
        known class, struct, interface, or module definition patterns.

        Args:
            tree: Parsed AST tree from tree-sitter.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects representing classes/interfaces.

        Example:
            >>> tree = parser.parse(b"class Foo:\\n    pass")
            >>> classes = extractor.extract_classes(tree, source, "test.py")
            >>> [c.name for c in classes]
            ['Foo']
        """
        results: List[ExtractedCode] = []

        def visit(node: Node) -> None:
            """Recursively visit AST nodes to find classes."""
            if node.type in CLASS_NODE_TYPES:
                extracted = self._extract_class_node(node, source, file_path)
                if extracted:
                    results.append(extracted)
                    self._log.debug(
                        "class_extracted",
                        name=extracted.name,
                        node_type=node.type,
                        file=file_path,
                    )

            # Continue traversing children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "class_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def _extract_function_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a function definition from an AST node.

        Converts a function-like AST node into an ExtractedCode object
        with name, location, source code, and flags.

        Args:
            node: AST node representing a function.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            ExtractedCode object if extraction successful, None otherwise.
        """
        # Find the function name
        name_node = self._find_name_node(node)
        if name_node is None:
            self._log.debug(
                "function_name_not_found",
                node_type=node.type,
                file=file_path,
            )
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Get source code
        source_code = self._get_node_text(node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(node)

        # Determine element type (method if inside a class, function otherwise)
        element_type = self._determine_function_type(node)

        # Check for async
        is_async = self._is_async_function(node)

        # Get parent class if this is a method
        parent_class = self._find_parent_class_name(node, source)

        # Build qualified name
        qualified_name = f"{parent_class}.{name}" if parent_class else name

        return ExtractedCode(
            name=name,
            qualified_name=qualified_name,
            element_type=element_type,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            parent_class=parent_class,
            is_async=is_async,
            is_private=name.startswith("_"),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_class_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a class definition from an AST node.

        Converts a class-like AST node into an ExtractedCode object
        with name, location, source code, and method names.

        Args:
            node: AST node representing a class/struct/interface.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            ExtractedCode object if extraction successful, None otherwise.
        """
        # Find the class name
        name_node = self._find_name_node(node)
        if name_node is None:
            self._log.debug(
                "class_name_not_found",
                node_type=node.type,
                file=file_path,
            )
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Get source code
        source_code = self._get_node_text(node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(node)

        # Determine element type
        element_type = self._determine_class_type(node)

        # Find method names within the class
        methods = self._find_method_names(node, source)

        # Find base classes
        bases = self._find_base_classes(node, source)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=element_type,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            methods=methods,
            bases=bases,
            is_private=name.startswith("_"),
            line_count=location.end_line - location.start_line + 1,
        )

    def _find_name_node(self, node: Node) -> Optional[Node]:
        """
        Find the identifier/name node within a definition.

        Searches the immediate children of the node for a name-like node.
        If not found in immediate children, performs a deeper search.

        Args:
            node: AST node to search within.

        Returns:
            Node containing the name/identifier if found, None otherwise.
        """
        # First, check immediate children
        for child in node.children:
            if child.type in NAME_NODE_TYPES:
                return child

        # Check for named children (tree-sitter specific)
        if hasattr(node, 'child_by_field_name'):
            # Try common field names
            for field_name in ('name', 'identifier', 'declarator'):
                field_child = node.child_by_field_name(field_name)
                if field_child is not None:
                    # The field might be a wrapper, look for identifier inside
                    if field_child.type in NAME_NODE_TYPES:
                        return field_child
                    # Search one level deeper
                    for subchild in field_child.children:
                        if subchild.type in NAME_NODE_TYPES:
                            return subchild

        # Fallback: search deeper (one more level)
        for child in node.children:
            result = self._find_name_in_children(child)
            if result:
                return result

        return None

    def _find_name_in_children(self, node: Node) -> Optional[Node]:
        """
        Search for a name node in the children of a node.

        Args:
            node: Node to search within.

        Returns:
            Name node if found, None otherwise.
        """
        for child in node.children:
            if child.type in NAME_NODE_TYPES:
                return child
        return None

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """
        Get the source text for an AST node.

        Args:
            node: AST node to extract text from.
            source: Original source code as bytes.

        Returns:
            Source text as string, empty string if extraction fails.
        """
        try:
            return source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        except Exception as e:
            self._log.warning(
                "text_extraction_failed",
                error=str(e),
                node_type=node.type,
            )
            return ""

    def _create_location(self, node: Node) -> ASTNodeLocation:
        """
        Create an ASTNodeLocation from a tree-sitter node.

        Args:
            node: AST node to create location from.

        Returns:
            ASTNodeLocation with line, column, and byte offset information.
        """
        return ASTNodeLocation(
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )

    def _determine_function_type(self, node: Node) -> CodeElementType:
        """
        Determine if a function node is a function or method.

        Args:
            node: Function AST node.

        Returns:
            CodeElementType.METHOD if inside a class, otherwise FUNCTION.
        """
        # Check node type hints
        if 'method' in node.type.lower():
            return CodeElementType.METHOD

        # Check if parent is a class-like node
        parent = node.parent
        while parent is not None:
            if parent.type in CLASS_NODE_TYPES:
                return CodeElementType.METHOD
            parent = parent.parent

        return CodeElementType.FUNCTION

    def _determine_class_type(self, node: Node) -> CodeElementType:
        """
        Determine the type of class-like node.

        Args:
            node: Class-like AST node.

        Returns:
            CodeElementType based on node type (CLASS, INTERFACE, STRUCT, etc.).
        """
        node_type_lower = node.type.lower()

        if 'interface' in node_type_lower or 'protocol' in node_type_lower:
            return CodeElementType.INTERFACE
        elif 'struct' in node_type_lower:
            return CodeElementType.STRUCT
        elif 'enum' in node_type_lower:
            return CodeElementType.ENUM
        elif 'module' in node_type_lower or 'namespace' in node_type_lower:
            return CodeElementType.MODULE
        elif 'trait' in node_type_lower:
            return CodeElementType.INTERFACE  # Traits are similar to interfaces

        return CodeElementType.CLASS

    def _is_async_function(self, node: Node) -> bool:
        """
        Check if a function node represents an async function.

        Args:
            node: Function AST node.

        Returns:
            True if the function is async, False otherwise.
        """
        # Check node type
        if 'async' in node.type.lower():
            return True

        # Check for async keyword in children
        for child in node.children:
            if child.type in ('async', 'async_keyword'):
                return True
            # Check text content
            if child.type == 'keyword' or child.is_named is False:
                try:
                    if child.text and child.text.decode('utf-8', errors='ignore') == 'async':
                        return True
                except Exception:
                    pass

        return False

    def _find_parent_class_name(self, node: Node, source: bytes) -> Optional[str]:
        """
        Find the name of the parent class for a method.

        Args:
            node: Method AST node.
            source: Original source code.

        Returns:
            Parent class name if found, None otherwise.
        """
        parent = node.parent
        while parent is not None:
            if parent.type in CLASS_NODE_TYPES:
                name_node = self._find_name_node(parent)
                if name_node:
                    return self._get_node_text(name_node, source)
            parent = parent.parent
        return None

    def _find_method_names(self, node: Node, source: bytes) -> List[str]:
        """
        Find all method names within a class node.

        Args:
            node: Class AST node.
            source: Original source code.

        Returns:
            List of method names found in the class.
        """
        methods: List[str] = []

        def visit(child: Node) -> None:
            if child.type in FUNCTION_NODE_TYPES:
                name_node = self._find_name_node(child)
                if name_node:
                    name = self._get_node_text(name_node, source)
                    if name:
                        methods.append(name)
                return  # Don't recurse into nested functions

            # Skip nested classes
            if child.type in CLASS_NODE_TYPES:
                return

            for subchild in child.children:
                visit(subchild)

        for child in node.children:
            visit(child)

        return methods

    def _find_base_classes(self, node: Node, source: bytes) -> List[str]:
        """
        Find base class names for a class definition.

        Args:
            node: Class AST node.
            source: Original source code.

        Returns:
            List of base class names.
        """
        bases: List[str] = []

        # Look for common inheritance patterns
        for child in node.children:
            # Python: argument_list after class name contains bases
            if child.type in (
                'argument_list',
                'superclass',
                'base_class_clause',
                'extends_clause',
                'implements_clause',
                'superclasses',
                'type_parameters',
            ):
                for subchild in child.children:
                    if subchild.type in NAME_NODE_TYPES:
                        base_name = self._get_node_text(subchild, source)
                        if base_name and base_name not in ('(', ')', ','):
                            bases.append(base_name)
                    # Handle dotted names
                    elif subchild.type in ('type', 'generic_type', 'identifier'):
                        base_name = self._get_node_text(subchild, source)
                        if base_name:
                            bases.append(base_name)

        return bases


# =============================================================================
# Auto-registration
# =============================================================================

def _register_generic_extractor() -> None:
    """
    Register the GenericExtractor as the fallback extractor.

    This function is called when the module is imported to ensure
    the generic extractor is always available.
    """
    from ..parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    registry.register_extractor("generic", GenericExtractor())
    logger.debug("GenericExtractor registered as fallback")


# Auto-register on module import
_register_generic_extractor()


__all__ = [
    "GenericExtractor",
    "FUNCTION_NODE_TYPES",
    "CLASS_NODE_TYPES",
    "NAME_NODE_TYPES",
]
