"""
Python-specific code extractor with full AST support.

This module provides a specialized extractor for Python source code that
extracts functions, methods, and classes with rich metadata including:
- Docstrings (Google, NumPy, Sphinx styles)
- Decorators (@staticmethod, @classmethod, @property, custom)
- Type hints (parameters and return types)
- Async functions and generators

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import List, Optional, Set

import structlog
from tree_sitter import Node, Tree

from ..models import CodeElementType, ExtractedCode, ParameterInfo
from .base import BaseCodeExtractor

logger = structlog.get_logger(__name__)


# =============================================================================
# Python-specific AST Node Types
# =============================================================================

# Decorators that affect function/method behavior
SPECIAL_DECORATORS: Set[str] = {
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "property",
    "cached_property",
}


class PythonExtractor(BaseCodeExtractor):
    """
    Specialized extractor for Python source code.

    Provides accurate extraction of Python code elements with full
    support for Python-specific features like decorators, type hints,
    docstrings, async functions, and generators.

    Features:
        - Docstrings: Google, NumPy, Sphinx styles
        - Decorators: @staticmethod, @classmethod, @property, custom
        - Type hints: Parameters, return types, complex types
        - Async/generators: async def, yield, yield from
        - Private detection: _private, __dunder__

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("python")
        >>> source = b'''
        ... @decorator
        ... def func(x: int) -> str:
        ...     \"\"\"Docstring.\"\"\"
        ...     return str(x)
        ... '''
        >>> tree = parser.parse(source)
        >>> extractor = PythonExtractor()
        >>> funcs = extractor.extract_functions(tree, source, "test.py")
        >>> funcs[0].decorators
        ['decorator']
        >>> funcs[0].docstring
        'Docstring.'
    """

    def __init__(self) -> None:
        """Initialize the Python extractor with logging."""
        super().__init__()
        self._log = logger.bind(extractor="PythonExtractor")
        self._log.debug("PythonExtractor initialized")

    @property
    def language_name(self) -> str:
        """Return language name for this extractor."""
        return "python"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        """Return AST node types this extractor handles."""
        return ("function_definition", "class_definition", "decorated_definition")

    # =========================================================================
    # Main Extraction Methods
    # =========================================================================

    def extract_functions(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all function and method definitions from Python AST.

        Handles both regular functions and decorated functions, extracting
        full metadata including docstrings, decorators, type hints, and flags.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Python source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all functions/methods.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()  # Track by node id to avoid duplicates

        def visit(node: Node, parent_class: Optional[str] = None) -> None:
            """Recursively visit nodes to find functions."""
            node_id = id(node)

            # Handle decorated definitions
            if node.type == "decorated_definition":
                # Get the actual function/class inside
                for child in node.children:
                    if child.type == "function_definition":
                        if id(child) not in visited_nodes:
                            visited_nodes.add(id(child))
                            extracted = self._extract_function_node(
                                child, source, file_path, parent_class, node
                            )
                            if extracted:
                                results.append(extracted)
                                self._log.debug(
                                    "function_extracted",
                                    name=extracted.name,
                                    decorators=extracted.decorators,
                                    file=file_path,
                                )
                    elif child.type == "class_definition":
                        # Process decorated class's methods
                        class_name = self._get_class_name(child, source)
                        if class_name:
                            self._visit_class_methods(
                                child, source, file_path, class_name, results, visited_nodes
                            )
                return  # Don't recurse into decorated_definition children normally

            # Handle regular function definitions
            if node.type == "function_definition" and node_id not in visited_nodes:
                visited_nodes.add(node_id)
                extracted = self._extract_function_node(node, source, file_path, parent_class, None)
                if extracted:
                    results.append(extracted)
                    self._log.debug(
                        "function_extracted",
                        name=extracted.name,
                        file=file_path,
                    )

            # Handle class definitions (to get methods)
            if node.type == "class_definition":
                class_name = self._get_class_name(node, source)
                if class_name:
                    self._visit_class_methods(
                        node, source, file_path, class_name, results, visited_nodes
                    )
                return  # Don't recurse deeper into classes

            # Recurse into children (but not into classes)
            for child in node.children:
                visit(child, parent_class)

        visit(tree.root_node)

        self._log.debug(
            "function_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def _visit_class_methods(
        self,
        class_node: Node,
        source: bytes,
        file_path: str,
        class_name: str,
        results: List[ExtractedCode],
        visited_nodes: Set[int],
    ) -> None:
        """Visit and extract methods from a class body."""
        body = class_node.child_by_field_name("body")
        if not body:
            return

        for child in body.children:
            if child.type == "decorated_definition":
                for subchild in child.children:
                    if subchild.type == "function_definition":
                        if id(subchild) not in visited_nodes:
                            visited_nodes.add(id(subchild))
                            extracted = self._extract_function_node(
                                subchild, source, file_path, class_name, child
                            )
                            if extracted:
                                results.append(extracted)
            elif child.type == "function_definition":
                if id(child) not in visited_nodes:
                    visited_nodes.add(id(child))
                    extracted = self._extract_function_node(
                        child, source, file_path, class_name, None
                    )
                    if extracted:
                        results.append(extracted)

    def extract_classes(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all class definitions from Python AST.

        Extracts classes with their docstrings, base classes, decorators,
        and method names.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Python source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all classes.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find classes."""
            # Handle decorated class definitions
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "class_definition":
                        if id(child) not in visited_nodes:
                            visited_nodes.add(id(child))
                            extracted = self._extract_class_node(child, source, file_path, node)
                            if extracted:
                                results.append(extracted)
                                self._log.debug(
                                    "class_extracted",
                                    name=extracted.name,
                                    decorators=extracted.decorators,
                                    file=file_path,
                                )
                return

            # Handle regular class definitions
            if node.type == "class_definition" and id(node) not in visited_nodes:
                visited_nodes.add(id(node))
                extracted = self._extract_class_node(node, source, file_path, None)
                if extracted:
                    results.append(extracted)
                    self._log.debug(
                        "class_extracted",
                        name=extracted.name,
                        file=file_path,
                    )

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "class_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    # =========================================================================
    # Node Extraction Methods
    # =========================================================================

    def _extract_function_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
        parent_class: Optional[str],
        decorated_node: Optional[Node],
    ) -> Optional[ExtractedCode]:
        """
        Extract a function/method from its AST node.

        Args:
            node: The function_definition node.
            source: Original source code.
            file_path: Path to the file.
            parent_class: Name of parent class if this is a method.
            decorated_node: The decorated_definition node if function is decorated.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get function name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Determine the full node (with decorators if present)
        full_node = decorated_node if decorated_node else node

        # Get source code
        source_code = self._get_node_text(full_node, source)
        if not source_code:
            return None

        # Create location from full node
        location = self._create_location(full_node)

        # Extract decorators
        decorators = self._extract_decorators(decorated_node, source) if decorated_node else []

        # Determine element type (function or method)
        element_type = CodeElementType.METHOD if parent_class else CodeElementType.FUNCTION

        # Build qualified name
        qualified_name = f"{parent_class}.{name}" if parent_class else name

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Check for async
        is_async = self._is_async(node)

        # Check for generator
        is_generator = self._is_generator(node)

        # Check for special decorators
        is_static = "staticmethod" in decorators
        is_abstract = "abstractmethod" in decorators

        # Check if private
        is_private = self._is_private_name(name)

        return ExtractedCode(
            name=name,
            qualified_name=qualified_name,
            element_type=element_type,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            parent_class=parent_class,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
            is_static=is_static,
            is_abstract=is_abstract,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_class_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
        decorated_node: Optional[Node],
    ) -> Optional[ExtractedCode]:
        """
        Extract a class from its AST node.

        Args:
            node: The class_definition node.
            source: Original source code.
            file_path: Path to the file.
            decorated_node: The decorated_definition node if class is decorated.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get class name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Determine the full node (with decorators if present)
        full_node = decorated_node if decorated_node else node

        # Get source code
        source_code = self._get_node_text(full_node, source)
        if not source_code:
            return None

        # Create location from full node
        location = self._create_location(full_node)

        # Extract decorators
        decorators = self._extract_decorators(decorated_node, source) if decorated_node else []

        # Extract base classes
        bases = self._extract_base_classes(node, source)

        # Extract method names
        methods = self._extract_method_names(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Check if private
        is_private = self._is_private_name(name)

        # Check if abstract (has ABC base or abstractmethod decorator)
        is_abstract = (
            "ABC" in bases or "ABCMeta" in bases or any("abstract" in d.lower() for d in decorators)
        )

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.CLASS,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=methods,
            bases=bases,
            decorators=decorators,
            is_abstract=is_abstract,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_class_name(self, class_node: Node, source: bytes) -> Optional[str]:
        """Get the name of a class from its AST node."""
        name_node = class_node.child_by_field_name("name")
        if name_node:
            return self._get_node_text(name_node, source)
        return None

    def _extract_docstring(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract docstring from a function or class body.

        The docstring is the first string literal in the body.
        In tree-sitter Python, this is a direct 'string' child of 'block'.

        Args:
            node: Function or class definition node.
            source: Original source code.

        Returns:
            Docstring content without quotes, or None if not found.
        """
        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None

        # In tree-sitter Python, docstring is the first 'string' node in block
        # (not wrapped in expression_statement)
        for child in body.children:
            if child.type == "string":
                docstring = self._get_node_text(child, source)
                return self._clean_docstring(docstring)
            elif child.type == "expression_statement":
                # Some tree-sitter versions may wrap it
                for subchild in child.children:
                    if subchild.type == "string":
                        docstring = self._get_node_text(subchild, source)
                        return self._clean_docstring(docstring)
                # If expression_statement doesn't contain string, stop looking
                break
            elif child.type not in ("comment", "newline", "NEWLINE", "INDENT", "DEDENT"):
                # First non-trivial node is not a string - no docstring
                break

        return None

    def _clean_docstring(self, raw_docstring: str) -> str:
        """
        Clean docstring by removing quotes and normalizing whitespace.

        Args:
            raw_docstring: Raw docstring with quotes.

        Returns:
            Cleaned docstring content.
        """
        # Remove triple quotes
        if raw_docstring.startswith('"""') and raw_docstring.endswith('"""'):
            return raw_docstring[3:-3].strip()
        if raw_docstring.startswith("'''") and raw_docstring.endswith("'''"):
            return raw_docstring[3:-3].strip()
        # Remove single quotes
        if raw_docstring.startswith('"') and raw_docstring.endswith('"'):
            return raw_docstring[1:-1].strip()
        if raw_docstring.startswith("'") and raw_docstring.endswith("'"):
            return raw_docstring[1:-1].strip()
        return raw_docstring.strip()

    def _extract_decorators(self, decorated_node: Optional[Node], source: bytes) -> List[str]:
        """
        Extract decorator names from a decorated definition.

        Args:
            decorated_node: The decorated_definition node.
            source: Original source code.

        Returns:
            List of decorator names (without @).
        """
        if not decorated_node:
            return []

        decorators: List[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                # Get decorator text without @
                dec_text = self._get_node_text(child, source)
                if dec_text.startswith("@"):
                    dec_text = dec_text[1:]

                # Extract just the name (before parentheses)
                # e.g., "property" from "@property" or "decorator(arg)" from "@decorator(arg)"
                paren_idx = dec_text.find("(")
                if paren_idx > 0:
                    dec_name = dec_text[:paren_idx].strip()
                else:
                    dec_name = dec_text.strip()

                # Handle dotted decorators like functools.wraps
                # Keep the full path for now
                if dec_name:
                    decorators.append(dec_name)

        return decorators

    def _extract_parameters(self, node: Node, source: bytes) -> List[ParameterInfo]:
        """
        Extract parameters from a function definition.

        Args:
            node: Function definition node.
            source: Original source code.

        Returns:
            List of ParameterInfo objects.
        """
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return []

        parameters: List[ParameterInfo] = []

        for child in params_node.children:
            param_info = self._extract_single_parameter(child, source)
            if param_info:
                parameters.append(param_info)

        return parameters

    def _extract_single_parameter(self, node: Node, source: bytes) -> Optional[ParameterInfo]:
        """
        Extract a single parameter from its AST node.

        Handles various parameter types:
        - identifier: simple parameter
        - typed_parameter: parameter with type annotation
        - default_parameter: parameter with default value
        - typed_default_parameter: parameter with type and default
        - list_splat_pattern / dictionary_splat_pattern: *args, **kwargs

        Args:
            node: Parameter node.
            source: Original source code.

        Returns:
            ParameterInfo or None if not a parameter.
        """
        node_type = node.type

        # Skip punctuation
        if node_type in ("(", ")", ",", "/"):
            return None

        name: Optional[str] = None
        type_annotation: Optional[str] = None
        default_value: Optional[str] = None

        if node_type == "identifier":
            # Simple parameter: x
            name = self._get_node_text(node, source)

        elif node_type == "typed_parameter":
            # Parameter with type: x: int
            for child in node.children:
                if child.type == "identifier":
                    name = self._get_node_text(child, source)
                elif child.type == "type":
                    type_annotation = self._get_node_text(child, source)

        elif node_type == "default_parameter":
            # Parameter with default: x=5
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if name_node:
                name = self._get_node_text(name_node, source)
            if value_node:
                default_value = self._get_node_text(value_node, source)

        elif node_type == "typed_default_parameter":
            # Parameter with type and default: x: int = 5
            name_node = node.child_by_field_name("name")
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")
            if name_node:
                name = self._get_node_text(name_node, source)
            if type_node:
                type_annotation = self._get_node_text(type_node, source)
            if value_node:
                default_value = self._get_node_text(value_node, source)

        elif node_type == "list_splat_pattern":
            # *args
            for child in node.children:
                if child.type == "identifier":
                    name = "*" + self._get_node_text(child, source)
                    break

        elif node_type == "dictionary_splat_pattern":
            # **kwargs
            for child in node.children:
                if child.type == "identifier":
                    name = "**" + self._get_node_text(child, source)
                    break

        if name:
            return ParameterInfo(
                name=name,
                type_annotation=type_annotation,
                default_value=default_value,
            )

        return None

    def _extract_return_type(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract return type annotation from a function.

        Args:
            node: Function definition node.
            source: Original source code.

        Returns:
            Return type as string, or None if not annotated.
        """
        return_type_node = node.child_by_field_name("return_type")
        if return_type_node:
            return self._get_node_text(return_type_node, source)
        return None

    def _extract_base_classes(self, node: Node, source: bytes) -> List[str]:
        """
        Extract base class names from a class definition.

        Args:
            node: Class definition node.
            source: Original source code.

        Returns:
            List of base class names.
        """
        bases: List[str] = []

        # Look for superclasses (argument_list after class name)
        superclasses = node.child_by_field_name("superclasses")
        if superclasses:
            for child in superclasses.children:
                if child.type in ("identifier", "attribute"):
                    base_name = self._get_node_text(child, source)
                    if base_name and base_name not in ("(", ")", ","):
                        bases.append(base_name)

        return bases

    def _extract_method_names(self, node: Node, source: bytes) -> List[str]:
        """
        Extract method names from a class body.

        Args:
            node: Class definition node.
            source: Original source code.

        Returns:
            List of method names.
        """
        methods: List[str] = []

        body = node.child_by_field_name("body")
        if not body:
            return methods

        for child in body.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    methods.append(self._get_node_text(name_node, source))
            elif child.type == "decorated_definition":
                for subchild in child.children:
                    if subchild.type == "function_definition":
                        name_node = subchild.child_by_field_name("name")
                        if name_node:
                            methods.append(self._get_node_text(name_node, source))

        return methods

    def _is_async(self, node: Node) -> bool:
        """
        Check if a function is async.

        Args:
            node: Function definition node.

        Returns:
            True if function is async.
        """
        # Check for 'async' keyword before 'def'
        if node.parent:
            prev_sibling = node.prev_sibling
            if prev_sibling and prev_sibling.type == "async":
                return True

        # Check first child for 'async' keyword
        for child in node.children:
            if child.type == "async":
                return True
            # In tree-sitter-python, async is not always a separate node
            # Check if 'async' appears in the function definition text before 'def'
            break

        # Fallback: check source text
        try:
            text = node.text.decode("utf-8", errors="ignore") if node.text else ""
            return text.strip().startswith("async ")
        except Exception:
            return False

    def _is_generator(self, node: Node) -> bool:
        """
        Check if a function is a generator (contains yield).

        Args:
            node: Function definition node.

        Returns:
            True if function contains yield/yield from.
        """
        body = node.child_by_field_name("body")
        if not body:
            return False

        def has_yield(n: Node) -> bool:
            """Recursively search for yield statements."""
            if n.type in ("yield", "yield_expression"):
                return True
            # Don't search into nested functions
            if n.type == "function_definition":
                return False
            for child in n.children:
                if has_yield(child):
                    return True
            return False

        return has_yield(body)


# =============================================================================
# Auto-registration
# =============================================================================


def _register_python_extractor() -> None:
    """
    Register the PythonExtractor for Python language.

    This function is called when the module is imported to make
    the extractor available for Python files.
    """
    from ..parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    registry.register_extractor("python", PythonExtractor())
    logger.debug("PythonExtractor registered for 'python' language")


# Auto-register on module import
_register_python_extractor()


__all__ = [
    "PythonExtractor",
    "SPECIAL_DECORATORS",
]
