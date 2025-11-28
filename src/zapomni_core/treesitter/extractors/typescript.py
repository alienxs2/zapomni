"""
TypeScript/JavaScript-specific code extractor with full AST support.

This module provides a specialized extractor for TypeScript and JavaScript
source code that extracts functions, methods, classes, interfaces, type aliases,
and enums with rich metadata including:
- JSDoc comments
- Decorators (Angular, NestJS, etc.)
- Type annotations (generics, union, intersection types)
- Access modifiers (public, private, protected)
- Async functions and generators
- Arrow functions with proper name resolution

Supports both TypeScript (.ts, .tsx) and JavaScript (.js, .jsx) files.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import List, Optional, Set

import structlog
from tree_sitter import Node, Tree

from .base import BaseCodeExtractor
from ..models import ASTNodeLocation, CodeElementType, ExtractedCode, ParameterInfo

logger = structlog.get_logger(__name__)


# =============================================================================
# TypeScript-specific AST Node Types
# =============================================================================

# Function-like node types
TYPESCRIPT_FUNCTION_TYPES: Set[str] = {
    "function_declaration",
    "generator_function_declaration",
    "arrow_function",
    "method_definition",
}

# Class-like node types
TYPESCRIPT_CLASS_TYPES: Set[str] = {
    "class_declaration",
    "abstract_class_declaration",
}

# TypeScript-specific type definition node types
TYPESCRIPT_TYPE_TYPES: Set[str] = {
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
}

# Special decorators that affect method behavior
TYPESCRIPT_SPECIAL_DECORATORS: Set[str] = {
    "Injectable",
    "Component",
    "Directive",
    "Module",
    "Controller",
    "Service",
    "Get",
    "Post",
    "Put",
    "Delete",
    "Patch",
}


class TypeScriptExtractor(BaseCodeExtractor):
    """
    Specialized extractor for TypeScript and JavaScript source code.

    Provides accurate extraction of code elements with full support for
    TypeScript-specific features like interfaces, type aliases, generics,
    decorators, and access modifiers.

    Features:
        - JSDoc comments: /** ... */ style documentation
        - Decorators: @Component, @Injectable, custom decorators
        - Type annotations: Generics, union types, intersection types
        - Interfaces and type aliases: TypeScript-specific constructs
        - Enums: Regular and const enums
        - Access modifiers: public, private, protected
        - Async/generators: async functions, generator functions
        - Arrow functions: With proper name resolution from parent

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("typescript")
        >>> source = b'''
        ... /** A greeting function. */
        ... function greet(name: string): string {
        ...     return `Hello, ${name}!`;
        ... }
        ... '''
        >>> tree = parser.parse(source)
        >>> extractor = TypeScriptExtractor()
        >>> funcs = extractor.extract_functions(tree, source, "test.ts")
        >>> funcs[0].docstring
        'A greeting function.'
    """

    def __init__(self) -> None:
        """Initialize the TypeScript extractor with logging."""
        super().__init__()
        self._log = logger.bind(extractor="TypeScriptExtractor")
        self._log.debug("TypeScriptExtractor initialized")

    @property
    def language_name(self) -> str:
        """Return language name for this extractor."""
        return "typescript"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        """Return AST node types this extractor handles."""
        return (
            # Functions
            "function_declaration",
            "generator_function_declaration",
            "arrow_function",
            "method_definition",
            # Classes
            "class_declaration",
            "abstract_class_declaration",
            # TypeScript-specific
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            # Exports (contain decorated classes/functions)
            "export_statement",
        )

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
        Extract all function and method definitions from TypeScript/JavaScript AST.

        Handles regular functions, generator functions, arrow functions,
        and class methods with full metadata extraction.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all functions/methods.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node, parent_class: Optional[str] = None) -> None:
            """Recursively visit nodes to find functions."""
            node_id = id(node)

            # Handle export statements (may contain functions/classes)
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                if declaration:
                    if declaration.type in ("function_declaration", "generator_function_declaration"):
                        if id(declaration) not in visited_nodes:
                            visited_nodes.add(id(declaration))
                            extracted = self._extract_function_node(
                                declaration, source, file_path, parent_class, node
                            )
                            if extracted:
                                results.append(extracted)
                    elif declaration.type in ("class_declaration", "abstract_class_declaration"):
                        class_name = self._get_class_name(declaration, source)
                        if class_name:
                            self._visit_class_methods(
                                declaration, source, file_path, class_name,
                                results, visited_nodes, node
                            )
                    elif declaration.type == "lexical_declaration":
                        # Check for arrow functions in const/let
                        for child in declaration.children:
                            if child.type == "variable_declarator":
                                arrow = child.child_by_field_name("value")
                                if arrow and arrow.type == "arrow_function":
                                    if id(arrow) not in visited_nodes:
                                        visited_nodes.add(id(arrow))
                                        extracted = self._extract_arrow_function(
                                            arrow, child, source, file_path, parent_class, node
                                        )
                                        if extracted:
                                            results.append(extracted)
                return

            # Handle regular function declarations
            if node.type in ("function_declaration", "generator_function_declaration"):
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_function_node(
                        node, source, file_path, parent_class, None
                    )
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "function_extracted",
                            name=extracted.name,
                            file=file_path,
                        )

            # Handle arrow functions in variable declarations
            if node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        arrow = child.child_by_field_name("value")
                        if arrow and arrow.type == "arrow_function":
                            if id(arrow) not in visited_nodes:
                                visited_nodes.add(id(arrow))
                                extracted = self._extract_arrow_function(
                                    arrow, child, source, file_path, parent_class, None
                                )
                                if extracted:
                                    results.append(extracted)

            # Handle class definitions (to get methods)
            if node.type in ("class_declaration", "abstract_class_declaration"):
                class_name = self._get_class_name(node, source)
                if class_name:
                    self._visit_class_methods(
                        node, source, file_path, class_name,
                        results, visited_nodes, None
                    )
                return

            # Recurse into children
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
        export_node: Optional[Node],
    ) -> None:
        """Visit and extract methods from a class body."""
        body = class_node.child_by_field_name("body")
        if not body:
            return

        for child in body.children:
            if child.type == "method_definition":
                if id(child) not in visited_nodes:
                    visited_nodes.add(id(child))
                    extracted = self._extract_method_node(
                        child, source, file_path, class_name
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
        Extract all class definitions from TypeScript/JavaScript AST.

        Extracts both regular and abstract classes with their metadata
        including JSDoc, decorators, base classes, and implemented interfaces.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all classes.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find classes."""
            # Handle export statements
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                if declaration and declaration.type in ("class_declaration", "abstract_class_declaration"):
                    if id(declaration) not in visited_nodes:
                        visited_nodes.add(id(declaration))
                        extracted = self._extract_class_node(
                            declaration, source, file_path, node
                        )
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
            if node.type in ("class_declaration", "abstract_class_declaration"):
                if id(node) not in visited_nodes:
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

    def extract_interfaces(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all interface definitions from TypeScript AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all interfaces.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find interfaces."""
            # Handle export statements
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                if declaration and declaration.type == "interface_declaration":
                    if id(declaration) not in visited_nodes:
                        visited_nodes.add(id(declaration))
                        extracted = self._extract_interface_node(
                            declaration, source, file_path
                        )
                        if extracted:
                            results.append(extracted)
                return

            # Handle regular interface definitions
            if node.type == "interface_declaration":
                if id(node) not in visited_nodes:
                    visited_nodes.add(id(node))
                    extracted = self._extract_interface_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "interface_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def extract_types(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all type alias definitions from TypeScript AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all type aliases.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find type aliases."""
            # Handle export statements
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                if declaration and declaration.type == "type_alias_declaration":
                    if id(declaration) not in visited_nodes:
                        visited_nodes.add(id(declaration))
                        extracted = self._extract_type_alias_node(
                            declaration, source, file_path
                        )
                        if extracted:
                            results.append(extracted)
                return

            # Handle regular type alias definitions
            if node.type == "type_alias_declaration":
                if id(node) not in visited_nodes:
                    visited_nodes.add(id(node))
                    extracted = self._extract_type_alias_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "type_alias_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def extract_enums(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all enum definitions from TypeScript AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all enums.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find enums."""
            # Handle export statements
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                if declaration and declaration.type == "enum_declaration":
                    if id(declaration) not in visited_nodes:
                        visited_nodes.add(id(declaration))
                        extracted = self._extract_enum_node(
                            declaration, source, file_path
                        )
                        if extracted:
                            results.append(extracted)
                return

            # Handle regular enum definitions
            if node.type == "enum_declaration":
                if id(node) not in visited_nodes:
                    visited_nodes.add(id(node))
                    extracted = self._extract_enum_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "enum_extraction_complete",
            count=len(results),
            file=file_path,
        )
        return results

    def extract_all(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all code elements from TypeScript/JavaScript AST.

        Combines functions, classes, interfaces, type aliases, and enums,
        sorted by their starting line number.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of all ExtractedCode objects sorted by start_line.
        """
        self._log.debug(
            "extracting_all_elements",
            language=self.language_name,
            file_path=file_path,
        )

        functions = self.extract_functions(tree, source, file_path)
        classes = self.extract_classes(tree, source, file_path)
        interfaces = self.extract_interfaces(tree, source, file_path)
        types = self.extract_types(tree, source, file_path)
        enums = self.extract_enums(tree, source, file_path)

        all_elements = functions + classes + interfaces + types + enums
        all_elements.sort(key=lambda x: x.location.start_line)

        self._log.debug(
            "extraction_complete",
            language=self.language_name,
            file_path=file_path,
            functions_count=len(functions),
            classes_count=len(classes),
            interfaces_count=len(interfaces),
            types_count=len(types),
            enums_count=len(enums),
            total_count=len(all_elements),
        )

        return all_elements

    # =========================================================================
    # Node Extraction Methods
    # =========================================================================

    def _extract_function_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
        parent_class: Optional[str],
        export_node: Optional[Node],
    ) -> Optional[ExtractedCode]:
        """
        Extract a function from its AST node.

        Args:
            node: The function_declaration or generator_function_declaration node.
            source: Original source code.
            file_path: Path to the file.
            parent_class: Name of parent class if this is a method.
            export_node: The export_statement node if function is exported.

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

        # Determine the full node for source code extraction
        full_node = export_node if export_node else node

        # Get source code
        source_code = self._get_node_text(full_node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(full_node)

        # Extract JSDoc
        jsdoc_node = export_node if export_node else node
        docstring = self._extract_jsdoc(jsdoc_node, source)

        # Extract decorators from export node if present
        decorators = self._extract_decorators_from_siblings(
            export_node if export_node else node, source
        )

        # Check for async
        is_async = self._is_async(node)

        # Check for generator
        is_generator = node.type == "generator_function_declaration"

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Determine element type
        element_type = CodeElementType.METHOD if parent_class else CodeElementType.FUNCTION

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
            docstring=docstring,
            parent_class=parent_class,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
            is_static=False,
            is_abstract=False,
            is_private=self._is_private_name(name),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_arrow_function(
        self,
        arrow_node: Node,
        declarator_node: Node,
        source: bytes,
        file_path: str,
        parent_class: Optional[str],
        export_node: Optional[Node],
    ) -> Optional[ExtractedCode]:
        """
        Extract an arrow function from its AST node.

        Arrow functions get their name from the parent variable_declarator.

        Args:
            arrow_node: The arrow_function node.
            declarator_node: The variable_declarator containing the arrow function.
            source: Original source code.
            file_path: Path to the file.
            parent_class: Name of parent class if this is a method.
            export_node: The export_statement node if function is exported.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get function name from variable declarator
        name_node = declarator_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Find the lexical_declaration parent for full source
        parent = declarator_node.parent
        full_node = export_node if export_node else (parent if parent else declarator_node)

        # Get source code
        source_code = self._get_node_text(full_node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(full_node)

        # Extract JSDoc
        jsdoc_node = export_node if export_node else full_node
        docstring = self._extract_jsdoc(jsdoc_node, source)

        # Check for async
        is_async = self._is_async(arrow_node)

        # Extract parameters
        parameters = self._extract_parameters(arrow_node, source)

        # Extract return type
        return_type = self._extract_return_type(arrow_node, source)

        # Determine element type
        element_type = CodeElementType.METHOD if parent_class else CodeElementType.FUNCTION

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
            docstring=docstring,
            parent_class=parent_class,
            parameters=parameters,
            return_type=return_type,
            decorators=[],
            is_async=is_async,
            is_generator=False,  # Arrow functions can't be generators
            is_static=False,
            is_abstract=False,
            is_private=self._is_private_name(name),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_method_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
        parent_class: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a method from its AST node.

        Args:
            node: The method_definition node.
            source: Original source code.
            file_path: Path to the file.
            parent_class: Name of parent class.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get method name
        name_node = node.child_by_field_name("name")
        if not name_node:
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

        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source)

        # Extract decorators
        decorators = self._extract_method_decorators(node, source)

        # Check for async
        is_async = self._is_async(node)

        # Check for generator (method with *)
        is_generator = self._has_generator_marker(node)

        # Check for static
        is_static = self._has_static_modifier(node)

        # Check for abstract
        is_abstract = self._has_abstract_modifier(node)

        # Check access modifier for private
        access = self._get_access_modifier(node)
        is_private = access == "private" or self._is_private_name(name)

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Check for getter/setter
        is_getter = self._is_getter(node)
        is_setter = self._is_setter(node)

        # Add getter/setter to decorators for consistency
        if is_getter:
            decorators = ["get"] + decorators
        if is_setter:
            decorators = ["set"] + decorators

        return ExtractedCode(
            name=name,
            qualified_name=f"{parent_class}.{name}",
            element_type=CodeElementType.METHOD,
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
        export_node: Optional[Node],
    ) -> Optional[ExtractedCode]:
        """
        Extract a class from its AST node.

        Args:
            node: The class_declaration or abstract_class_declaration node.
            source: Original source code.
            file_path: Path to the file.
            export_node: The export_statement node if class is exported.

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

        # Determine full node
        full_node = export_node if export_node else node

        # Get source code
        source_code = self._get_node_text(full_node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(full_node)

        # Extract JSDoc
        jsdoc_node = export_node if export_node else node
        docstring = self._extract_jsdoc(jsdoc_node, source)

        # Extract decorators
        decorators = self._extract_decorators_from_siblings(
            export_node if export_node else node, source
        )

        # Check if abstract
        is_abstract = node.type == "abstract_class_declaration"

        # Extract base classes and implemented interfaces
        bases = self._extract_class_heritage(node, source)

        # Extract method names
        methods = self._extract_method_names(node, source)

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
            is_private=self._is_private_name(name),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_interface_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract an interface from its AST node.

        Args:
            node: The interface_declaration node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get interface name
        name_node = node.child_by_field_name("name")
        if not name_node:
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

        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source)

        # Extract extended interfaces
        bases = self._extract_interface_extends(node, source)

        # Extract method signatures
        methods = self._extract_interface_methods(node, source)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.INTERFACE,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=methods,
            bases=bases,
            decorators=[],
            is_abstract=True,  # Interfaces are inherently abstract
            is_private=self._is_private_name(name),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_type_alias_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a type alias from its AST node.

        Args:
            node: The type_alias_declaration node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get type alias name
        name_node = node.child_by_field_name("name")
        if not name_node:
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

        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.TYPE_ALIAS,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            decorators=[],
            is_private=self._is_private_name(name),
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_enum_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract an enum from its AST node.

        Args:
            node: The enum_declaration node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get enum name
        name_node = node.child_by_field_name("name")
        if not name_node:
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

        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source)

        # Extract enum members as "methods"
        members = self._extract_enum_members(node, source)

        # Check if const enum
        is_const = self._is_const_enum(node)
        decorators = ["const"] if is_const else []

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.ENUM,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=members,  # Store enum members in methods field
            decorators=decorators,
            is_private=self._is_private_name(name),
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

    def _extract_jsdoc(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract JSDoc comment from preceding sibling.

        JSDoc comments appear as prev_sibling nodes with type 'comment'
        and content starting with '/**'.

        Args:
            node: The node to find JSDoc for.
            source: Original source code.

        Returns:
            JSDoc content without comment markers, or None if not found.
        """
        prev = node.prev_sibling

        # Skip whitespace/newline nodes
        while prev and prev.type in ("", "\n", "comment"):
            if prev.type == "comment":
                text = self._get_node_text(prev, source)
                if text.startswith("/**"):
                    return self._clean_jsdoc(text)
            prev = prev.prev_sibling

        return None

    def _clean_jsdoc(self, raw_jsdoc: str) -> str:
        """
        Clean JSDoc comment by removing markers and normalizing whitespace.

        Args:
            raw_jsdoc: Raw JSDoc with /** ... */ markers.

        Returns:
            Cleaned JSDoc content.
        """
        # Remove /** and */
        content = raw_jsdoc
        if content.startswith("/**"):
            content = content[3:]
        if content.endswith("*/"):
            content = content[:-2]

        # Split into lines and clean each
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            # Remove leading * and whitespace
            line = line.strip()
            if line.startswith("*"):
                line = line[1:].strip()
            cleaned_lines.append(line)

        # Join and remove empty leading/trailing lines
        result = "\n".join(cleaned_lines).strip()
        return result

    def _extract_decorators_from_siblings(
        self, node: Node, source: bytes
    ) -> List[str]:
        """
        Extract decorators from preceding siblings.

        In TypeScript, decorators appear as siblings before the declaration.

        Args:
            node: The decorated node.
            source: Original source code.

        Returns:
            List of decorator names.
        """
        decorators: List[str] = []
        prev = node.prev_sibling

        while prev:
            if prev.type == "decorator":
                dec_text = self._get_node_text(prev, source)
                dec_name = self._parse_decorator_name(dec_text)
                if dec_name:
                    decorators.insert(0, dec_name)  # Maintain order
            elif prev.type not in ("comment", "", "\n"):
                break
            prev = prev.prev_sibling

        return decorators

    def _extract_method_decorators(self, node: Node, source: bytes) -> List[str]:
        """
        Extract decorators for a method definition.

        Method decorators appear as preceding siblings in class body.

        Args:
            node: The method_definition node.
            source: Original source code.

        Returns:
            List of decorator names.
        """
        return self._extract_decorators_from_siblings(node, source)

    def _parse_decorator_name(self, decorator_text: str) -> Optional[str]:
        """
        Parse decorator name from decorator text.

        Args:
            decorator_text: Full decorator text like '@Decorator' or '@Dec(args)'.

        Returns:
            Decorator name without @ and arguments.
        """
        if not decorator_text.startswith("@"):
            return None

        # Remove @
        text = decorator_text[1:]

        # Remove arguments
        paren_idx = text.find("(")
        if paren_idx > 0:
            text = text[:paren_idx]

        return text.strip() if text else None

    def _extract_parameters(self, node: Node, source: bytes) -> List[ParameterInfo]:
        """
        Extract parameters from a function/method definition.

        Args:
            node: Function or method definition node.
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

    def _extract_single_parameter(
        self, node: Node, source: bytes
    ) -> Optional[ParameterInfo]:
        """
        Extract a single parameter from its AST node.

        Handles:
        - required_parameter: name: Type
        - optional_parameter: name?: Type
        - rest_parameter: ...name: Type[]

        Args:
            node: Parameter node.
            source: Original source code.

        Returns:
            ParameterInfo or None if not a parameter.
        """
        node_type = node.type

        # Skip punctuation
        if node_type in ("(", ")", ","):
            return None

        name: Optional[str] = None
        type_annotation: Optional[str] = None
        default_value: Optional[str] = None

        if node_type == "required_parameter":
            # Get name (may be identifier, pattern, or rest_pattern)
            for child in node.children:
                if child.type == "identifier":
                    name = self._get_node_text(child, source)
                elif child.type == "rest_pattern":
                    # Handle rest parameter: ...name
                    for rest_child in child.children:
                        if rest_child.type == "identifier":
                            name = "..." + self._get_node_text(rest_child, source)
                            break
                elif child.type == "type_annotation":
                    type_annotation = self._extract_type_from_annotation(child, source)

        elif node_type == "optional_parameter":
            for child in node.children:
                if child.type == "identifier":
                    name = self._get_node_text(child, source)
                elif child.type == "type_annotation":
                    type_annotation = self._extract_type_from_annotation(child, source)
            # Check for default value
            for child in node.children:
                if child.type not in ("identifier", "type_annotation", "?", ":"):
                    # This might be the default value
                    if child.prev_sibling and self._get_node_text(child.prev_sibling, source) == "=":
                        default_value = self._get_node_text(child, source)
                        break

        elif node_type == "rest_parameter":
            for child in node.children:
                if child.type == "identifier":
                    name = "..." + self._get_node_text(child, source)
                elif child.type == "type_annotation":
                    type_annotation = self._extract_type_from_annotation(child, source)

        elif node_type == "identifier":
            # Simple parameter without type
            name = self._get_node_text(node, source)

        if name:
            return ParameterInfo(
                name=name,
                type_annotation=type_annotation,
                default_value=default_value,
            )

        return None

    def _extract_type_from_annotation(
        self, type_annotation_node: Node, source: bytes
    ) -> Optional[str]:
        """
        Extract type string from a type_annotation node.

        Args:
            type_annotation_node: The type_annotation node.
            source: Original source code.

        Returns:
            Type as string.
        """
        # Type annotation contains ': Type', we want just 'Type'
        text = self._get_node_text(type_annotation_node, source)
        if text.startswith(":"):
            return text[1:].strip()
        return text

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
            return self._extract_type_from_annotation(return_type_node, source)
        return None

    def _extract_class_heritage(self, node: Node, source: bytes) -> List[str]:
        """
        Extract base classes and implemented interfaces.

        Args:
            node: Class definition node.
            source: Original source code.

        Returns:
            List of base class and interface names.
        """
        bases: List[str] = []

        # Look for class_heritage child
        for child in node.children:
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    if heritage_child.type == "extends_clause":
                        # Get the extended class
                        for ext_child in heritage_child.children:
                            if ext_child.type in ("identifier", "type_identifier"):
                                bases.append(self._get_node_text(ext_child, source))
                    elif heritage_child.type == "implements_clause":
                        # Get implemented interfaces
                        for impl_child in heritage_child.children:
                            if impl_child.type in ("identifier", "type_identifier"):
                                bases.append(self._get_node_text(impl_child, source))

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
            if child.type == "method_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    methods.append(self._get_node_text(name_node, source))

        return methods

    def _extract_interface_extends(self, node: Node, source: bytes) -> List[str]:
        """
        Extract extended interfaces from an interface declaration.

        Args:
            node: Interface declaration node.
            source: Original source code.

        Returns:
            List of extended interface names.
        """
        bases: List[str] = []

        for child in node.children:
            if child.type == "extends_type_clause":
                for ext_child in child.children:
                    if ext_child.type in ("identifier", "type_identifier"):
                        bases.append(self._get_node_text(ext_child, source))

        return bases

    def _extract_interface_methods(self, node: Node, source: bytes) -> List[str]:
        """
        Extract method signature names from an interface body.

        Args:
            node: Interface declaration node.
            source: Original source code.

        Returns:
            List of method names.
        """
        methods: List[str] = []

        body = node.child_by_field_name("body")
        if not body:
            return methods

        for child in body.children:
            if child.type == "method_signature":
                name_node = child.child_by_field_name("name")
                if name_node:
                    methods.append(self._get_node_text(name_node, source))
            elif child.type == "property_signature":
                # Property signatures are not methods, skip
                pass

        return methods

    def _extract_enum_members(self, node: Node, source: bytes) -> List[str]:
        """
        Extract member names from an enum body.

        Args:
            node: Enum declaration node.
            source: Original source code.

        Returns:
            List of enum member names.
        """
        members: List[str] = []

        body = node.child_by_field_name("body")
        if not body:
            return members

        for child in body.children:
            if child.type == "property_identifier":
                members.append(self._get_node_text(child, source))
            elif child.type == "enum_assignment":
                name_node = child.child_by_field_name("name")
                if name_node:
                    members.append(self._get_node_text(name_node, source))

        return members

    def _is_async(self, node: Node) -> bool:
        """
        Check if a function/method is async.

        Args:
            node: Function or method definition node.

        Returns:
            True if function is async.
        """
        for child in node.children:
            if child.type == "async":
                return True
        return False

    def _has_generator_marker(self, node: Node) -> bool:
        """
        Check if a method has a generator marker (*).

        Args:
            node: Method definition node.

        Returns:
            True if method is a generator.
        """
        for child in node.children:
            if child.type == "*":
                return True
        return False

    def _has_static_modifier(self, node: Node) -> bool:
        """
        Check if a method has static modifier.

        Args:
            node: Method definition node.

        Returns:
            True if method is static.
        """
        for child in node.children:
            if child.type == "static":
                return True
        return False

    def _has_abstract_modifier(self, node: Node) -> bool:
        """
        Check if a method has abstract modifier.

        Args:
            node: Method definition node.

        Returns:
            True if method is abstract.
        """
        for child in node.children:
            if child.type == "abstract":
                return True
        return False

    def _get_access_modifier(self, node: Node) -> Optional[str]:
        """
        Get access modifier (public/private/protected) from a node.

        Args:
            node: Node to check.

        Returns:
            'public', 'private', 'protected', or None.
        """
        for child in node.children:
            if child.type == "accessibility_modifier":
                # Check children for the actual keyword
                for subchild in child.children:
                    if subchild.type in ("private", "protected", "public"):
                        return subchild.type
                # Fallback: check the accessibility_modifier text
                modifier_text = self._get_node_text(child, node.text or b"")
                if "private" in modifier_text:
                    return "private"
                elif "protected" in modifier_text:
                    return "protected"
                elif "public" in modifier_text:
                    return "public"
        return None

    def _is_getter(self, node: Node) -> bool:
        """
        Check if method is a getter.

        Args:
            node: Method definition node.

        Returns:
            True if method is a getter.
        """
        for child in node.children:
            if child.type == "get":
                return True
        return False

    def _is_setter(self, node: Node) -> bool:
        """
        Check if method is a setter.

        Args:
            node: Method definition node.

        Returns:
            True if method is a setter.
        """
        for child in node.children:
            if child.type == "set":
                return True
        return False

    def _is_const_enum(self, node: Node) -> bool:
        """
        Check if enum is a const enum.

        Args:
            node: Enum declaration node.

        Returns:
            True if enum is const.
        """
        for child in node.children:
            if child.type == "const":
                return True
        return False


# =============================================================================
# Auto-registration
# =============================================================================

def _register_typescript_extractor() -> None:
    """
    Register the TypeScriptExtractor for TypeScript and JavaScript languages.

    This function is called when the module is imported to make
    the extractor available for TypeScript and JavaScript files.
    """
    from ..parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    extractor = TypeScriptExtractor()

    # Register for TypeScript
    registry.register_extractor("typescript", extractor)
    logger.debug("TypeScriptExtractor registered for 'typescript' language")

    # Also register for JavaScript (same AST structure)
    registry.register_extractor("javascript", extractor)
    logger.debug("TypeScriptExtractor registered for 'javascript' language")


# Auto-register on module import
_register_typescript_extractor()


__all__ = [
    "TypeScriptExtractor",
    "TYPESCRIPT_FUNCTION_TYPES",
    "TYPESCRIPT_CLASS_TYPES",
    "TYPESCRIPT_TYPE_TYPES",
    "TYPESCRIPT_SPECIAL_DECORATORS",
]
