"""
Go-specific code extractor with full AST support.

This module provides a specialized extractor for Go source code that
extracts functions, methods, structs, and interfaces with rich metadata including:
- Doc comments (// style, before declarations)
- Methods with receivers (pointer and value)
- Multiple return values
- Named returns
- Generics (type parameters) - Go 1.18+
- Embedded types in structs

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import List, Optional, Set

import structlog
from tree_sitter import Node, Tree

from ..models import ASTNodeLocation, CodeElementType, ExtractedCode, ParameterInfo
from .base import BaseCodeExtractor

logger = structlog.get_logger(__name__)


# =============================================================================
# Go-specific AST Node Types
# =============================================================================

# Function-like node types
GO_FUNCTION_TYPES: Set[str] = {
    "function_declaration",
    "method_declaration",
}

# Type definition node types
GO_TYPE_TYPES: Set[str] = {
    "type_declaration",
    "type_spec",
}

# Struct and interface types
GO_STRUCT_TYPES: Set[str] = {
    "struct_type",
}

GO_INTERFACE_TYPES: Set[str] = {
    "interface_type",
}


class GoExtractor(BaseCodeExtractor):
    """
    Specialized extractor for Go source code.

    Provides accurate extraction of Go code elements with full support for
    Go-specific features like receivers, multiple returns, interfaces,
    structs, and generics.

    Features:
        - Doc comments: // style comments before declarations
        - Methods with receivers: value and pointer receivers
        - Multiple return values: func foo() (int, error)
        - Named returns: func foo() (result int, err error)
        - Generics: type parameters on functions and types (Go 1.18+)
        - Embedded types: anonymous fields in structs
        - Private detection: lowercase first letter = private

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("go")
        >>> source = b'''
        ... // Add adds two integers.
        ... func Add(a, b int) int {
        ...     return a + b
        ... }
        ... '''
        >>> tree = parser.parse(source)
        >>> extractor = GoExtractor()
        >>> funcs = extractor.extract_functions(tree, source, "test.go")
        >>> funcs[0].docstring
        'Add adds two integers.'
    """

    def __init__(self) -> None:
        """Initialize the Go extractor with logging."""
        super().__init__()
        self._log = logger.bind(extractor="GoExtractor")
        self._log.debug("GoExtractor initialized")

    @property
    def language_name(self) -> str:
        """Return language name for this extractor."""
        return "go"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        """Return AST node types this extractor handles."""
        return ("function_declaration", "method_declaration", "type_declaration")

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
        Extract all function and method definitions from Go AST.

        Handles both regular functions and methods with receivers,
        extracting full metadata including doc comments, parameters,
        return types, and type parameters.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Go source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all functions/methods.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find functions and methods."""
            node_id = id(node)

            # Handle function declarations
            if node.type == "function_declaration":
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_function_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "function_extracted",
                            name=extracted.name,
                            file=file_path,
                        )

            # Handle method declarations
            elif node.type == "method_declaration":
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_method_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "method_extracted",
                            name=extracted.name,
                            receiver=extracted.parent_class,
                            file=file_path,
                        )

            # Recurse into children
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
        Extract all struct and interface definitions from Go AST.

        In Go, structs and interfaces are the closest equivalents to classes.
        This method extracts both types.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Go source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all structs and interfaces.
        """
        # Combine structs and interfaces as they serve class-like purposes in Go
        structs = self.extract_structs(tree, source, file_path)
        interfaces = self.extract_interfaces(tree, source, file_path)

        all_types = structs + interfaces
        all_types.sort(key=lambda x: x.location.start_line)

        return all_types

    def extract_structs(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all struct definitions from Go AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Go source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all structs.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find struct type specs."""
            # Look for type declarations containing struct types
            if node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        self._process_type_spec(
                            child, node, source, file_path, results, visited_nodes, "struct"
                        )

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "struct_extraction_complete",
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
        Extract all interface definitions from Go AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Go source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all interfaces.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find interface type specs."""
            # Look for type declarations containing interface types
            if node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        self._process_type_spec(
                            child, node, source, file_path, results, visited_nodes, "interface"
                        )

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

    def _process_type_spec(
        self,
        type_spec_node: Node,
        type_decl_node: Node,
        source: bytes,
        file_path: str,
        results: List[ExtractedCode],
        visited_nodes: Set[int],
        target_type: str,
    ) -> None:
        """
        Process a type_spec node and extract struct or interface if it matches target_type.

        Args:
            type_spec_node: The type_spec AST node.
            type_decl_node: The parent type_declaration node.
            source: Original source code.
            file_path: Path to the file.
            results: List to append results to.
            visited_nodes: Set of visited node IDs.
            target_type: Either "struct" or "interface".
        """
        node_id = id(type_spec_node)
        if node_id in visited_nodes:
            return

        # Get the name node
        name_node = type_spec_node.child_by_field_name("name")
        if not name_node:
            return

        # Find the type node (struct_type or interface_type)
        type_node = type_spec_node.child_by_field_name("type")
        if not type_node:
            return

        # Check if it's the target type
        if target_type == "struct" and type_node.type == "struct_type":
            visited_nodes.add(node_id)
            extracted = self._extract_struct_node(type_spec_node, type_decl_node, source, file_path)
            if extracted:
                results.append(extracted)
                self._log.debug(
                    "struct_extracted",
                    name=extracted.name,
                    file=file_path,
                )
        elif target_type == "interface" and type_node.type == "interface_type":
            visited_nodes.add(node_id)
            extracted = self._extract_interface_node(
                type_spec_node, type_decl_node, source, file_path
            )
            if extracted:
                results.append(extracted)
                self._log.debug(
                    "interface_extracted",
                    name=extracted.name,
                    file=file_path,
                )

    def extract_all(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all code elements from Go AST.

        Combines functions, methods, structs, and interfaces,
        sorted by their starting line number.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Go source code as bytes.
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
        structs = self.extract_structs(tree, source, file_path)
        interfaces = self.extract_interfaces(tree, source, file_path)

        all_elements = functions + structs + interfaces
        all_elements.sort(key=lambda x: x.location.start_line)

        self._log.debug(
            "extraction_complete",
            language=self.language_name,
            file_path=file_path,
            functions_count=len(functions),
            structs_count=len(structs),
            interfaces_count=len(interfaces),
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
    ) -> Optional[ExtractedCode]:
        """
        Extract a function from its AST node.

        Args:
            node: The function_declaration node.
            source: Original source code.
            file_path: Path to the file.

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

        # Get source code
        source_code = self._get_node_text(node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(node)

        # Extract doc comments
        docstring = self._extract_go_doc(node, source)

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract type parameters (generics)
        type_params = self._extract_type_parameters(node, source)
        decorators = [f"type_params:{type_params}"] if type_params else []

        # Check if private (lowercase first letter)
        is_private = self._is_go_private(name)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.FUNCTION,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            parent_class=None,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=False,  # Go doesn't have async keyword
            is_generator=False,
            is_static=False,
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_method_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a method from its AST node.

        Args:
            node: The method_declaration node.
            source: Original source code.
            file_path: Path to the file.

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

        # Get receiver type
        receiver_type, is_pointer_receiver = self._extract_receiver_type(node, source)

        # Get source code
        source_code = self._get_node_text(node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(node)

        # Extract doc comments
        docstring = self._extract_go_doc(node, source)

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Build qualified name
        qualified_name = f"{receiver_type}.{name}" if receiver_type else name

        # Build decorators (include pointer receiver info)
        decorators = []
        if is_pointer_receiver:
            decorators.append("pointer_receiver")

        # Extract type parameters (generics)
        type_params = self._extract_type_parameters(node, source)
        if type_params:
            decorators.append(f"type_params:{type_params}")

        # Check if private
        is_private = self._is_go_private(name)

        return ExtractedCode(
            name=name,
            qualified_name=qualified_name,
            element_type=CodeElementType.METHOD,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            parent_class=receiver_type,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=False,
            is_generator=False,
            is_static=False,
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_struct_node(
        self,
        type_spec_node: Node,
        type_decl_node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a struct from its AST node.

        Args:
            type_spec_node: The type_spec node containing the struct.
            type_decl_node: The parent type_declaration node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get struct name
        name_node = type_spec_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Get source code (from type_declaration for full context)
        source_code = self._get_node_text(type_decl_node, source)
        if not source_code:
            return None

        # Create location from type_declaration
        location = self._create_location(type_decl_node)

        # Extract doc comments
        docstring = self._extract_go_doc(type_decl_node, source)

        # Extract field names as "methods" for consistency
        fields = self._extract_struct_fields(type_spec_node, source)

        # Extract embedded types as "bases"
        embedded = self._extract_embedded_types(type_spec_node, source)

        # Extract type parameters (generics)
        type_params = self._extract_type_parameters(type_spec_node, source)
        decorators = [f"type_params:{type_params}"] if type_params else []

        # Check if private
        is_private = self._is_go_private(name)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.STRUCT,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=fields,  # Store field names in methods
            bases=embedded,  # Store embedded types in bases
            decorators=decorators,
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_interface_node(
        self,
        type_spec_node: Node,
        type_decl_node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract an interface from its AST node.

        Args:
            type_spec_node: The type_spec node containing the interface.
            type_decl_node: The parent type_declaration node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get interface name
        name_node = type_spec_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        if not name:
            return None

        # Get source code
        source_code = self._get_node_text(type_decl_node, source)
        if not source_code:
            return None

        # Create location
        location = self._create_location(type_decl_node)

        # Extract doc comments
        docstring = self._extract_go_doc(type_decl_node, source)

        # Extract method signatures
        methods = self._extract_interface_methods(type_spec_node, source)

        # Extract embedded interfaces as "bases"
        embedded = self._extract_embedded_interfaces(type_spec_node, source)

        # Extract type parameters (generics)
        type_params = self._extract_type_parameters(type_spec_node, source)
        decorators = [f"type_params:{type_params}"] if type_params else []

        # Check if private
        is_private = self._is_go_private(name)

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
            bases=embedded,
            decorators=decorators,
            is_abstract=True,  # Interfaces are inherently abstract
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_go_doc(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract Go doc comments from preceding siblings.

        Go doc comments are // style comments immediately before
        the declaration, with no blank lines between.

        Args:
            node: The node to find doc comments for.
            source: Original source code.

        Returns:
            Doc comment content without // markers, or None if not found.
        """
        comments: List[str] = []
        prev = node.prev_sibling

        # Collect all adjacent comment nodes
        while prev:
            if prev.type == "comment":
                comment_text = self._get_node_text(prev, source)
                comments.insert(0, comment_text)
                prev = prev.prev_sibling
            else:
                break

        if not comments:
            return None

        # Clean up comments
        cleaned_lines = []
        for comment in comments:
            if comment.startswith("//"):
                # Remove // and leading space
                line = comment[2:].lstrip()
                cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines).strip()

        return None

    def _is_go_private(self, name: str) -> bool:
        """
        Check if a name is private in Go (lowercase first letter).

        Args:
            name: The identifier name to check.

        Returns:
            True if the name starts with a lowercase letter.
        """
        if not name:
            return False
        return name[0].islower()

    def _extract_parameters(self, node: Node, source: bytes) -> List[ParameterInfo]:
        """
        Extract parameters from a function/method declaration.

        Handles Go's parameter_list syntax including grouped parameters
        (e.g., "a, b int" where both have the same type).

        Args:
            node: Function or method declaration node.
            source: Original source code.

        Returns:
            List of ParameterInfo objects.
        """
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return []

        parameters: List[ParameterInfo] = []

        for child in params_node.children:
            if child.type == "parameter_declaration":
                param_infos = self._extract_parameter_declaration(child, source)
                parameters.extend(param_infos)
            elif child.type == "variadic_parameter_declaration":
                param_info = self._extract_variadic_parameter(child, source)
                if param_info:
                    parameters.append(param_info)

        return parameters

    def _extract_parameter_declaration(self, node: Node, source: bytes) -> List[ParameterInfo]:
        """
        Extract parameters from a parameter_declaration node.

        Handles grouped parameters like "a, b int".

        Args:
            node: Parameter declaration node.
            source: Original source code.

        Returns:
            List of ParameterInfo objects.
        """
        parameters: List[ParameterInfo] = []
        names: List[str] = []
        type_annotation: Optional[str] = None

        for child in node.children:
            if child.type == "identifier":
                names.append(self._get_node_text(child, source))
            elif child.type not in (",", "(", ")"):
                # This is likely the type
                type_annotation = self._get_node_text(child, source)

        # Create ParameterInfo for each name
        for name in names:
            if name:
                parameters.append(
                    ParameterInfo(
                        name=name,
                        type_annotation=type_annotation,
                        default_value=None,
                    )
                )

        return parameters

    def _extract_variadic_parameter(self, node: Node, source: bytes) -> Optional[ParameterInfo]:
        """
        Extract a variadic parameter (...type).

        Args:
            node: Variadic parameter declaration node.
            source: Original source code.

        Returns:
            ParameterInfo or None.
        """
        name: Optional[str] = None
        type_annotation: Optional[str] = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source)
            elif child.type == "...":
                continue
            else:
                # This is likely the element type
                type_annotation = "..." + self._get_node_text(child, source)

        if name:
            return ParameterInfo(
                name=name,
                type_annotation=type_annotation,
                default_value=None,
            )

        return None

    def _extract_return_type(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract return type from a function/method declaration.

        Handles single returns, multiple returns, and named returns.

        Args:
            node: Function or method declaration node.
            source: Original source code.

        Returns:
            Return type as string, or None if no return type.
        """
        result_node = node.child_by_field_name("result")
        if not result_node:
            return None

        return self._get_node_text(result_node, source)

    def _extract_receiver_type(self, node: Node, source: bytes) -> tuple[Optional[str], bool]:
        """
        Extract the receiver type from a method declaration.

        Args:
            node: Method declaration node.
            source: Original source code.

        Returns:
            Tuple of (receiver_type, is_pointer_receiver).
        """
        receiver_node = node.child_by_field_name("receiver")
        if not receiver_node:
            return None, False

        receiver_type: Optional[str] = None
        is_pointer = False

        # Find the type inside the receiver parameter list
        for child in receiver_node.children:
            if child.type == "parameter_declaration":
                for subchild in child.children:
                    if subchild.type == "pointer_type":
                        is_pointer = True
                        # Get the type inside the pointer
                        for ptr_child in subchild.children:
                            if ptr_child.type == "type_identifier":
                                receiver_type = self._get_node_text(ptr_child, source)
                                break
                    elif subchild.type == "type_identifier":
                        receiver_type = self._get_node_text(subchild, source)

        return receiver_type, is_pointer

    def _extract_type_parameters(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract type parameters (generics) from a declaration.

        Args:
            node: Function, method, or type_spec node.
            source: Original source code.

        Returns:
            Type parameters as string, or None if no type parameters.
        """
        type_params_node = node.child_by_field_name("type_parameters")
        if type_params_node:
            return self._get_node_text(type_params_node, source)
        return None

    def _extract_struct_fields(self, type_spec_node: Node, source: bytes) -> List[str]:
        """
        Extract field names from a struct.

        Args:
            type_spec_node: The type_spec node containing the struct.
            source: Original source code.

        Returns:
            List of field names.
        """
        fields: List[str] = []

        type_node = type_spec_node.child_by_field_name("type")
        if not type_node or type_node.type != "struct_type":
            return fields

        # Find field_declaration_list
        for child in type_node.children:
            if child.type == "field_declaration_list":
                for field_child in child.children:
                    if field_child.type == "field_declaration":
                        # Extract field names
                        for fc in field_child.children:
                            if fc.type == "field_identifier":
                                fields.append(self._get_node_text(fc, source))

        return fields

    def _extract_embedded_types(self, type_spec_node: Node, source: bytes) -> List[str]:
        """
        Extract embedded (anonymous) types from a struct.

        Args:
            type_spec_node: The type_spec node containing the struct.
            source: Original source code.

        Returns:
            List of embedded type names.
        """
        embedded: List[str] = []

        type_node = type_spec_node.child_by_field_name("type")
        if not type_node or type_node.type != "struct_type":
            return embedded

        # Find field_declaration_list
        for child in type_node.children:
            if child.type == "field_declaration_list":
                for field_child in child.children:
                    if field_child.type == "field_declaration":
                        # Check if this is an embedded field (no field_identifier, just type)
                        has_field_name = False
                        type_name = None
                        for fc in field_child.children:
                            if fc.type == "field_identifier":
                                has_field_name = True
                            elif fc.type in ("type_identifier", "qualified_type", "pointer_type"):
                                type_name = self._get_node_text(fc, source)

                        if not has_field_name and type_name:
                            embedded.append(type_name)

        return embedded

    def _extract_interface_methods(self, type_spec_node: Node, source: bytes) -> List[str]:
        """
        Extract method signatures from an interface.

        Args:
            type_spec_node: The type_spec node containing the interface.
            source: Original source code.

        Returns:
            List of method names.
        """
        methods: List[str] = []

        type_node = type_spec_node.child_by_field_name("type")
        if not type_node or type_node.type != "interface_type":
            return methods

        for child in type_node.children:
            # Tree-sitter Go uses method_elem with field_identifier for method names
            if child.type == "method_elem":
                for subchild in child.children:
                    if subchild.type == "field_identifier":
                        methods.append(self._get_node_text(subchild, source))
                        break
            # Also support method_spec for compatibility
            elif child.type == "method_spec":
                name_node = child.child_by_field_name("name")
                if name_node:
                    methods.append(self._get_node_text(name_node, source))

        return methods

    def _extract_embedded_interfaces(self, type_spec_node: Node, source: bytes) -> List[str]:
        """
        Extract embedded interfaces.

        Args:
            type_spec_node: The type_spec node containing the interface.
            source: Original source code.

        Returns:
            List of embedded interface names.
        """
        embedded: List[str] = []

        type_node = type_spec_node.child_by_field_name("type")
        if not type_node or type_node.type != "interface_type":
            return embedded

        for child in type_node.children:
            # Embedded interfaces appear as type_elem containing type_identifier
            if child.type == "type_elem":
                for subchild in child.children:
                    if subchild.type in ("type_identifier", "qualified_type"):
                        embedded.append(self._get_node_text(subchild, source))
            # Also check direct type_identifier/qualified_type for compatibility
            elif child.type in ("type_identifier", "qualified_type"):
                embedded.append(self._get_node_text(child, source))

        return embedded


# =============================================================================
# Auto-registration
# =============================================================================


def _register_go_extractor() -> None:
    """
    Register the GoExtractor for Go language.

    This function is called when the module is imported to make
    the extractor available for Go files.
    """
    from ..parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    registry.register_extractor("go", GoExtractor())
    logger.debug("GoExtractor registered for 'go' language")


# Auto-register on module import
_register_go_extractor()


__all__ = [
    "GoExtractor",
    "GO_FUNCTION_TYPES",
    "GO_TYPE_TYPES",
    "GO_STRUCT_TYPES",
    "GO_INTERFACE_TYPES",
]
