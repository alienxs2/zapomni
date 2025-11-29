"""
Rust-specific code extractor with full AST support.

This module provides a specialized extractor for Rust source code that
extracts functions, methods, structs, traits, enums, and macros with rich metadata including:
- Doc comments (/// style and //! inner doc comments)
- Methods with receivers (self, &self, &mut self)
- impl blocks (inherent and trait implementations)
- Generics with type bounds and lifetimes
- Async functions
- Attributes (#[derive], #[cfg], etc.)
- Visibility modifiers (pub, pub(crate), pub(super))

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import List, Optional, Set, Tuple

import structlog
from tree_sitter import Node, Tree

from ..models import ASTNodeLocation, CodeElementType, ExtractedCode, ParameterInfo
from .base import BaseCodeExtractor

logger = structlog.get_logger(__name__)


# =============================================================================
# Rust-specific AST Node Types
# =============================================================================

# Function-like node types
RUST_FUNCTION_TYPES: Set[str] = {
    "function_item",
    "function_signature_item",
}

# Struct node types
RUST_STRUCT_TYPES: Set[str] = {
    "struct_item",
}

# Trait node types
RUST_TRAIT_TYPES: Set[str] = {
    "trait_item",
}

# Enum node types
RUST_ENUM_TYPES: Set[str] = {
    "enum_item",
}

# Macro node types
RUST_MACRO_TYPES: Set[str] = {
    "macro_definition",
    "macro_rules!",
}

# impl block node types
RUST_IMPL_TYPES: Set[str] = {
    "impl_item",
}


class RustExtractor(BaseCodeExtractor):
    """
    Specialized extractor for Rust source code.

    Provides accurate extraction of Rust code elements with full support for
    Rust-specific features like impl blocks, traits, generics, lifetimes,
    attributes, and visibility modifiers.

    Features:
        - Doc comments: /// and //! style comments
        - Methods with receivers: self, &self, &mut self
        - impl blocks: inherent and trait implementations
        - Generics: type parameters with bounds (<T: Display + Clone>)
        - Lifetimes: 'a, 'static, etc.
        - Async functions: async fn support
        - Attributes: #[derive], #[cfg], etc.
        - Visibility: pub, pub(crate), pub(super), private (no modifier)

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("rust")
        >>> source = b'''
        ... /// Adds two integers.
        ... pub fn add(a: i32, b: i32) -> i32 {
        ...     a + b
        ... }
        ... '''
        >>> tree = parser.parse(source)
        >>> extractor = RustExtractor()
        >>> funcs = extractor.extract_functions(tree, source, "test.rs")
        >>> funcs[0].docstring
        'Adds two integers.'
    """

    def __init__(self) -> None:
        """Initialize the Rust extractor with logging."""
        super().__init__()
        self._log = logger.bind(extractor="RustExtractor")
        self._log.debug("RustExtractor initialized")

    @property
    def language_name(self) -> str:
        """Return language name for this extractor."""
        return "rust"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        """Return AST node types this extractor handles."""
        return (
            "function_item",
            "impl_item",
            "struct_item",
            "trait_item",
            "enum_item",
            "macro_definition",
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
        Extract all function definitions from Rust AST.

        Handles both standalone functions and methods within impl blocks,
        extracting full metadata including doc comments, parameters,
        return types, generics, and async markers.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all functions/methods.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node, impl_context: Optional[Tuple[str, Optional[str]]] = None) -> None:
            """
            Recursively visit nodes to find functions and methods.

            Args:
                node: Current AST node.
                impl_context: Optional tuple of (type_name, trait_name) from parent impl block.
            """
            node_id = id(node)

            # Handle standalone function items
            if node.type == "function_item":
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_function_node(node, source, file_path, impl_context)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "function_extracted",
                            name=extracted.name,
                            file=file_path,
                        )

            # Handle impl blocks - extract methods
            elif node.type == "impl_item":
                impl_type, trait_name = self._extract_impl_context(node, source)
                if impl_type:
                    # Process children within impl context
                    for child in node.children:
                        if child.type == "declaration_list":
                            for item in child.children:
                                if item.type == "function_item":
                                    if id(item) not in visited_nodes:
                                        visited_nodes.add(id(item))
                                        extracted = self._extract_function_node(
                                            item, source, file_path, (impl_type, trait_name)
                                        )
                                        if extracted:
                                            results.append(extracted)
                                            self._log.debug(
                                                "method_extracted",
                                                name=extracted.name,
                                                receiver=extracted.parent_class,
                                                file=file_path,
                                            )

            # Recurse into children (but not into impl blocks which we handle specially)
            if node.type != "impl_item":
                for child in node.children:
                    visit(child, impl_context)

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
        Extract all struct, trait, and enum definitions from Rust AST.

        In Rust, structs, traits, and enums serve as class-like constructs.
        This method extracts all three types.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all structs, traits, and enums.
        """
        # Combine structs, traits, and enums as they serve class-like purposes in Rust
        structs = self.extract_structs(tree, source, file_path)
        traits = self.extract_traits(tree, source, file_path)
        enums = self.extract_enums(tree, source, file_path)

        all_types = structs + traits + enums
        all_types.sort(key=lambda x: x.location.start_line)

        return all_types

    def extract_structs(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all struct definitions from Rust AST.

        Handles regular structs, tuple structs, and unit structs.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all structs.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find struct items."""
            if node.type == "struct_item":
                node_id = id(node)
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_struct_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "struct_extracted",
                            name=extracted.name,
                            file=file_path,
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

    def extract_traits(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
    ) -> List[ExtractedCode]:
        """
        Extract all trait definitions from Rust AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all traits.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find trait items."""
            if node.type == "trait_item":
                node_id = id(node)
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_trait_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "trait_extracted",
                            name=extracted.name,
                            file=file_path,
                        )

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        self._log.debug(
            "trait_extraction_complete",
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
        Extract all enum definitions from Rust AST.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
            file_path: Path to the source file.

        Returns:
            List of ExtractedCode objects for all enums.
        """
        results: List[ExtractedCode] = []
        visited_nodes: Set[int] = set()

        def visit(node: Node) -> None:
            """Recursively visit nodes to find enum items."""
            if node.type == "enum_item":
                node_id = id(node)
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    extracted = self._extract_enum_node(node, source, file_path)
                    if extracted:
                        results.append(extracted)
                        self._log.debug(
                            "enum_extracted",
                            name=extracted.name,
                            file=file_path,
                        )

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
        Extract all code elements from Rust AST.

        Combines functions, methods, structs, traits, and enums,
        sorted by their starting line number.

        Args:
            tree: Parsed tree-sitter AST tree.
            source: Original Rust source code as bytes.
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
        traits = self.extract_traits(tree, source, file_path)
        enums = self.extract_enums(tree, source, file_path)

        all_elements = functions + structs + traits + enums
        all_elements.sort(key=lambda x: x.location.start_line)

        self._log.debug(
            "extraction_complete",
            language=self.language_name,
            file_path=file_path,
            functions_count=len(functions),
            structs_count=len(structs),
            traits_count=len(traits),
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
        impl_context: Optional[Tuple[str, Optional[str]]] = None,
    ) -> Optional[ExtractedCode]:
        """
        Extract a function or method from its AST node.

        Args:
            node: The function_item node.
            source: Original source code.
            file_path: Path to the file.
            impl_context: Optional tuple of (impl_type, trait_name) from parent impl block.

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
        docstring = self._extract_rust_doc(node, source)

        # Extract parameters
        parameters = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Check if async
        is_async = self._is_async_function(node, source)

        # Extract generics and lifetimes
        generics = self._extract_generics(node, source)
        lifetimes = self._extract_lifetimes(node, source)

        # Extract attributes
        attributes = self._extract_attributes(node, source)

        # Build decorators list
        decorators = attributes.copy()
        if generics:
            decorators.append(f"generics:{generics}")
        if lifetimes:
            decorators.append(f"lifetimes:{','.join(lifetimes)}")
        if is_async:
            decorators.append("async")

        # Determine if this is a method
        is_method = impl_context is not None
        parent_class = impl_context[0] if impl_context else None
        trait_name = impl_context[1] if impl_context and len(impl_context) > 1 else None

        # Detect receiver type (self, &self, &mut self)
        receiver_info = self._extract_receiver_info(node, source)
        if receiver_info:
            decorators.append(f"receiver:{receiver_info}")

        # Build qualified name
        if parent_class:
            if trait_name:
                qualified_name = f"{parent_class}::{trait_name}.{name}"
            else:
                qualified_name = f"{parent_class}.{name}"
        else:
            qualified_name = name

        # Determine element type
        element_type = CodeElementType.METHOD if is_method else CodeElementType.FUNCTION

        # Check visibility (private = no pub modifier)
        is_private = self._is_rust_private(node, source)

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
            is_generator=False,  # Rust doesn't have generators in the same sense
            is_static=receiver_info is None and is_method,  # Method without self is static
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_struct_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a struct from its AST node.

        Handles regular structs, tuple structs, and unit structs.

        Args:
            node: The struct_item node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get struct name
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
        docstring = self._extract_rust_doc(node, source)

        # Extract field names
        fields = self._extract_struct_fields(node, source)

        # Extract generics
        generics = self._extract_generics(node, source)
        lifetimes = self._extract_lifetimes(node, source)

        # Extract attributes (including #[derive])
        attributes = self._extract_attributes(node, source)

        # Build decorators list
        decorators = attributes.copy()
        if generics:
            decorators.append(f"generics:{generics}")
        if lifetimes:
            decorators.append(f"lifetimes:{','.join(lifetimes)}")

        # Check visibility
        is_private = self._is_rust_private(node, source)

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
            bases=[],  # Structs don't have inheritance in Rust
            decorators=decorators,
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    def _extract_trait_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
    ) -> Optional[ExtractedCode]:
        """
        Extract a trait from its AST node.

        Args:
            node: The trait_item node.
            source: Original source code.
            file_path: Path to the file.

        Returns:
            ExtractedCode object or None if extraction fails.
        """
        # Get trait name
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
        docstring = self._extract_rust_doc(node, source)

        # Extract method signatures
        methods = self._extract_trait_methods(node, source)

        # Extract supertraits (trait bounds)
        supertraits = self._extract_supertraits(node, source)

        # Extract generics
        generics = self._extract_generics(node, source)
        lifetimes = self._extract_lifetimes(node, source)

        # Extract attributes
        attributes = self._extract_attributes(node, source)

        # Build decorators list
        decorators = attributes.copy()
        if generics:
            decorators.append(f"generics:{generics}")
        if lifetimes:
            decorators.append(f"lifetimes:{','.join(lifetimes)}")

        # Check visibility
        is_private = self._is_rust_private(node, source)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.INTERFACE,  # Traits are like interfaces
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=methods,
            bases=supertraits,
            decorators=decorators,
            is_abstract=True,  # Traits are inherently abstract
            is_private=is_private,
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
            node: The enum_item node.
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

        # Extract doc comments
        docstring = self._extract_rust_doc(node, source)

        # Extract variant names
        variants = self._extract_enum_variants(node, source)

        # Extract generics
        generics = self._extract_generics(node, source)
        lifetimes = self._extract_lifetimes(node, source)

        # Extract attributes
        attributes = self._extract_attributes(node, source)

        # Build decorators list
        decorators = attributes.copy()
        if generics:
            decorators.append(f"generics:{generics}")
        if lifetimes:
            decorators.append(f"lifetimes:{','.join(lifetimes)}")

        # Check visibility
        is_private = self._is_rust_private(node, source)

        return ExtractedCode(
            name=name,
            qualified_name=name,
            element_type=CodeElementType.ENUM,
            language=self.language_name,
            file_path=file_path,
            location=location,
            source_code=source_code,
            docstring=docstring,
            methods=variants,  # Store variant names in methods
            bases=[],
            decorators=decorators,
            is_abstract=False,
            is_private=is_private,
            line_count=location.end_line - location.start_line + 1,
        )

    # =========================================================================
    # Helper Methods - Doc Comments
    # =========================================================================

    def _extract_rust_doc(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract Rust doc comments from preceding siblings.

        Rust doc comments are:
        - /// outer doc comments (before the item)
        - //! inner doc comments (at start of item)
        - /** */ block doc comments
        - /*! */ inner block doc comments

        Args:
            node: The node to find doc comments for.
            source: Original source code.

        Returns:
            Doc comment content without comment markers, or None if not found.
        """
        comments: List[str] = []
        prev = node.prev_sibling

        # Collect all adjacent line_comment and block_comment nodes
        while prev:
            if prev.type == "line_comment":
                comment_text = self._get_node_text(prev, source)
                if comment_text.startswith("///") or comment_text.startswith("//!"):
                    comments.insert(0, comment_text)
                    prev = prev.prev_sibling
                else:
                    break
            elif prev.type == "block_comment":
                comment_text = self._get_node_text(prev, source)
                if comment_text.startswith("/**") or comment_text.startswith("/*!"):
                    comments.insert(0, comment_text)
                    prev = prev.prev_sibling
                else:
                    break
            elif prev.type == "attribute_item":
                # Skip over attributes when looking for doc comments
                prev = prev.prev_sibling
            else:
                break

        if not comments:
            return None

        # Clean up comments
        cleaned_lines = []
        for comment in comments:
            if comment.startswith("///"):
                # Remove /// and leading space
                line = comment[3:].lstrip()
                cleaned_lines.append(line)
            elif comment.startswith("//!"):
                # Remove //! and leading space
                line = comment[3:].lstrip()
                cleaned_lines.append(line)
            elif comment.startswith("/**") and comment.endswith("*/"):
                # Remove /** and */ and process block
                block = comment[3:-2].strip()
                # Handle multi-line block comments
                for line in block.split("\n"):
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].lstrip()
                    cleaned_lines.append(line)
            elif comment.startswith("/*!") and comment.endswith("*/"):
                # Remove /*! and */ and process block
                block = comment[3:-2].strip()
                for line in block.split("\n"):
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].lstrip()
                    cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines).strip()

        return None

    # =========================================================================
    # Helper Methods - Visibility
    # =========================================================================

    def _is_rust_private(self, node: Node, source: bytes) -> bool:
        """
        Check if a Rust item is private (no visibility modifier).

        In Rust:
        - No modifier = private (module-level visibility)
        - pub = public
        - pub(crate) = crate-level visibility
        - pub(super) = parent module visibility
        - pub(in path) = custom path visibility

        Args:
            node: The item node to check.
            source: Original source code.

        Returns:
            True if the item has no visibility modifier (private).
        """
        # Check for visibility_modifier in children
        for child in node.children:
            if child.type == "visibility_modifier":
                return False

        return True

    def _extract_visibility(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract the visibility modifier from a node.

        Args:
            node: The node to extract visibility from.
            source: Original source code.

        Returns:
            Visibility string (e.g., "pub", "pub(crate)") or None if private.
        """
        for child in node.children:
            if child.type == "visibility_modifier":
                return self._get_node_text(child, source)

        return None

    # =========================================================================
    # Helper Methods - Parameters
    # =========================================================================

    def _extract_parameters(self, node: Node, source: bytes) -> List[ParameterInfo]:
        """
        Extract parameters from a function definition.

        Handles Rust's parameter syntax including:
        - Regular parameters: name: Type
        - Reference parameters: name: &Type, name: &mut Type
        - Self parameters: self, &self, &mut self

        Args:
            node: Function or method node.
            source: Original source code.

        Returns:
            List of ParameterInfo objects.
        """
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return []

        parameters: List[ParameterInfo] = []

        for child in params_node.children:
            if child.type == "parameter":
                param_info = self._extract_parameter(child, source)
                if param_info:
                    parameters.append(param_info)
            elif child.type == "self_parameter":
                # Handle self, &self, &mut self
                self_text = self._get_node_text(child, source)
                parameters.append(
                    ParameterInfo(
                        name="self",
                        type_annotation=self_text,
                        default_value=None,
                    )
                )

        return parameters

    def _extract_parameter(self, node: Node, source: bytes) -> Optional[ParameterInfo]:
        """
        Extract a single parameter from a parameter node.

        Args:
            node: Parameter node.
            source: Original source code.

        Returns:
            ParameterInfo or None.
        """
        pattern_node = node.child_by_field_name("pattern")
        type_node = node.child_by_field_name("type")

        if not pattern_node:
            return None

        name = self._get_node_text(pattern_node, source)
        type_annotation = self._get_node_text(type_node, source) if type_node else None

        return ParameterInfo(
            name=name,
            type_annotation=type_annotation,
            default_value=None,  # Rust doesn't have default parameters
        )

    def _extract_receiver_info(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract receiver information from a method.

        Args:
            node: Function node.
            source: Original source code.

        Returns:
            Receiver type ("self", "&self", "&mut self") or None.
        """
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return None

        for child in params_node.children:
            if child.type == "self_parameter":
                return self._get_node_text(child, source)

        return None

    # =========================================================================
    # Helper Methods - Return Type
    # =========================================================================

    def _extract_return_type(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract return type from a function definition.

        Args:
            node: Function node.
            source: Original source code.

        Returns:
            Return type as string, or None if no return type.
        """
        return_type_node = node.child_by_field_name("return_type")
        if not return_type_node:
            return None

        return self._get_node_text(return_type_node, source)

    # =========================================================================
    # Helper Methods - Async Detection
    # =========================================================================

    def _is_async_function(self, node: Node, source: bytes) -> bool:
        """
        Check if a function is async.

        Args:
            node: Function node.
            source: Original source code.

        Returns:
            True if the function is async.
        """
        source_text = self._get_node_text(node, source)
        # Check if function starts with async or has async modifier
        return source_text.strip().startswith("async ")

    # =========================================================================
    # Helper Methods - Generics and Lifetimes
    # =========================================================================

    def _extract_generics(self, node: Node, source: bytes) -> Optional[str]:
        """
        Extract type parameters (generics) from a declaration.

        Handles generic parameters with bounds like <T: Display + Clone>.

        Args:
            node: Function, struct, trait, or enum node.
            source: Original source code.

        Returns:
            Type parameters as string (without angle brackets), or None.
        """
        type_params_node = node.child_by_field_name("type_parameters")
        if type_params_node:
            params_text = self._get_node_text(type_params_node, source)
            # Remove angle brackets
            if params_text.startswith("<") and params_text.endswith(">"):
                return params_text[1:-1]
            return params_text
        return None

    def _extract_lifetimes(self, node: Node, source: bytes) -> List[str]:
        """
        Extract lifetime parameters from a declaration.

        Handles lifetimes like 'a, 'static.

        Args:
            node: Function, struct, trait, or enum node.
            source: Original source code.

        Returns:
            List of lifetime names.
        """
        lifetimes: List[str] = []

        type_params_node = node.child_by_field_name("type_parameters")
        if type_params_node:
            for child in type_params_node.children:
                if child.type == "lifetime":
                    lifetime = self._get_node_text(child, source)
                    lifetimes.append(lifetime)

        return lifetimes

    # =========================================================================
    # Helper Methods - Attributes
    # =========================================================================

    def _extract_attributes(self, node: Node, source: bytes) -> List[str]:
        """
        Extract attributes from preceding siblings.

        Handles #[derive(...)], #[cfg(...)], etc.

        Args:
            node: The node to find attributes for.
            source: Original source code.

        Returns:
            List of attribute strings.
        """
        attributes: List[str] = []
        prev = node.prev_sibling

        # Collect all adjacent attribute_item nodes
        while prev:
            if prev.type == "attribute_item":
                attr_text = self._get_node_text(prev, source)
                # Clean up the attribute (remove #[ and ])
                if attr_text.startswith("#[") and attr_text.endswith("]"):
                    attr_text = attr_text[2:-1]
                attributes.insert(0, attr_text)
                prev = prev.prev_sibling
            elif prev.type in ("line_comment", "block_comment"):
                # Skip over comments when collecting attributes
                prev = prev.prev_sibling
            else:
                break

        return attributes

    # =========================================================================
    # Helper Methods - impl Blocks
    # =========================================================================

    def _extract_impl_context(
        self, node: Node, source: bytes
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the type and trait from an impl block.

        Handles:
        - impl Type { ... }
        - impl Trait for Type { ... }

        Args:
            node: The impl_item node.
            source: Original source code.

        Returns:
            Tuple of (impl_type, trait_name) where trait_name is None for inherent impls.
        """
        impl_type: Optional[str] = None
        trait_name: Optional[str] = None

        # Get the type being implemented
        type_node = node.child_by_field_name("type")
        if type_node:
            impl_type = self._get_node_text(type_node, source)

        # Check for trait implementation
        trait_node = node.child_by_field_name("trait")
        if trait_node:
            trait_name = self._get_node_text(trait_node, source)

        return impl_type, trait_name

    # =========================================================================
    # Helper Methods - Struct Fields
    # =========================================================================

    def _extract_struct_fields(self, node: Node, source: bytes) -> List[str]:
        """
        Extract field names from a struct.

        Handles:
        - Regular structs with named fields
        - Tuple structs (returns numbered field names)
        - Unit structs (returns empty list)

        Args:
            node: The struct_item node.
            source: Original source code.

        Returns:
            List of field names.
        """
        fields: List[str] = []

        # Look for field_declaration_list (regular struct)
        for child in node.children:
            if child.type == "field_declaration_list":
                for field_child in child.children:
                    if field_child.type == "field_declaration":
                        name_node = field_child.child_by_field_name("name")
                        if name_node:
                            fields.append(self._get_node_text(name_node, source))

            # Handle tuple struct
            elif child.type == "ordered_field_declaration_list":
                field_index = 0
                for field_child in child.children:
                    if field_child.type == "ordered_field_declaration":
                        fields.append(f"_{field_index}")
                        field_index += 1

        return fields

    # =========================================================================
    # Helper Methods - Trait Methods
    # =========================================================================

    def _extract_trait_methods(self, node: Node, source: bytes) -> List[str]:
        """
        Extract method names from a trait.

        Args:
            node: The trait_item node.
            source: Original source code.

        Returns:
            List of method names.
        """
        methods: List[str] = []

        # Find declaration_list within the trait
        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_signature_item":
                        name_node = item.child_by_field_name("name")
                        if name_node:
                            methods.append(self._get_node_text(name_node, source))
                    elif item.type == "function_item":
                        name_node = item.child_by_field_name("name")
                        if name_node:
                            methods.append(self._get_node_text(name_node, source))

        return methods

    def _extract_supertraits(self, node: Node, source: bytes) -> List[str]:
        """
        Extract supertrait bounds from a trait.

        Handles: trait Foo: Bar + Baz { ... }

        Args:
            node: The trait_item node.
            source: Original source code.

        Returns:
            List of supertrait names.
        """
        supertraits: List[str] = []

        # Look for trait_bounds
        bounds_node = node.child_by_field_name("bounds")
        if bounds_node:
            for child in bounds_node.children:
                if child.type == "type_identifier":
                    supertraits.append(self._get_node_text(child, source))
                elif child.type == "generic_type":
                    supertraits.append(self._get_node_text(child, source))
                elif child.type == "scoped_type_identifier":
                    supertraits.append(self._get_node_text(child, source))

        return supertraits

    # =========================================================================
    # Helper Methods - Enum Variants
    # =========================================================================

    def _extract_enum_variants(self, node: Node, source: bytes) -> List[str]:
        """
        Extract variant names from an enum.

        Handles:
        - Unit variants: None
        - Tuple variants: Some(T)
        - Struct variants: Variant { field: Type }

        Args:
            node: The enum_item node.
            source: Original source code.

        Returns:
            List of variant names.
        """
        variants: List[str] = []

        # Find enum_variant_list
        for child in node.children:
            if child.type == "enum_variant_list":
                for variant in child.children:
                    if variant.type == "enum_variant":
                        name_node = variant.child_by_field_name("name")
                        if name_node:
                            variants.append(self._get_node_text(name_node, source))

        return variants


# =============================================================================
# Auto-registration
# =============================================================================


def _register_rust_extractor() -> None:
    """
    Register the RustExtractor for Rust language.

    This function is called when the module is imported to make
    the extractor available for Rust files.
    """
    from ..parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    registry.register_extractor("rust", RustExtractor())
    logger.debug("RustExtractor registered for 'rust' language")


# Auto-register on module import
_register_rust_extractor()


__all__ = [
    "RustExtractor",
    "RUST_FUNCTION_TYPES",
    "RUST_STRUCT_TYPES",
    "RUST_TRAIT_TYPES",
    "RUST_ENUM_TYPES",
    "RUST_MACRO_TYPES",
    "RUST_IMPL_TYPES",
]
