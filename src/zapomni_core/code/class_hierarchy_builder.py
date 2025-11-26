"""
ClassHierarchyBuilder - Extract and build class inheritance hierarchies.

Provides comprehensive class metadata extraction from Python AST:
- Class definitions with base classes (single and multiple inheritance)
- Method enumeration with type classification (instance/class/static)
- Attribute extraction
- Decorator information
- Docstring preservation
- Line number tracking

Handles complex inheritance patterns including:
- Multiple inheritance
- Nested classes
- Abstract base classes
- Mixin patterns

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from zapomni_core.exceptions import ExtractionError, ValidationError
from zapomni_core.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MethodInfo:
    """
    Information about a class method.

    Attributes:
        name: Method name
        type: Method type ('instance', 'class', 'static')
        args: List of argument names (excluding self/cls for instance/class methods)
        docstring: Method docstring (if present)
        decorators: List of decorator names
        lineno: Line number where method is defined
        end_lineno: End line number (if available)
        is_async: Whether method is async
    """

    name: str
    type: str
    args: List[str]
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    lineno: int = 0
    end_lineno: Optional[int] = None
    is_async: bool = False


@dataclass
class AttributeInfo:
    """
    Information about a class attribute.

    Attributes:
        name: Attribute name
        type_annotation: Type annotation string (if present)
        value: Default value as string (if present)
        lineno: Line number where attribute is defined
    """

    name: str
    type_annotation: Optional[str] = None
    value: Optional[str] = None
    lineno: int = 0


@dataclass
class ClassInfo:
    """
    Complete information about a class definition.

    Attributes:
        name: Class name
        base_classes: List of base class names/qualified names
        methods: List of MethodInfo objects
        attributes: List of AttributeInfo objects
        docstring: Class docstring
        decorators: List of decorator names
        lineno: Line number where class is defined
        end_lineno: End line number (if available)
        parent_class: Name of parent class if nested (None if module-level)
        is_abstract: Whether class is abstract (has abc.ABC base or abstractmethod)
    """

    name: str
    base_classes: List[str] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    attributes: List[AttributeInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    lineno: int = 0
    end_lineno: Optional[int] = None
    parent_class: Optional[str] = None
    is_abstract: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert ClassInfo to dictionary."""
        return {
            "name": self.name,
            "base_classes": self.base_classes,
            "methods": [
                {
                    "name": m.name,
                    "type": m.type,
                    "args": m.args,
                    "docstring": m.docstring,
                    "decorators": m.decorators,
                    "lineno": m.lineno,
                    "end_lineno": m.end_lineno,
                    "is_async": m.is_async,
                }
                for m in self.methods
            ],
            "attributes": [
                {
                    "name": a.name,
                    "type_annotation": a.type_annotation,
                    "value": a.value,
                    "lineno": a.lineno,
                }
                for a in self.attributes
            ],
            "docstring": self.docstring,
            "decorators": self.decorators,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "parent_class": self.parent_class,
            "is_abstract": self.is_abstract,
        }


@dataclass
class HierarchyNode:
    """
    Node in the class hierarchy tree.

    Attributes:
        class_info: ClassInfo object
        children: List of child classes (subclasses)
        depth: Depth in hierarchy (0 = root)
    """

    class_info: ClassInfo
    children: List[HierarchyNode] = field(default_factory=list)
    depth: int = 0


# ============================================================================
# ClassHierarchyBuilder
# ============================================================================


class ClassHierarchyBuilder:
    """
    Extract and build class inheritance hierarchies from Python AST.

    Provides methods to:
    1. Extract all class definitions from an AST tree
    2. Retrieve base classes for inheritance tracking
    3. Extract methods with type classification
    4. Extract class attributes
    5. Build inheritance hierarchy tree

    Usage:
        builder = ClassHierarchyBuilder()
        ast_tree = ast.parse(source_code)
        classes = builder.extract_classes(ast_tree)
        hierarchy = builder.build_hierarchy(classes)
    """

    def __init__(self) -> None:
        """Initialize ClassHierarchyBuilder."""
        self.logger = logger
        self._class_name_map: Dict[str, ClassInfo] = {}

    def extract_classes(self, ast_tree: ast.Module) -> List[ClassInfo]:
        """
        Extract all class definitions from an AST tree.

        Args:
            ast_tree: Parsed AST module

        Returns:
            List of ClassInfo objects

        Raises:
            ValidationError: If ast_tree is not an ast.Module
            ExtractionError: If extraction fails
        """
        if not isinstance(ast_tree, ast.Module):
            raise ValidationError(
                "ast_tree must be an ast.Module instance", error_code="ERR_INVALID_AST_TYPE"
            )

        try:
            self._class_name_map = {}
            visitor = _ClassVisitor()
            visitor.visit(ast_tree)

            # Build name map for quick lookup
            for class_info in visitor.classes:
                self._class_name_map[class_info.name] = class_info

            self.logger.info("classes_extracted", count=len(visitor.classes))

            return visitor.classes

        except Exception as exc:
            self.logger.error("class_extraction_failed", error=str(exc))
            raise ExtractionError(
                message="Failed to extract classes from AST",
                details={"error": str(exc)},
                original_exception=exc,
            )

    def get_base_classes(self, node: ast.ClassDef) -> List[str]:
        """
        Get base classes for a class definition node.

        Handles:
        - Simple names: Parent
        - Qualified names: module.Parent
        - Subscripted types: Generic[T]

        Args:
            node: ast.ClassDef node

        Returns:
            List of base class names/qualifications
        """
        try:
            base_classes = []
            for base in node.bases:
                base_str = ast.unparse(base)
                base_classes.append(base_str)
            return base_classes
        except Exception as exc:
            self.logger.warning(
                "base_class_extraction_failed", class_name=node.name, error=str(exc)
            )
            return []

    def get_methods(self, node: ast.ClassDef) -> List[MethodInfo]:
        """
        Extract methods from a class definition.

        Classifies methods as:
        - instance: Regular instance methods (have 'self')
        - class: Class methods (decorated with @classmethod)
        - static: Static methods (decorated with @staticmethod)

        Args:
            node: ast.ClassDef node

        Returns:
            List of MethodInfo objects
        """
        methods: List[MethodInfo] = []

        try:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorator_names = self._get_decorator_names(item)
                    method_type = self._classify_method(decorator_names)

                    # Get argument names (skip 'self'/'cls' based on type)
                    args = self._get_method_args(item, method_type)

                    method_info = MethodInfo(
                        name=item.name,
                        type=method_type,
                        args=args,
                        docstring=ast.get_docstring(item),
                        decorators=decorator_names,
                        lineno=item.lineno,
                        end_lineno=getattr(item, "end_lineno", None),
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                    )
                    methods.append(method_info)

            return methods

        except Exception as exc:
            self.logger.warning("method_extraction_failed", class_name=node.name, error=str(exc))
            return []

    def get_attributes(self, node: ast.ClassDef) -> List[AttributeInfo]:
        """
        Extract attributes from a class definition.

        Looks for:
        - Annotated assignments: name: Type = value
        - Regular assignments in class body
        - Excludes method definitions

        Args:
            node: ast.ClassDef node

        Returns:
            List of AttributeInfo objects
        """
        attributes: List[AttributeInfo] = []

        try:
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    # Annotated assignment: name: Type = value
                    attr_name = item.target.id if isinstance(item.target, ast.Name) else None
                    if attr_name:
                        type_annotation = ast.unparse(item.annotation)
                        value = ast.unparse(item.value) if item.value else None

                        attributes.append(
                            AttributeInfo(
                                name=attr_name,
                                type_annotation=type_annotation,
                                value=value,
                                lineno=item.lineno,
                            )
                        )

                elif isinstance(item, ast.Assign):
                    # Regular assignment (without type annotation)
                    # Only capture simple assignments (single target)
                    if len(item.targets) == 1:
                        target = item.targets[0]
                        if isinstance(target, ast.Name):
                            # Skip if this looks like a method or constant-like name
                            # (heuristic: starts with _)
                            if not target.id.startswith("_"):
                                value = ast.unparse(item.value)
                                attributes.append(
                                    AttributeInfo(
                                        name=target.id,
                                        value=value,
                                        lineno=item.lineno,
                                    )
                                )

            return attributes

        except Exception as exc:
            self.logger.warning("attribute_extraction_failed", class_name=node.name, error=str(exc))
            return []

    def build_hierarchy(
        self,
        classes: List[ClassInfo],
    ) -> Dict[str, HierarchyNode]:
        """
        Build inheritance hierarchy from extracted classes.

        Returns a dictionary mapping class names to HierarchyNode objects
        representing the inheritance tree. Identifies root classes (with no
        base classes from the set) and builds parent-child relationships.

        Args:
            classes: List of ClassInfo objects

        Returns:
            Dict mapping class names to HierarchyNode objects

        Raises:
            ExtractionError: If hierarchy building fails
        """
        try:
            # Build name -> ClassInfo map
            class_map: Dict[str, ClassInfo] = {c.name: c for c in classes}

            # Create hierarchy nodes
            nodes: Dict[str, HierarchyNode] = {
                name: HierarchyNode(class_info=class_info, depth=0)
                for name, class_info in class_map.items()
            }

            # Build parent-child relationships
            for class_name, class_info in class_map.items():
                for base_name in class_info.base_classes:
                    # Handle simple base names (not fully qualified)
                    if base_name in class_map:
                        parent_node = nodes[base_name]
                        child_node = nodes[class_name]
                        parent_node.children.append(child_node)
                        # Update depth
                        child_node.depth = parent_node.depth + 1

            self.logger.info(
                "hierarchy_built",
                total_classes=len(classes),
                roots=sum(
                    1
                    for c in class_info.base_classes
                    if not c in class_map
                    for class_info in classes
                ),
            )

            return nodes

        except Exception as exc:
            self.logger.error("hierarchy_building_failed", error=str(exc))
            raise ExtractionError(
                message="Failed to build class hierarchy",
                details={"error": str(exc)},
                original_exception=exc,
            )

    # ========================================================================
    # Private helpers
    # ========================================================================

    @staticmethod
    def _get_decorator_names(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract decorator names from function/method node."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                # e.g., property.setter -> "property.setter"
                decorators.append(ast.unparse(decorator))
            elif isinstance(decorator, ast.Call):
                # e.g., @decorator(arg) -> "decorator"
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(ast.unparse(decorator.func))
        return decorators

    @staticmethod
    def _classify_method(decorators: List[str]) -> str:
        """Classify method type based on decorators."""
        if "classmethod" in decorators:
            return "class"
        elif "staticmethod" in decorators:
            return "static"
        else:
            return "instance"

    @staticmethod
    def _get_method_args(
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        method_type: str,
    ) -> List[str]:
        """Extract argument names, skipping 'self' or 'cls' as appropriate."""
        args = []
        start_index = 0

        # Skip first argument for instance and class methods
        if method_type in ("instance", "class"):
            start_index = 1

        arguments = node.args

        # Positional-only args
        for arg in getattr(arguments, "posonlyargs", [])[start_index:]:
            args.append(arg.arg)

        # Regular args
        for arg in arguments.args[start_index:]:
            args.append(arg.arg)

        # *args
        if arguments.vararg:
            args.append(f"*{arguments.vararg.arg}")

        # Keyword-only args
        for arg in arguments.kwonlyargs:
            args.append(arg.arg)

        # **kwargs
        if arguments.kwarg:
            args.append(f"**{arguments.kwarg.arg}")

        return args


# ============================================================================
# AST Visitor
# ============================================================================


class _ClassVisitor(ast.NodeVisitor):
    """AST visitor to extract class definitions."""

    def __init__(self) -> None:
        """Initialize visitor."""
        self.classes: List[ClassInfo] = []
        self.parent_stack: List[str] = []
        self._builder = ClassHierarchyBuilder()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit ClassDef node."""
        # Get base classes
        base_classes = self._builder.get_base_classes(node)

        # Get methods
        methods = self._builder.get_methods(node)

        # Get attributes
        attributes = self._builder.get_attributes(node)

        # Get decorators
        decorators = [
            decorator.id if isinstance(decorator, ast.Name) else ast.unparse(decorator)
            for decorator in node.decorator_list
        ]

        # Check if abstract
        is_abstract = "abstractmethod" in decorators or any("ABC" in base for base in base_classes)

        # Create ClassInfo
        class_info = ClassInfo(
            name=node.name,
            base_classes=base_classes,
            methods=methods,
            attributes=attributes,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            parent_class=self.parent_stack[-1] if self.parent_stack else None,
            is_abstract=is_abstract,
        )

        self.classes.append(class_info)

        # Visit nested classes
        self.parent_stack.append(node.name)
        self.generic_visit(node)
        self.parent_stack.pop()
