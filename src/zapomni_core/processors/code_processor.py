"""
CodeProcessor utilities for analyzing Python source with the AST module.

Provides lightweight helpers to parse code, validate syntax, and enumerate
functions, classes, and imports while preserving metadata such as line numbers
and qualified names.
"""

import ast
from typing import Any, Dict, List, Sequence, Union

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.utils import get_logger

logger = get_logger(__name__)


def _ensure_code(code: Any) -> str:
    if not isinstance(code, str):
        logger.error("code_not_str", received_type=type(code).__name__)
        raise ValidationError("Code must be provided as a string.")
    return code


def _parse_tree(code: str) -> ast.Module:
    validated = _ensure_code(code)
    try:
        return ast.parse(validated)
    except SyntaxError as exc:
        logger.error(
            "syntax_error",
            message=str(exc),
            lineno=exc.lineno,
            offset=exc.offset,
        )
        raise ProcessingError(
            message="Failed to parse Python source code.",
            details={
                "lineno": exc.lineno,
                "offset": exc.offset,
                "text": exc.text.strip() if exc.text else None,
            },
            original_exception=exc,
        )


def _collect_arg_names(arguments: ast.arguments) -> List[str]:
    names: List[str] = []
    for arg in getattr(arguments, "posonlyargs", []):
        names.append(arg.arg)
    for arg in arguments.args:
        names.append(arg.arg)
    if arguments.vararg:
        names.append(f"*{arguments.vararg.arg}")
    for arg in arguments.kwonlyargs:
        names.append(arg.arg)
    if arguments.kwarg:
        names.append(f"**{arguments.kwarg.arg}")
    return names


def _build_function_summary(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    parent_names: Sequence[str],
    is_async: bool,
) -> Dict[str, Any]:
    qualified_parent = ".".join(parent_names) if parent_names else ""
    qualified_name = f"{qualified_parent}.{node.name}" if qualified_parent else node.name
    parent = parent_names[-1] if parent_names else None
    return {
        "name": node.name,
        "qualified_name": qualified_name,
        "args": _collect_arg_names(node.args),
        "docstring": ast.get_docstring(node),
        "lineno": node.lineno,
        "end_lineno": getattr(node, "end_lineno", None),
        "is_async": is_async,
        "parent": parent,
    }


def _collect_functions(tree: ast.Module) -> List[Dict[str, Any]]:
    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: List[Dict[str, Any]] = []
            self.function_stack: List[str] = []
            self.class_depth = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_depth += 1
            self.generic_visit(node)
            self.class_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self.class_depth == 0:
                self.functions.append(
                    _build_function_summary(node, tuple(self.function_stack), False)
                )
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if self.class_depth == 0:
                self.functions.append(
                    _build_function_summary(node, tuple(self.function_stack), True)
                )
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

    visitor = FunctionVisitor()
    visitor.visit(tree)
    return visitor.functions


def _collect_methods(node: ast.ClassDef, parent_names: List[str]) -> List[Dict[str, Any]]:
    methods: List[Dict[str, Any]] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            methods.append(_build_function_summary(child, parent_names, False))
        elif isinstance(child, ast.AsyncFunctionDef):
            methods.append(_build_function_summary(child, parent_names, True))
    return methods


def _collect_classes(tree: ast.Module) -> List[Dict[str, Any]]:
    class ClassVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: List[Dict[str, Any]] = []
            self.parent_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            qualified_parent = ".".join(self.parent_stack) if self.parent_stack else ""
            qualified_name = f"{qualified_parent}.{node.name}" if qualified_parent else node.name
            entry = {
                "name": node.name,
                "qualified_name": qualified_name,
                "bases": [ast.unparse(base) for base in node.bases],
                "docstring": ast.get_docstring(node),
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", None),
                "parent": self.parent_stack[-1] if self.parent_stack else None,
                "methods": _collect_methods(node, self.parent_stack + [node.name]),
            }
            self.classes.append(entry)

            self.parent_stack.append(node.name)
            self.generic_visit(node)
            self.parent_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.parent_stack.append(node.name)
            self.generic_visit(node)
            self.parent_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.parent_stack.append(node.name)
            self.generic_visit(node)
            self.parent_stack.pop()

    visitor = ClassVisitor()
    visitor.visit(tree)
    return visitor.classes


def _format_import(node: Union[ast.Import, ast.ImportFrom], code: str) -> str:
    statement = ast.get_source_segment(code, node)
    if statement:
        return statement.strip()
    return ast.unparse(node)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, code: str) -> None:
        self.code = code
        self.statements: List[str] = []
        self.details: List[Dict[str, Any]] = []

    def _record(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        statement = _format_import(node, self.code)
        self.statements.append(statement)
        self.details.append({"statement": statement, "lineno": node.lineno})

    def visit_Import(self, node: ast.Import) -> None:
        self._record(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record(node)


def _collect_import_metadata(tree: ast.Module, code: str) -> _ImportCollector:
    collector = _ImportCollector(code)
    collector.visit(tree)
    return collector


def _collect_imports(tree: ast.Module, code: str) -> List[str]:
    return _collect_import_metadata(tree, code).statements


def _collect_import_details(tree: ast.Module, code: str) -> List[Dict[str, Any]]:
    return _collect_import_metadata(tree, code).details


def parse_python(code: str) -> Dict[str, Any]:
    """
    Parse Python source and return metadata dictionaries.

    Returns functions, classes, and imports parsed from the AST.
    """
    tree = _parse_tree(code)
    return {
        "functions": _collect_functions(tree),
        "classes": _collect_classes(tree),
        "imports": _collect_imports(tree, code),
        "import_details": _collect_import_details(tree, code),
    }


def extract_functions(code: str) -> List[Dict[str, Any]]:
    """
    Extract all module-level function summaries from the source.
    """
    tree = _parse_tree(code)
    return _collect_functions(tree)


def extract_classes(code: str) -> List[Dict[str, Any]]:
    """
    Extract class summaries, including methods and base classes.
    """
    tree = _parse_tree(code)
    return _collect_classes(tree)


def extract_imports(code: str) -> List[str]:
    """
    Extract import statements in order of appearance.
    """
    tree = _parse_tree(code)
    return _collect_imports(tree, code)


def validate_syntax(code: str) -> bool:
    """
    Validate Python syntax, returning True when code parses successfully.
    """
    try:
        ast.parse(_ensure_code(code))
        return True
    except SyntaxError as exc:
        logger.warning(
            "syntax_validation_failed",
            message=str(exc),
            lineno=exc.lineno,
            offset=exc.offset,
        )
        return False
