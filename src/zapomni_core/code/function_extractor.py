"""
FunctionExtractor - Extract detailed function/method metadata from code.

Implements AST-based extraction of function metadata for code analysis and knowledge graphs.
Extracts signatures, parameters, docstrings, decorators, line ranges, and complexity metrics.

Achieves extraction goals:
- Function metadata: Name, signature, parameters, return type, docstring
- Code structure: Line ranges, decorators, async/generator/property status
- Quality metrics: Cyclomatic complexity score (when radon available)

Performance:
- AST parsing: ~1-10ms per file (depending on size)
- Metadata extraction: O(n) where n = number of functions
- Optional complexity calculation: ~10-50ms per file

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import structlog

from zapomni_core.exceptions import ExtractionError, ValidationError

logger = structlog.get_logger()

# Try to import radon for complexity calculation (optional)
try:
    from radon.complexity import ComplexityVisitor

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Parameter:
    """
    Function parameter metadata.

    Attributes:
        name: Parameter name
        annotation: Type annotation (if present)
        default: Default value representation (if present)
        kind: Parameter kind (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, etc.)
    """

    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    kind: str = "POSITIONAL_OR_KEYWORD"


@dataclass
class FunctionMetadata:
    """
    Complete function metadata extracted from AST.

    Attributes:
        name: Function name
        file_path: File path (optional, passed by caller)
        start_line: Starting line number in source file
        end_line: Ending line number in source file
        signature: Function signature string (e.g., "def foo(x: int) -> str:")
        parameters: List of Parameter objects
        return_type: Return type annotation (if present)
        docstring: Docstring (if present)
        decorators: List of decorator names (e.g., ["property", "staticmethod"])
        is_async: Whether function is async
        is_generator: Whether function contains yield statements
        is_property: Whether function has @property decorator
        complexity_score: Cyclomatic complexity (if calculated)
        body_lines: Source code lines of function (if available)
    """

    name: str
    start_line: int
    end_line: int
    signature: str
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    complexity_score: Optional[int] = None
    body_lines: List[str] = field(default_factory=list)
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "parameters": [
                {
                    "name": p.name,
                    "annotation": p.annotation,
                    "default": p.default,
                    "kind": p.kind,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "is_property": self.is_property,
            "complexity_score": self.complexity_score,
            "num_lines": self.end_line - self.start_line + 1,
        }


# ============================================================================
# FunctionExtractor Class
# ============================================================================


class FunctionExtractor:
    """
    Extract detailed function/method metadata from Python source code using AST.

    Provides comprehensive function metadata extraction:
    1. Parse Python source with ast module
    2. Extract all functions (top-level and nested)
    3. Gather metadata: signature, parameters, return type
    4. Extract docstring and decorators
    5. Calculate complexity (optional, requires radon)
    6. Detect async, generator, and property functions

    Attributes:
        extract_body: Whether to include function body lines
        calculate_complexity: Whether to calculate cyclomatic complexity
        max_file_size: Maximum file size to process (bytes)
    """

    def __init__(
        self,
        extract_body: bool = False,
        calculate_complexity: bool = False,
        max_file_size: int = 10_000_000,  # 10MB default
    ) -> None:
        """
        Initialize FunctionExtractor.

        Args:
            extract_body: Whether to include function body in extraction (default: False)
            calculate_complexity: Whether to calculate cyclomatic complexity (default: False)
            max_file_size: Maximum file size in bytes to process (default: 10MB)

        Raises:
            ValueError: If max_file_size is not positive
        """
        if max_file_size <= 0:
            raise ValueError(f"max_file_size must be positive, got {max_file_size}")

        self.extract_body = extract_body
        self._calculate_complexity_enabled = calculate_complexity
        self.max_file_size = max_file_size

        # Warn if complexity requested but radon not available
        if calculate_complexity and not RADON_AVAILABLE:
            logger.warning(
                "complexity_calculation_requested_but_radon_unavailable",
                hint="Install radon: pip install radon",
            )

        logger.info(
            "function_extractor_initialized",
            extract_body=extract_body,
            calculate_complexity=calculate_complexity,
            radon_available=RADON_AVAILABLE,
        )

    def extract_functions(
        self,
        source_code: str,
        file_path: Optional[str] = None,
    ) -> List[FunctionMetadata]:
        """
        Extract all functions from Python source code.

        Workflow:
        1. Validate input
        2. Parse with ast module
        3. Walk AST and find all function definitions
        4. Extract metadata for each function
        5. Optional: calculate complexity
        6. Return sorted by line number

        Args:
            source_code: Python source code string
            file_path: Optional file path for reference

        Returns:
            List of FunctionMetadata objects sorted by start line

        Raises:
            ValidationError: If source code is empty or too large
            ExtractionError: If AST parsing fails
        """
        # Validate input
        if not source_code or not source_code.strip():
            raise ValidationError(
                message="Source code cannot be empty",
                error_code="VAL_001",
                details={"file_path": file_path},
            )

        if len(source_code.encode("utf-8")) > self.max_file_size:
            raise ValidationError(
                message=f"Source code exceeds max size ({self.max_file_size} bytes)",
                error_code="VAL_002",
                details={
                    "file_path": file_path,
                    "size": len(source_code.encode("utf-8")),
                    "max_size": self.max_file_size,
                },
            )

        try:
            # Parse source code to AST
            ast_tree = ast.parse(source_code)
            logger.debug("ast_parsing_complete", file_path=file_path)

            # Extract functions from AST
            functions = self._extract_from_ast(
                ast_tree,
                source_code,
                file_path,
            )

            # Sort by line number
            functions = sorted(functions, key=lambda f: f.start_line)

            logger.info(
                "function_extraction_complete",
                num_functions=len(functions),
                file_path=file_path,
            )

            return functions

        except ValidationError:
            raise
        except SyntaxError as e:
            raise ExtractionError(
                message=f"Syntax error in source code: {str(e)}",
                error_code="EXTR_001",
                details={
                    "file_path": file_path,
                    "line": e.lineno,
                    "error": str(e),
                },
                original_exception=e,
            )
        except Exception as e:
            raise ExtractionError(
                message=f"Function extraction failed: {str(e)}",
                error_code="EXTR_001",
                details={"file_path": file_path},
                original_exception=e,
            )

    def get_function_signature(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> str:
        """
        Get function signature string from AST node.

        Generates signature in form: "def/async def name(params) -> return_type:"

        Args:
            node: AST function node

        Returns:
            Function signature string

        Raises:
            ExtractionError: If signature generation fails
        """
        try:
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            name = node.name
            params_str = self._format_parameters(node.args)
            return_type = ""

            if node.returns:
                return_type_str = ast.unparse(node.returns)
                return_type = f" -> {return_type_str}"

            return f"{prefix} {name}({params_str}){return_type}:"

        except Exception as e:
            raise ExtractionError(
                message=f"Failed to get function signature: {str(e)}",
                error_code="EXTR_002",
                original_exception=e,
            )

    def get_function_body(
        self,
        source_code: str,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> List[str]:
        """
        Extract function body lines from source code.

        Args:
            source_code: Complete source code
            node: AST function node

        Returns:
            List of source code lines in function body

        Raises:
            ValidationError: If line numbers invalid
        """
        try:
            lines = source_code.split("\n")

            # Line numbers in AST are 1-indexed
            start_idx = node.body[0].lineno - 1 if node.body else node.lineno
            end_idx = node.end_lineno  # end_lineno is inclusive

            if start_idx < 0 or end_idx > len(lines):
                logger.warning(
                    "invalid_line_numbers",
                    start=start_idx,
                    end=end_idx,
                    total_lines=len(lines),
                )
                return []

            return lines[start_idx:end_idx]

        except Exception as e:
            logger.warning(
                "function_body_extraction_failed",
                error=str(e),
            )
            return []

    def get_docstring(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Optional[str]:
        """
        Extract docstring from function node.

        Uses ast.get_docstring which handles both single-line and multi-line docstrings.

        Args:
            node: AST function node

        Returns:
            Docstring if present, None otherwise
        """
        try:
            docstring = ast.get_docstring(node)
            return docstring.strip() if docstring else None
        except Exception as e:
            logger.debug(
                "docstring_extraction_failed",
                function=node.name,
                error=str(e),
            )
            return None

    def calculate_complexity(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Optional[int]:
        """
        Calculate cyclomatic complexity of function.

        Requires radon library. If not available or complexity disabled, returns None.

        Uses radon's ComplexityVisitor for accurate calculation following
        McCabe's cyclomatic complexity metric.

        Args:
            node: AST function node

        Returns:
            Cyclomatic complexity score (1-based, 1 = simplest) or None

        Raises:
            ExtractionError: If radon available but calculation fails
        """
        # Return None if not requested or radon not available
        if not self._calculate_complexity_enabled or not RADON_AVAILABLE:
            return None

        try:
            visitor = ComplexityVisitor.from_function(node)
            return visitor.complexity

        except Exception as e:
            logger.warning(
                "complexity_calculation_failed",
                function=node.name,
                error=str(e),
            )
            return None

    def _extract_from_ast(
        self,
        ast_tree: ast.Module,
        source_code: str,
        file_path: Optional[str] = None,
    ) -> List[FunctionMetadata]:
        """
        Internal: Walk AST and extract all functions.

        Args:
            ast_tree: Parsed AST tree
            source_code: Original source code
            file_path: Optional file path

        Returns:
            List of FunctionMetadata objects
        """
        functions = []

        # Find all function nodes (top-level and nested)
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    metadata = self._extract_function_metadata(
                        node,
                        source_code,
                        file_path,
                    )
                    functions.append(metadata)
                except Exception as e:
                    logger.warning(
                        "function_metadata_extraction_failed",
                        function=getattr(node, "name", "unknown"),
                        error=str(e),
                    )
                    # Continue with next function instead of failing

        return functions

    def _extract_function_metadata(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source_code: str,
        file_path: Optional[str] = None,
    ) -> FunctionMetadata:
        """
        Internal: Extract complete metadata for a single function.

        Args:
            node: AST function node
            source_code: Original source code
            file_path: Optional file path

        Returns:
            FunctionMetadata object

        Raises:
            ExtractionError: If critical metadata extraction fails
        """
        # Get basic metadata
        name = node.name
        start_line = node.lineno
        end_line = node.end_lineno or node.lineno

        # Get signature
        signature = self.get_function_signature(node)

        # Get parameters
        parameters = self._extract_parameters(node.args)

        # Get return type
        return_type = ast.unparse(node.returns) if node.returns else None

        # Get docstring
        docstring = self.get_docstring(node)

        # Get decorators
        decorators = [ast.unparse(dec).split("(")[0].split(".")[-1] for dec in node.decorator_list]

        # Check if async
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Check if generator (has yield or yield from)
        is_generator = self._has_yield(node)

        # Check if property
        is_property = "property" in decorators

        # Calculate complexity (optional)
        complexity_score = self.calculate_complexity(node)

        # Get body lines (optional)
        body_lines = []
        if self.extract_body:
            body_lines = self.get_function_body(source_code, node)

        return FunctionMetadata(
            name=name,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
            is_property=is_property,
            complexity_score=complexity_score,
            body_lines=body_lines,
            file_path=file_path,
        )

    def _extract_parameters(
        self,
        args_node: ast.arguments,
    ) -> List[Parameter]:
        """
        Internal: Extract parameter list from function arguments node.

        Args:
            args_node: ast.arguments node

        Returns:
            List of Parameter objects
        """
        parameters = []

        # Handle positional parameters
        for arg in args_node.args:
            param = Parameter(
                name=arg.arg,
                annotation=(ast.unparse(arg.annotation) if arg.annotation else None),
                kind="POSITIONAL_OR_KEYWORD",
            )
            parameters.append(param)

        # Handle *args
        if args_node.vararg:
            param = Parameter(
                name=f"*{args_node.vararg.arg}",
                annotation=(
                    ast.unparse(args_node.vararg.annotation)
                    if args_node.vararg.annotation
                    else None
                ),
                kind="VAR_POSITIONAL",
            )
            parameters.append(param)

        # Handle keyword-only parameters
        for arg in args_node.kwonlyargs:
            param = Parameter(
                name=arg.arg,
                annotation=(ast.unparse(arg.annotation) if arg.annotation else None),
                kind="KEYWORD_ONLY",
            )
            parameters.append(param)

        # Handle **kwargs
        if args_node.kwarg:
            param = Parameter(
                name=f"**{args_node.kwarg.arg}",
                annotation=(
                    ast.unparse(args_node.kwarg.annotation) if args_node.kwarg.annotation else None
                ),
                kind="VAR_KEYWORD",
            )
            parameters.append(param)

        # Add default values
        num_defaults = len(args_node.defaults)
        num_args = len(args_node.args)

        for i, default in enumerate(args_node.defaults):
            # Defaults apply to the last N arguments
            arg_idx = num_args - num_defaults + i
            if arg_idx < len(parameters):
                parameters[arg_idx].default = ast.unparse(default)

        # Add keyword-only defaults
        for arg, default in zip(args_node.kwonlyargs, args_node.kw_defaults or []):
            for param in parameters:
                if param.name == arg.arg and default:
                    param.default = ast.unparse(default)

        return parameters

    def _format_parameters(self, args_node: ast.arguments) -> str:
        """
        Internal: Format parameters as string.

        Args:
            args_node: ast.arguments node

        Returns:
            Formatted parameter string
        """
        params = self._extract_parameters(args_node)
        parts = []

        for param in params:
            part = param.name
            if param.annotation:
                part += f": {param.annotation}"
            if param.default:
                part += f" = {param.default}"
            parts.append(part)

        return ", ".join(parts)

    def _has_yield(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Internal: Check if function contains yield or yield from statements.

        Args:
            node: AST function node

        Returns:
            True if function is a generator
        """
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
