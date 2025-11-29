"""Pydantic models for Tree-sitter AST extraction."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CodeElementType(str, Enum):
    """Type of code element extracted from AST.

    Represents different structural elements that can be
    extracted from source code across various programming languages.
    """

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    MODULE = "module"
    IMPORT = "import"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    UNKNOWN = "unknown"


class ASTNodeLocation(BaseModel):
    """Location information for an AST node in source code.

    Stores both line/column positions and byte offsets for precise
    source code location tracking.
    """

    model_config = ConfigDict(frozen=True)

    start_line: int = Field(..., ge=0, description="Starting line number (0-indexed)")
    end_line: int = Field(..., ge=0, description="Ending line number (0-indexed)")
    start_column: int = Field(..., ge=0, description="Starting column number (0-indexed)")
    end_column: int = Field(..., ge=0, description="Ending column number (0-indexed)")
    start_byte: int = Field(..., ge=0, description="Starting byte offset in file")
    end_byte: int = Field(..., ge=0, description="Ending byte offset in file")


class ParameterInfo(BaseModel):
    """Information about a function/method parameter.

    Captures parameter name, optional type annotation, and default value
    for documenting function signatures.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Parameter name")
    type_annotation: Optional[str] = Field(
        default=None,
        description="Type annotation if present (e.g., 'str', 'int', 'List[str]')"
    )
    default_value: Optional[str] = Field(
        default=None,
        description="Default value as string if present"
    )


class ExtractedCode(BaseModel):
    """Extracted code element from AST parsing.

    Represents a single code element (function, class, method, etc.)
    extracted from source code with full metadata including location,
    docstrings, parameters, and various flags.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Identity
    name: str = Field(..., min_length=1, description="Element name (e.g., 'calculate_total')")
    qualified_name: str = Field(
        ...,
        min_length=1,
        description="Fully qualified name (e.g., 'module.ClassName.method')"
    )
    element_type: CodeElementType = Field(..., description="Type of code element")

    # Source information
    language: str = Field(..., min_length=1, description="Programming language (e.g., 'python')")
    file_path: str = Field(..., min_length=1, description="Absolute path to source file")
    location: ASTNodeLocation = Field(..., description="Location in source file")
    source_code: str = Field(..., min_length=1, description="Raw source code of the element")

    # Documentation
    docstring: Optional[str] = Field(default=None, description="Docstring if present")

    # Class-related
    parent_class: Optional[str] = Field(
        default=None,
        description="Parent class name for methods"
    )
    methods: List[str] = Field(
        default_factory=list,
        description="Method names for classes"
    )
    bases: List[str] = Field(
        default_factory=list,
        description="Base class names for classes"
    )

    # Function/method-related
    parameters: List[ParameterInfo] = Field(
        default_factory=list,
        description="Function/method parameters"
    )
    return_type: Optional[str] = Field(
        default=None,
        description="Return type annotation if present"
    )
    decorators: List[str] = Field(
        default_factory=list,
        description="Decorator names (e.g., ['staticmethod', 'property'])"
    )

    # Flags
    is_async: bool = Field(default=False, description="Whether function is async")
    is_generator: bool = Field(default=False, description="Whether function is a generator")
    is_static: bool = Field(default=False, description="Whether method is static")
    is_abstract: bool = Field(default=False, description="Whether method is abstract")
    is_private: bool = Field(
        default=False,
        description="Whether element is private (e.g., starts with _)"
    )

    # Metrics
    line_count: int = Field(default=0, ge=0, description="Number of lines in element")


class ParseResult(BaseModel):
    """Result of parsing a single source file.

    Contains all extracted code elements from a file along with
    parsing metadata and any errors encountered.
    """

    model_config = ConfigDict(validate_assignment=True)

    file_path: str = Field(..., min_length=1, description="Absolute path to parsed file")
    language: str = Field(..., min_length=1, description="Detected programming language")
    functions: List[ExtractedCode] = Field(
        default_factory=list,
        description="Extracted top-level functions"
    )
    classes: List[ExtractedCode] = Field(
        default_factory=list,
        description="Extracted classes with their methods"
    )
    parse_time_ms: float = Field(
        ...,
        ge=0,
        description="Time taken to parse file in milliseconds"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any parsing errors or warnings"
    )


class CallType(str, Enum):
    """Type of function/method call."""
    FUNCTION = "function"       # Direct function call: foo()
    METHOD = "method"           # Method call: obj.method()
    CONSTRUCTOR = "constructor" # Class instantiation: MyClass()
    STATIC = "static"           # Static method: Type::method() or Class.method()
    MACRO = "macro"             # Rust macros: println!()
    BUILTIN = "builtin"         # Built-in functions: print(), len()


class FunctionCall(BaseModel, frozen=True):
    """Represents a function/method call site in source code."""

    # Caller context
    caller_qualified_name: str
    caller_file_path: str

    # Callee info
    callee_name: str
    callee_qualified_name: Optional[str] = None

    # Location of call site
    location: ASTNodeLocation

    # Call characteristics
    call_type: CallType = CallType.FUNCTION
    receiver: Optional[str] = None
    arguments_count: int = 0
    is_await: bool = False
    is_chained: bool = False

    # Resolution status
    is_resolved: bool = False
    is_external: bool = False


class CallGraph(BaseModel):
    """Collection of function calls for a file or codebase."""

    file_path: str
    language: str
    calls: List[FunctionCall] = Field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def resolved_calls(self) -> int:
        return sum(1 for c in self.calls if c.is_resolved)

    @property
    def external_calls(self) -> int:
        return sum(1 for c in self.calls if c.is_external)
