"""
Call graph analyzer for extracting function call relationships.

This module provides a multi-language call graph analyzer that uses tree-sitter
to extract function calls from source code and build caller/callee relationships.

Supports: Python, Go, Rust, TypeScript/JavaScript

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, List, Optional, Set, Tuple

import structlog
from tree_sitter import Node, Tree

from ..models import (
    ASTNodeLocation,
    CallGraph,
    CallType,
    CodeElementType,
    ExtractedCode,
    FunctionCall,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Built-in Functions by Language
# =============================================================================

PYTHON_BUILTINS: Set[str] = {
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict",
    "tuple", "set", "frozenset", "type", "isinstance", "issubclass", "id",
    "hash", "repr", "abs", "min", "max", "sum", "all", "any", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "open", "input", "getattr",
    "setattr", "hasattr", "delattr", "vars", "dir", "callable", "super",
    "property", "staticmethod", "classmethod", "object", "iter", "next",
    "slice", "round", "pow", "divmod", "hex", "oct", "bin", "chr", "ord",
    "format", "ascii", "eval", "exec", "compile", "globals", "locals",
    "memoryview", "bytearray", "bytes", "complex", "help", "exit", "quit",
}

GO_BUILTINS: Set[str] = {
    "append", "cap", "close", "complex", "copy", "delete", "imag", "len",
    "make", "new", "panic", "print", "println", "real", "recover",
}

RUST_MACROS: Set[str] = {
    "println", "print", "eprintln", "eprint", "format", "write", "writeln",
    "vec", "panic", "assert", "assert_eq", "assert_ne", "debug_assert",
    "debug_assert_eq", "debug_assert_ne", "todo", "unimplemented", "unreachable",
    "cfg", "env", "option_env", "concat", "stringify", "include", "include_str",
    "include_bytes", "module_path", "file", "line", "column",
}

JS_TS_BUILTINS: Set[str] = {
    "console", "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURI",
    "decodeURI", "encodeURIComponent", "decodeURIComponent", "eval",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "fetch", "require", "import", "alert", "confirm", "prompt",
}


class CallGraphAnalyzer:
    """
    Multi-language call graph analyzer using tree-sitter.

    Extracts function calls from source code and builds caller/callee relationships.
    Works with existing extractors to resolve qualified names.

    Features:
        - Multi-language support: Python, Go, Rust, TypeScript, JavaScript
        - Call type detection: function, method, constructor, macro, builtin
        - Receiver extraction for method calls
        - Async call detection (await expressions)
        - Chained call detection
        - Argument counting

    Example:
        >>> from tree_sitter_language_pack import get_parser
        >>> parser = get_parser("python")
        >>> source = b'''
        ... def foo():
        ...     bar()
        ...     obj.method()
        ... '''
        >>> tree = parser.parse(source)
        >>> analyzer = CallGraphAnalyzer()
        >>> calls = analyzer.analyze_file(tree, source, "test.py", "python", [])
        >>> len(calls.calls)
        2
    """

    # Language-specific call node types
    CALL_NODE_TYPES: Dict[str, Set[str]] = {
        "python": {"call"},
        "go": {"call_expression"},
        "rust": {"call_expression", "method_call_expression", "macro_invocation"},
        "typescript": {"call_expression", "new_expression"},
        "javascript": {"call_expression", "new_expression"},
    }

    # Function definition node types (boundaries - don't recurse into these)
    FUNCTION_BOUNDARIES: Dict[str, Set[str]] = {
        "python": {"function_definition", "lambda"},
        "go": {"function_declaration", "method_declaration", "func_literal"},
        "rust": {"function_item", "closure_expression"},
        "typescript": {
            "function_declaration", "arrow_function", "method_definition",
            "function_expression", "generator_function_declaration",
        },
        "javascript": {
            "function_declaration", "arrow_function", "method_definition",
            "function_expression", "generator_function_declaration",
        },
    }

    # Builtins by language
    BUILTIN_FUNCTIONS: Dict[str, Set[str]] = {
        "python": PYTHON_BUILTINS,
        "go": GO_BUILTINS,
        "rust": RUST_MACROS,
        "typescript": JS_TS_BUILTINS,
        "javascript": JS_TS_BUILTINS,
    }

    def __init__(self) -> None:
        """Initialize the call graph analyzer."""
        self._log = logger.bind(analyzer="CallGraphAnalyzer")
        self._log.debug("CallGraphAnalyzer initialized")

    # =========================================================================
    # Main Analysis Methods
    # =========================================================================

    def analyze_file(
        self,
        tree: Tree,
        source: bytes,
        file_path: str,
        language: str,
        extracted_functions: List[ExtractedCode],
    ) -> CallGraph:
        """
        Analyze all calls in a file.

        Iterates through all extracted functions and methods, finding
        call sites within each function's body.

        Args:
            tree: Parsed tree-sitter AST
            source: Source code bytes
            file_path: Path to source file
            language: Programming language (python, go, rust, typescript, javascript)
            extracted_functions: Functions already extracted by language-specific extractor

        Returns:
            CallGraph with all detected calls
        """
        self._log.debug(
            "analyzing_file",
            file_path=file_path,
            language=language,
            function_count=len(extracted_functions),
        )

        all_calls: List[FunctionCall] = []

        # Process each extracted function
        for func in extracted_functions:
            # Only analyze functions and methods (not classes, interfaces, etc.)
            if func.element_type not in (CodeElementType.FUNCTION, CodeElementType.METHOD):
                continue

            try:
                calls = self.analyze_function(func, tree, source, language)
                all_calls.extend(calls)
            except Exception as e:
                self._log.warning(
                    "function_analysis_failed",
                    function=func.qualified_name,
                    error=str(e),
                )

        # Sort calls by location
        all_calls.sort(key=lambda c: (c.location.start_line, c.location.start_column))

        self._log.debug(
            "file_analysis_complete",
            file_path=file_path,
            total_calls=len(all_calls),
        )

        return CallGraph(
            file_path=file_path,
            language=language,
            calls=all_calls,
        )

    def analyze_function(
        self,
        func: ExtractedCode,
        tree: Tree,
        source: bytes,
        language: str,
    ) -> List[FunctionCall]:
        """
        Extract all calls made within a single function.

        Finds the function node in the AST by matching location,
        then extracts all call expressions from its body.

        Args:
            func: The extracted function to analyze
            tree: Parsed tree-sitter AST
            source: Source code bytes
            language: Programming language

        Returns:
            List of FunctionCall objects found in the function body
        """
        self._log.debug(
            "analyzing_function",
            function=func.qualified_name,
            location=f"{func.location.start_line}:{func.location.start_column}",
        )

        # Find the function node in the AST by location
        func_node = self._find_function_node(tree.root_node, func.location, language)
        if not func_node:
            self._log.warning(
                "function_node_not_found",
                function=func.qualified_name,
                start_line=func.location.start_line,
            )
            return []

        # Get the function body
        body_node = self._get_function_body(func_node, language)
        if not body_node:
            # Some functions may not have a body (e.g., abstract methods, declarations)
            return []

        # Extract all calls from the body
        calls = self._extract_calls_from_node(
            node=body_node,
            source=source,
            language=language,
            caller_qualified_name=func.qualified_name,
            caller_file_path=func.file_path,
        )

        self._log.debug(
            "function_analysis_complete",
            function=func.qualified_name,
            calls_found=len(calls),
        )

        return calls

    # =========================================================================
    # Node Finding Methods
    # =========================================================================

    def _find_function_node(
        self,
        root: Node,
        location: ASTNodeLocation,
        language: str,
    ) -> Optional[Node]:
        """
        Find a function node in the AST by its location.

        Searches for a function node that matches the given location's
        start position (line and column).

        Args:
            root: Root node of the AST
            location: Location to search for
            language: Programming language

        Returns:
            The matching function node, or None if not found
        """
        boundaries = self.FUNCTION_BOUNDARIES.get(language, set())

        def find_node(node: Node) -> Optional[Node]:
            """Recursively search for matching function node."""
            # Check if this node matches
            if node.type in boundaries:
                if (node.start_point[0] == location.start_line and
                    node.start_point[1] == location.start_column):
                    return node

            # Handle decorated definitions in Python
            if language == "python" and node.type == "decorated_definition":
                # Check the decorated function inside
                for child in node.children:
                    if child.type == "function_definition":
                        # The decorated_definition starts at the decorator,
                        # but we might have stored the location from the decorator
                        if (node.start_point[0] == location.start_line and
                            node.start_point[1] == location.start_column):
                            return child

            # Handle impl blocks in Rust
            if language == "rust" and node.type == "impl_item":
                for child in node.children:
                    if child.type == "declaration_list":
                        for item in child.children:
                            result = find_node(item)
                            if result:
                                return result

            # Recurse into children
            for child in node.children:
                result = find_node(child)
                if result:
                    return result

            return None

        return find_node(root)

    def _get_function_body(self, func_node: Node, language: str) -> Optional[Node]:
        """
        Get the body node of a function.

        Different languages use different field names for the function body.

        Args:
            func_node: The function node
            language: Programming language

        Returns:
            The body node, or None if not found
        """
        # Try common field names
        body = func_node.child_by_field_name("body")
        if body:
            return body

        # Try block (Go)
        body = func_node.child_by_field_name("block")
        if body:
            return body

        # For methods, the body might be directly inside
        if language in ("typescript", "javascript"):
            for child in func_node.children:
                if child.type == "statement_block":
                    return child

        # For arrow functions, the body can be an expression or block
        if func_node.type == "arrow_function":
            for child in func_node.children:
                if child.type in ("statement_block", "expression"):
                    return child
                # Arrow function with expression body (no braces)
                if child.type not in ("=>", "async", "formal_parameters", "type_annotation"):
                    return child

        return None

    # =========================================================================
    # Call Extraction Methods
    # =========================================================================

    def _extract_calls_from_node(
        self,
        node: Node,
        source: bytes,
        language: str,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> List[FunctionCall]:
        """
        Recursively extract all call expressions from a node.

        Traverses the AST, finding call nodes and parsing them according
        to the language. Stops at function boundaries to avoid extracting
        calls from nested functions.

        Args:
            node: The node to search within
            source: Source code bytes
            language: Programming language
            caller_qualified_name: Qualified name of the calling function
            caller_file_path: File path of the caller

        Returns:
            List of FunctionCall objects
        """
        calls: List[FunctionCall] = []
        call_types = self.CALL_NODE_TYPES.get(language, set())
        boundaries = self.FUNCTION_BOUNDARIES.get(language, set())

        def visit(current_node: Node) -> None:
            """Recursively visit nodes to find calls."""
            # Stop at function boundaries (nested functions)
            if current_node.type in boundaries:
                return

            # Check if this is a call node
            if current_node.type in call_types:
                call = self._parse_call_node(
                    call_node=current_node,
                    source=source,
                    language=language,
                    caller_qualified_name=caller_qualified_name,
                    caller_file_path=caller_file_path,
                )
                if call:
                    calls.append(call)

            # Recurse into children
            for child in current_node.children:
                visit(child)

        visit(node)
        return calls

    def _parse_call_node(
        self,
        call_node: Node,
        source: bytes,
        language: str,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """
        Parse a call node based on the language.

        Dispatches to language-specific parsing methods.

        Args:
            call_node: The call expression node
            source: Source code bytes
            language: Programming language
            caller_qualified_name: Qualified name of the calling function
            caller_file_path: File path of the caller

        Returns:
            FunctionCall object or None if parsing fails
        """
        try:
            if language == "python":
                return self._parse_python_call(
                    call_node, source, caller_qualified_name, caller_file_path
                )
            elif language == "go":
                return self._parse_go_call(
                    call_node, source, caller_qualified_name, caller_file_path
                )
            elif language == "rust":
                return self._parse_rust_call(
                    call_node, source, caller_qualified_name, caller_file_path
                )
            elif language in ("typescript", "javascript"):
                return self._parse_typescript_call(
                    call_node, source, language, caller_qualified_name, caller_file_path
                )
            else:
                self._log.warning("unsupported_language", language=language)
                return None
        except Exception as e:
            self._log.warning(
                "call_parsing_failed",
                node_type=call_node.type,
                error=str(e),
            )
            return None

    # =========================================================================
    # Python Call Parsing
    # =========================================================================

    def _parse_python_call(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """
        Parse a Python call expression.

        Python call structure:
        - call
          - function: identifier (direct call) or attribute (method call)
          - arguments: argument_list

        Args:
            call_node: The call node
            source: Source code bytes
            caller_qualified_name: Qualified name of the caller
            caller_file_path: File path of the caller

        Returns:
            FunctionCall object or None
        """
        # Get the function being called
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        callee_name: str
        receiver: Optional[str] = None
        call_type: CallType = CallType.FUNCTION

        if func_node.type == "identifier":
            # Direct function call: foo()
            callee_name = self._get_node_text(func_node, source)

            # Check if it's a builtin
            if callee_name in self.BUILTIN_FUNCTIONS.get("python", set()):
                call_type = CallType.BUILTIN
            # Check if it looks like a constructor (PascalCase)
            elif callee_name and callee_name[0].isupper():
                call_type = CallType.CONSTRUCTOR

        elif func_node.type == "attribute":
            # Method call: obj.method()
            object_node = func_node.child_by_field_name("object")
            attribute_node = func_node.child_by_field_name("attribute")

            if not attribute_node:
                return None

            callee_name = self._get_node_text(attribute_node, source)
            receiver = self._get_node_text(object_node, source) if object_node else None
            call_type = CallType.METHOD

        elif func_node.type == "subscript":
            # Call on subscript: items[0]()
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        else:
            # Other callable (e.g., call result, parenthesized expression)
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        # Count arguments
        args_count = self._count_python_arguments(call_node)

        # Check for await
        is_await = self._is_await_expression(call_node)

        # Check for chaining
        is_chained = self._is_chained_call(call_node, source)

        # Create location
        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,  # Resolution happens later
            location=location,
            call_type=call_type,
            receiver=receiver,
            arguments_count=args_count,
            is_await=is_await,
            is_chained=is_chained,
            is_resolved=False,
            is_external=False,
        )

    def _count_python_arguments(self, call_node: Node) -> int:
        """Count the number of arguments in a Python call."""
        args_node = call_node.child_by_field_name("arguments")
        if not args_node:
            return 0

        count = 0
        for child in args_node.children:
            # Skip parentheses and commas
            if child.type not in ("(", ")", ","):
                count += 1
        return count

    # =========================================================================
    # Go Call Parsing
    # =========================================================================

    def _parse_go_call(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """
        Parse a Go call expression.

        Go call structure:
        - call_expression
          - function: identifier (direct) or selector_expression (method/package)
          - arguments: argument_list

        Args:
            call_node: The call_expression node
            source: Source code bytes
            caller_qualified_name: Qualified name of the caller
            caller_file_path: File path of the caller

        Returns:
            FunctionCall object or None
        """
        # Get the function being called
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        callee_name: str
        receiver: Optional[str] = None
        call_type: CallType = CallType.FUNCTION

        if func_node.type == "identifier":
            # Direct function call: foo()
            callee_name = self._get_node_text(func_node, source)

            # Check if it's a builtin
            if callee_name in self.BUILTIN_FUNCTIONS.get("go", set()):
                call_type = CallType.BUILTIN
            # Check if it looks like a type constructor (PascalCase)
            elif callee_name and callee_name[0].isupper():
                call_type = CallType.CONSTRUCTOR

        elif func_node.type == "selector_expression":
            # Method or package call: obj.Method() or pkg.Func()
            operand_node = func_node.child_by_field_name("operand")
            field_node = func_node.child_by_field_name("field")

            if not field_node:
                return None

            callee_name = self._get_node_text(field_node, source)
            receiver = self._get_node_text(operand_node, source) if operand_node else None

            # Heuristic: lowercase receiver is likely a variable (method call)
            # uppercase receiver is likely a package (static/package call)
            if receiver and receiver[0].islower():
                call_type = CallType.METHOD
            else:
                call_type = CallType.STATIC

        elif func_node.type == "type_conversion_expression":
            # Type conversion: int(x)
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.CONSTRUCTOR

        elif func_node.type == "parenthesized_expression":
            # Call on parenthesized expression
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        else:
            # Other callable (func literal, index expression, etc.)
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        # Count arguments
        args_count = self._count_go_arguments(call_node)

        # Check for chaining
        is_chained = self._is_chained_call(call_node, source)

        # Create location
        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,
            location=location,
            call_type=call_type,
            receiver=receiver,
            arguments_count=args_count,
            is_await=False,  # Go doesn't have await
            is_chained=is_chained,
            is_resolved=False,
            is_external=False,
        )

    def _count_go_arguments(self, call_node: Node) -> int:
        """Count the number of arguments in a Go call."""
        args_node = call_node.child_by_field_name("arguments")
        if not args_node:
            return 0

        count = 0
        for child in args_node.children:
            if child.type not in ("(", ")", ","):
                count += 1
        return count

    # =========================================================================
    # Rust Call Parsing
    # =========================================================================

    def _parse_rust_call(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """
        Parse a Rust call expression.

        Rust has three types of calls:
        - call_expression: func(args)
        - method_call_expression: value.method(args)
        - macro_invocation: macro!(args)

        Args:
            call_node: The call node
            source: Source code bytes
            caller_qualified_name: Qualified name of the caller
            caller_file_path: File path of the caller

        Returns:
            FunctionCall object or None
        """
        if call_node.type == "call_expression":
            return self._parse_rust_call_expression(
                call_node, source, caller_qualified_name, caller_file_path
            )
        elif call_node.type == "method_call_expression":
            return self._parse_rust_method_call(
                call_node, source, caller_qualified_name, caller_file_path
            )
        elif call_node.type == "macro_invocation":
            return self._parse_rust_macro_invocation(
                call_node, source, caller_qualified_name, caller_file_path
            )
        else:
            return None

    def _parse_rust_call_expression(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """Parse a Rust call_expression (func(args))."""
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        callee_name: str
        receiver: Optional[str] = None
        call_type: CallType = CallType.FUNCTION

        if func_node.type == "identifier":
            # Direct function call
            callee_name = self._get_node_text(func_node, source)

            # Check if it looks like a constructor (PascalCase)
            if callee_name and callee_name[0].isupper():
                call_type = CallType.CONSTRUCTOR

        elif func_node.type == "scoped_identifier":
            # Scoped call: Type::func() or module::func()
            # Find the last identifier as the function name
            name_node = None
            path_parts = []
            for child in func_node.children:
                if child.type == "identifier":
                    if name_node is not None:
                        path_parts.append(self._get_node_text(name_node, source))
                    name_node = child

            callee_name = self._get_node_text(name_node, source) if name_node else ""
            receiver = "::".join(path_parts) if path_parts else None
            call_type = CallType.STATIC

        elif func_node.type == "field_expression":
            # Method-like call: obj.method()
            # In Rust, this appears as field_expression when the method isn't a method_call_expression
            field_node = None
            value_node = None
            for child in func_node.children:
                if child.type == "field_identifier":
                    field_node = child
                elif child.type not in (".", ):
                    if value_node is None:
                        value_node = child

            if field_node:
                callee_name = self._get_node_text(field_node, source)
                receiver = self._get_node_text(value_node, source) if value_node else None
                call_type = CallType.METHOD
            else:
                callee_name = self._get_node_text(func_node, source)
                call_type = CallType.FUNCTION

        elif func_node.type == "generic_function":
            # Generic function call: func::<T>()
            func_name_node = func_node.child_by_field_name("function")
            if func_name_node:
                callee_name = self._get_node_text(func_name_node, source)
            else:
                callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        else:
            # Other callable
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        # Count arguments
        args_count = self._count_rust_arguments(call_node)

        # Check for await (.await is postfix in Rust)
        is_await = self._is_rust_await_expression(call_node)

        # Check for chaining
        is_chained = self._is_chained_call(call_node, source)

        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,
            location=location,
            call_type=call_type,
            receiver=receiver,
            arguments_count=args_count,
            is_await=is_await,
            is_chained=is_chained,
            is_resolved=False,
            is_external=False,
        )

    def _parse_rust_method_call(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """Parse a Rust method_call_expression (value.method(args))."""
        # Get the method name
        name_node = call_node.child_by_field_name("name")
        if not name_node:
            return None

        callee_name = self._get_node_text(name_node, source)

        # Get the receiver (value)
        value_node = call_node.child_by_field_name("value")
        receiver = self._get_node_text(value_node, source) if value_node else None

        # Count arguments
        args_count = self._count_rust_arguments(call_node)

        # Check for await
        is_await = self._is_rust_await_expression(call_node)

        # Check for chaining
        is_chained = self._is_chained_call(call_node, source)

        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,
            location=location,
            call_type=CallType.METHOD,
            receiver=receiver,
            arguments_count=args_count,
            is_await=is_await,
            is_chained=is_chained,
            is_resolved=False,
            is_external=False,
        )

    def _parse_rust_macro_invocation(
        self,
        call_node: Node,
        source: bytes,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """Parse a Rust macro_invocation (macro!(args))."""
        # Get the macro name - it's the first identifier child
        macro_node = None
        for child in call_node.children:
            if child.type == "identifier":
                macro_node = child
                break
            elif child.type == "scoped_identifier":
                # For qualified macros like std::println!
                macro_node = child
                break

        if not macro_node:
            return None

        callee_name = self._get_node_text(macro_node, source)
        # Remove trailing ! if present (shouldn't be there, ! is separate)
        if callee_name.endswith("!"):
            callee_name = callee_name[:-1]

        # Add ! for display purposes
        callee_name_display = callee_name + "!"

        # Check if it's a standard macro
        is_builtin = callee_name in self.BUILTIN_FUNCTIONS.get("rust", set())

        # Count arguments in token_tree
        args_count = self._count_rust_macro_arguments(call_node)

        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name_display,
            callee_qualified_name=None,
            location=location,
            call_type=CallType.MACRO,
            receiver=None,
            arguments_count=args_count,
            is_await=False,
            is_chained=False,
            is_resolved=False,
            is_external=is_builtin,
        )

    def _count_rust_macro_arguments(self, call_node: Node) -> int:
        """Count the number of arguments in a Rust macro invocation."""
        # Find token_tree child
        for child in call_node.children:
            if child.type == "token_tree":
                # Count comma-separated items (rough estimate)
                text = self._get_node_text(child, child.text or b"")
                if text.startswith("(") and text.endswith(")"):
                    text = text[1:-1]
                if not text.strip():
                    return 0
                # Count top-level commas
                return text.count(",") + 1 if text.strip() else 0
        return 0

    def _count_rust_arguments(self, call_node: Node) -> int:
        """Count the number of arguments in a Rust call."""
        args_node = call_node.child_by_field_name("arguments")
        if not args_node:
            return 0

        count = 0
        for child in args_node.children:
            if child.type not in ("(", ")", ","):
                count += 1
        return count

    def _is_rust_await_expression(self, node: Node) -> bool:
        """Check if a Rust node is followed by .await."""
        # In Rust, await is postfix: expr.await
        parent = node.parent
        if parent and parent.type == "await_expression":
            return True
        # Also check if the node itself has an await field
        if node.type == "await_expression":
            return True
        return False

    # =========================================================================
    # TypeScript/JavaScript Call Parsing
    # =========================================================================

    def _parse_typescript_call(
        self,
        call_node: Node,
        source: bytes,
        language: str,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """
        Parse a TypeScript/JavaScript call expression.

        TypeScript/JS call types:
        - call_expression: func(args), obj.method(args)
        - new_expression: new Class(args)

        Args:
            call_node: The call node
            source: Source code bytes
            language: "typescript" or "javascript"
            caller_qualified_name: Qualified name of the caller
            caller_file_path: File path of the caller

        Returns:
            FunctionCall object or None
        """
        if call_node.type == "new_expression":
            return self._parse_typescript_new_expression(
                call_node, source, language, caller_qualified_name, caller_file_path
            )
        else:
            return self._parse_typescript_call_expression(
                call_node, source, language, caller_qualified_name, caller_file_path
            )

    def _parse_typescript_call_expression(
        self,
        call_node: Node,
        source: bytes,
        language: str,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """Parse a TypeScript/JS call_expression."""
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        callee_name: str
        receiver: Optional[str] = None
        call_type: CallType = CallType.FUNCTION

        if func_node.type == "identifier":
            # Direct function call
            callee_name = self._get_node_text(func_node, source)

            # Check if it's a builtin
            if callee_name in self.BUILTIN_FUNCTIONS.get(language, set()):
                call_type = CallType.BUILTIN
            # Check if it looks like a constructor (PascalCase)
            elif callee_name and callee_name[0].isupper():
                call_type = CallType.CONSTRUCTOR

        elif func_node.type == "member_expression":
            # Method call: obj.method()
            object_node = func_node.child_by_field_name("object")
            property_node = func_node.child_by_field_name("property")

            if not property_node:
                return None

            callee_name = self._get_node_text(property_node, source)
            receiver = self._get_node_text(object_node, source) if object_node else None
            call_type = CallType.METHOD

            # Check for console.log and similar builtins
            if receiver == "console":
                call_type = CallType.BUILTIN

        elif func_node.type == "subscript_expression":
            # Dynamic call: obj[key]()
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        elif func_node.type == "call_expression":
            # Chained call: func()()
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        elif func_node.type == "parenthesized_expression":
            # Call on parenthesized expression
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        else:
            # Other callable (arrow function, etc.)
            callee_name = self._get_node_text(func_node, source)
            call_type = CallType.FUNCTION

        # Count arguments
        args_count = self._count_typescript_arguments(call_node)

        # Check for await
        is_await = self._is_await_expression(call_node)

        # Check for chaining
        is_chained = self._is_chained_call(call_node, source)

        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,
            location=location,
            call_type=call_type,
            receiver=receiver,
            arguments_count=args_count,
            is_await=is_await,
            is_chained=is_chained,
            is_resolved=False,
            is_external=False,
        )

    def _parse_typescript_new_expression(
        self,
        call_node: Node,
        source: bytes,
        language: str,
        caller_qualified_name: str,
        caller_file_path: str,
    ) -> Optional[FunctionCall]:
        """Parse a TypeScript/JS new_expression (new Class(args))."""
        # Get the constructor
        constructor_node = call_node.child_by_field_name("constructor")
        if not constructor_node:
            # Try to find identifier or member_expression child
            for child in call_node.children:
                if child.type in ("identifier", "member_expression"):
                    constructor_node = child
                    break

        if not constructor_node:
            return None

        callee_name: str
        receiver: Optional[str] = None

        if constructor_node.type == "identifier":
            callee_name = self._get_node_text(constructor_node, source)
        elif constructor_node.type == "member_expression":
            object_node = constructor_node.child_by_field_name("object")
            property_node = constructor_node.child_by_field_name("property")
            callee_name = self._get_node_text(property_node, source) if property_node else ""
            receiver = self._get_node_text(object_node, source) if object_node else None
        else:
            callee_name = self._get_node_text(constructor_node, source)

        # Count arguments
        args_count = self._count_typescript_arguments(call_node)

        location = self._create_location(call_node)

        return FunctionCall(
            caller_qualified_name=caller_qualified_name,
            caller_file_path=caller_file_path,
            callee_name=callee_name,
            callee_qualified_name=None,
            location=location,
            call_type=CallType.CONSTRUCTOR,
            receiver=receiver,
            arguments_count=args_count,
            is_await=False,  # new expressions are not awaited directly
            is_chained=False,
            is_resolved=False,
            is_external=False,
        )

    def _count_typescript_arguments(self, call_node: Node) -> int:
        """Count the number of arguments in a TypeScript/JS call."""
        args_node = call_node.child_by_field_name("arguments")
        if not args_node:
            return 0

        count = 0
        for child in args_node.children:
            if child.type not in ("(", ")", ","):
                count += 1
        return count

    # =========================================================================
    # Common Helper Methods
    # =========================================================================

    def _is_await_expression(self, node: Node) -> bool:
        """Check if a call is inside an await expression."""
        parent = node.parent
        if parent and parent.type in ("await_expression", "await"):
            return True
        return False

    def _is_chained_call(self, node: Node, source: bytes) -> bool:
        """
        Check if this call is part of a call chain.

        A chained call is when the result of one call is immediately
        used to call another method (e.g., foo().bar()).
        """
        parent = node.parent

        # Check if parent is a member_expression or attribute (method chain)
        if parent and parent.type in ("member_expression", "attribute", "selector_expression", "field_expression"):
            return True

        # Check if this call's function is another call
        func_field = node.child_by_field_name("function")
        if func_field and func_field.type in ("call_expression", "call", "method_call_expression"):
            return True

        return False

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """Extract text from a tree-sitter node."""
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _create_location(self, node: Node) -> ASTNodeLocation:
        """Create ASTNodeLocation from tree-sitter node."""
        return ASTNodeLocation(
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CallGraphAnalyzer",
    "PYTHON_BUILTINS",
    "GO_BUILTINS",
    "RUST_MACROS",
    "JS_TS_BUILTINS",
]
