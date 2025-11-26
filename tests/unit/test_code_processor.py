"""
Unit tests for the CodeProcessor AST utilities.

Exercises function/class/import extraction, syntax validation, and edge cases.
"""

from typing import Dict, List

import pytest

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.processors.code_processor import (
    extract_classes,
    extract_functions,
    extract_imports,
    parse_python,
    validate_syntax,
)


def _find_entry(entries: List[Dict], key: str, value: str) -> Dict:
    """Helper to locate an entry by a given key/value pair."""
    for entry in entries:
        if entry.get(key) == value:
            return entry
    raise AssertionError(f"Entry with {key}={value} not found")


def test_extract_functions_returns_documented_entries():
    """Functions should include args, docstrings, and lineno metadata."""
    code = '''
def add(a, b=1):
    """Add two values."""
    return a + b

async def measure():
    """Awaitable measurement stub."""
    return 42
'''

    functions = extract_functions(code)

    assert len(functions) == 2
    add_info = _find_entry(functions, "name", "add")
    assert add_info["docstring"] == "Add two values."
    assert add_info["args"] == ["a", "b"]
    assert add_info["is_async"] is False
    assert add_info["parent"] is None

    measure_info = _find_entry(functions, "name", "measure")
    assert measure_info["is_async"] is True
    assert measure_info["args"] == []
    assert measure_info["qualified_name"] == "measure"


def test_extract_classes_exposes_methods_and_bases():
    """Class extraction should capture bases and nested methods metadata."""
    code = '''
class Alpha(Base, mixin):
    """Primary class."""

    def method(self, value):
        """Return a value."""
        return value
'''

    classes = extract_classes(code)

    assert len(classes) == 1
    alpha = classes[0]
    assert alpha["name"] == "Alpha"
    assert alpha["bases"] == ["Base", "mixin"]
    assert alpha["docstring"] == "Primary class."
    assert len(alpha["methods"]) == 1
    method = alpha["methods"][0]
    assert method["name"] == "method"
    assert method["qualified_name"] == "Alpha.method"
    assert method["parent"] == "Alpha"
    assert method["args"] == ["self", "value"]


def test_extract_imports_and_parse_python_agree():
    """Import extraction should preserve statement order and parse_python bundles results."""
    code = """
import os
from typing import List as L, Dict
from .sub import helper
"""

    imports = extract_imports(code)
    assert imports == [
        "import os",
        "from typing import List as L, Dict",
        "from .sub import helper",
    ]

    parsed = parse_python(code)
    assert parsed["imports"] == imports
    assert parsed["functions"] == []
    assert parsed["classes"] == []
    assert parsed["import_details"] == [
        {"statement": "import os", "lineno": 2},
        {"statement": "from typing import List as L, Dict", "lineno": 3},
        {"statement": "from .sub import helper", "lineno": 4},
    ]


def test_validate_syntax_and_syntax_error_propagation():
    """Invalid Python should signal False for validation and raise ProcessingError on extraction."""
    invalid_code = "def broken("

    assert validate_syntax(invalid_code) is False
    with pytest.raises(ProcessingError):
        extract_functions(invalid_code)

    with pytest.raises(ProcessingError):
        extract_classes(invalid_code)

    with pytest.raises(ProcessingError):
        extract_imports(invalid_code)


def test_empty_code_returns_empty_collections():
    """Empty source should produce empty summaries without errors."""
    empty_code = ""

    assert extract_functions(empty_code) == []
    assert extract_classes(empty_code) == []
    assert extract_imports(empty_code) == []

    parsed = parse_python(empty_code)
    assert parsed["functions"] == []
    assert parsed["classes"] == []
    assert parsed["imports"] == []
    assert validate_syntax(empty_code) is True


def test_nested_structures_capture_qualified_names():
    """Nested functions and inner classes should expose qualified names for disambiguation."""
    code = '''
def outer(param):
    """Outer doc."""

    def inner(value):
        """Inner doc."""
        return value

    async def inner_async():
        def inner_inner():
            return inner_inner

        return inner_inner

    class InnerClass:
        """Nested class."""

        def method(self):
            """Method doc."""
            return param

    return inner
'''

    functions = extract_functions(code)
    outer_entry = _find_entry(functions, "name", "outer")
    assert outer_entry["qualified_name"] == "outer"

    inner_entry = _find_entry(functions, "qualified_name", "outer.inner")
    assert inner_entry["parent"] == "outer"

    deep_entry = _find_entry(functions, "qualified_name", "outer.inner_async.inner_inner")
    assert deep_entry["parent"] == "inner_async"

    classes = extract_classes(code)
    inner_class = _find_entry(classes, "qualified_name", "outer.InnerClass")
    assert inner_class["parent"] == "outer"
    assert inner_class["methods"][0]["qualified_name"] == "outer.InnerClass.method"

    parsed = parse_python(code)
    assert any(c["qualified_name"] == "outer.InnerClass" for c in parsed["classes"])
    assert any(f["qualified_name"] == "outer.inner" for f in parsed["functions"])


def test_non_string_code_raises_validation_error():
    """Non-string input should raise ValidationError."""
    with pytest.raises(ValidationError):
        extract_functions(None)  # type: ignore

    with pytest.raises(ValidationError):
        extract_classes(123)  # type: ignore

    with pytest.raises(ValidationError):
        extract_imports(False)  # type: ignore
