"""Code element extractors for different programming languages.

This module provides extractors that analyze AST trees and extract
structured code elements (functions, classes, methods, etc.).

Classes:
    BaseCodeExtractor: Abstract base class for all extractors.
    GenericExtractor: Universal fallback extractor for any language.
    PythonExtractor: Specialized extractor for Python with full AST support.

Constants:
    FUNCTION_NODE_TYPES: Set of AST node types for function definitions.
    CLASS_NODE_TYPES: Set of AST node types for class definitions.
    NAME_NODE_TYPES: Set of AST node types for identifiers/names.
    SPECIAL_DECORATORS: Python decorators that affect function behavior.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from .base import BaseCodeExtractor
from .generic import (
    CLASS_NODE_TYPES,
    FUNCTION_NODE_TYPES,
    NAME_NODE_TYPES,
    GenericExtractor,
)
from .python import PythonExtractor, SPECIAL_DECORATORS

__all__ = [
    # Base class
    "BaseCodeExtractor",
    # Generic extractor
    "GenericExtractor",
    # Python extractor
    "PythonExtractor",
    # Node type sets
    "FUNCTION_NODE_TYPES",
    "CLASS_NODE_TYPES",
    "NAME_NODE_TYPES",
    # Python constants
    "SPECIAL_DECORATORS",
]
