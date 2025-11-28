"""Code element extractors for different programming languages.

This module provides extractors that analyze AST trees and extract
structured code elements (functions, classes, methods, etc.).

Classes:
    BaseCodeExtractor: Abstract base class for all extractors.
    GenericExtractor: Universal fallback extractor for any language.

Constants:
    FUNCTION_NODE_TYPES: Set of AST node types for function definitions.
    CLASS_NODE_TYPES: Set of AST node types for class definitions.
    NAME_NODE_TYPES: Set of AST node types for identifiers/names.

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

__all__ = [
    # Base class
    "BaseCodeExtractor",
    # Generic extractor
    "GenericExtractor",
    # Node type sets
    "FUNCTION_NODE_TYPES",
    "CLASS_NODE_TYPES",
    "NAME_NODE_TYPES",
]
