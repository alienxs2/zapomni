"""Code element extractors for different programming languages.

This module provides extractors that analyze AST trees and extract
structured code elements (functions, classes, methods, etc.).

Classes:
    BaseCodeExtractor: Abstract base class for all extractors.
    GenericExtractor: Universal fallback extractor for any language.
    PythonExtractor: Specialized extractor for Python with full AST support.
    TypeScriptExtractor: Specialized extractor for TypeScript/JavaScript.
    GoExtractor: Specialized extractor for Go with full AST support.
    RustExtractor: Specialized extractor for Rust with full AST support.

Constants:
    FUNCTION_NODE_TYPES: Set of AST node types for function definitions.
    CLASS_NODE_TYPES: Set of AST node types for class definitions.
    NAME_NODE_TYPES: Set of AST node types for identifiers/names.
    SPECIAL_DECORATORS: Python decorators that affect function behavior.
    TYPESCRIPT_FUNCTION_TYPES: TypeScript function node types.
    TYPESCRIPT_CLASS_TYPES: TypeScript class node types.
    TYPESCRIPT_TYPE_TYPES: TypeScript type definition node types.
    TYPESCRIPT_SPECIAL_DECORATORS: TypeScript special decorators.
    GO_FUNCTION_TYPES: Go function node types.
    GO_TYPE_TYPES: Go type definition node types.
    GO_STRUCT_TYPES: Go struct node types.
    GO_INTERFACE_TYPES: Go interface node types.
    RUST_FUNCTION_TYPES: Rust function node types.
    RUST_STRUCT_TYPES: Rust struct node types.
    RUST_TRAIT_TYPES: Rust trait node types.
    RUST_ENUM_TYPES: Rust enum node types.
    RUST_MACRO_TYPES: Rust macro node types.
    RUST_IMPL_TYPES: Rust impl block node types.

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
from .go import (
    GO_FUNCTION_TYPES,
    GO_INTERFACE_TYPES,
    GO_STRUCT_TYPES,
    GO_TYPE_TYPES,
    GoExtractor,
)
from .python import SPECIAL_DECORATORS, PythonExtractor
from .rust import (
    RUST_ENUM_TYPES,
    RUST_FUNCTION_TYPES,
    RUST_IMPL_TYPES,
    RUST_MACRO_TYPES,
    RUST_STRUCT_TYPES,
    RUST_TRAIT_TYPES,
    RustExtractor,
)
from .typescript import (
    TYPESCRIPT_CLASS_TYPES,
    TYPESCRIPT_FUNCTION_TYPES,
    TYPESCRIPT_SPECIAL_DECORATORS,
    TYPESCRIPT_TYPE_TYPES,
    TypeScriptExtractor,
)

__all__ = [
    # Base class
    "BaseCodeExtractor",
    # Generic extractor
    "GenericExtractor",
    # Python extractor
    "PythonExtractor",
    # TypeScript extractor
    "TypeScriptExtractor",
    # Go extractor
    "GoExtractor",
    # Rust extractor
    "RustExtractor",
    # Node type sets
    "FUNCTION_NODE_TYPES",
    "CLASS_NODE_TYPES",
    "NAME_NODE_TYPES",
    # Python constants
    "SPECIAL_DECORATORS",
    # TypeScript constants
    "TYPESCRIPT_FUNCTION_TYPES",
    "TYPESCRIPT_CLASS_TYPES",
    "TYPESCRIPT_TYPE_TYPES",
    "TYPESCRIPT_SPECIAL_DECORATORS",
    # Go constants
    "GO_FUNCTION_TYPES",
    "GO_TYPE_TYPES",
    "GO_STRUCT_TYPES",
    "GO_INTERFACE_TYPES",
    # Rust constants
    "RUST_FUNCTION_TYPES",
    "RUST_STRUCT_TYPES",
    "RUST_TRAIT_TYPES",
    "RUST_ENUM_TYPES",
    "RUST_MACRO_TYPES",
    "RUST_IMPL_TYPES",
]
