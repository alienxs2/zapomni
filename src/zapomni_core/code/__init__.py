"""
Code analysis module for Zapomni Phase 3.

Provides tools for parsing, analyzing, and understanding Python code structure:
- FunctionExtractor: Extract detailed function/method metadata from source code
- CallGraphAnalyzer: Analyze function call relationships and dependencies
- ClassHierarchyBuilder: Extract class definitions and inheritance hierarchies
- ASTCodeChunker: Chunk code at AST level (functions, classes, modules)
- CodeRepositoryIndexer: Index and analyze code repositories
- Future modules: code quality metrics, dependency analysis, etc.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.code.ast_chunker import ASTCodeChunker, CodeMetadata, SupportedLanguage
from zapomni_core.code.call_graph_analyzer import (
    CallGraphAnalyzer,
    FunctionCall,
    FunctionDef,
    ImportMapping,
)
from zapomni_core.code.class_hierarchy_builder import (
    AttributeInfo,
    ClassHierarchyBuilder,
    ClassInfo,
    HierarchyNode,
    MethodInfo,
)
from zapomni_core.code.function_extractor import (
    FunctionExtractor,
    FunctionMetadata,
    Parameter,
)
from zapomni_core.code.repository_indexer import CodeRepositoryIndexer

__all__ = [
    "FunctionExtractor",
    "FunctionMetadata",
    "Parameter",
    "CallGraphAnalyzer",
    "FunctionCall",
    "FunctionDef",
    "ImportMapping",
    "ClassHierarchyBuilder",
    "ClassInfo",
    "MethodInfo",
    "AttributeInfo",
    "HierarchyNode",
    "CodeRepositoryIndexer",
    "ASTCodeChunker",
    "CodeMetadata",
    "SupportedLanguage",
]
