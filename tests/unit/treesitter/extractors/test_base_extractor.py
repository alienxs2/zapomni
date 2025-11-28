"""Unit tests for zapomni_core.treesitter.extractors.base module."""

import pytest
from typing import List
from unittest.mock import MagicMock

from tree_sitter import Tree

from zapomni_core.treesitter.extractors.base import BaseCodeExtractor
from zapomni_core.treesitter.models import (
    ASTNodeLocation,
    CodeElementType,
    ExtractedCode,
)


class ConcreteExtractor(BaseCodeExtractor):
    """Concrete implementation of BaseCodeExtractor for testing."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def supported_node_types(self) -> tuple[str, ...]:
        return ("function_definition", "class_definition")

    def extract_functions(
        self, tree: Tree, source: bytes, file_path: str
    ) -> List[ExtractedCode]:
        """Return a mock function for testing."""
        location = ASTNodeLocation(
            start_line=0,
            end_line=1,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=20,
        )
        return [
            ExtractedCode(
                name="test_func",
                qualified_name="test_func",
                element_type=CodeElementType.FUNCTION,
                language="python",
                file_path=file_path,
                location=location,
                source_code="def test_func(): pass",
            )
        ]

    def extract_classes(
        self, tree: Tree, source: bytes, file_path: str
    ) -> List[ExtractedCode]:
        """Return a mock class for testing."""
        location = ASTNodeLocation(
            start_line=5,
            end_line=10,
            start_column=0,
            end_column=10,
            start_byte=50,
            end_byte=100,
        )
        return [
            ExtractedCode(
                name="TestClass",
                qualified_name="TestClass",
                element_type=CodeElementType.CLASS,
                language="python",
                file_path=file_path,
                location=location,
                source_code="class TestClass: pass",
            )
        ]


class TestBaseCodeExtractorProperties:
    """Tests for BaseCodeExtractor abstract properties."""

    def test_language_name(self):
        """Test language_name property."""
        extractor = ConcreteExtractor()
        assert extractor.language_name == "python"

    def test_supported_node_types(self):
        """Test supported_node_types property."""
        extractor = ConcreteExtractor()
        assert "function_definition" in extractor.supported_node_types
        assert "class_definition" in extractor.supported_node_types


class TestExtractAll:
    """Tests for extract_all method."""

    def test_extract_all_combines_functions_and_classes(self, python_tree, python_source_code):
        """Test that extract_all combines functions and classes."""
        extractor = ConcreteExtractor()
        results = extractor.extract_all(python_tree, python_source_code, "/test.py")

        assert len(results) == 2
        names = [r.name for r in results]
        assert "test_func" in names
        assert "TestClass" in names

    def test_extract_all_sorted_by_start_line(self, python_tree, python_source_code):
        """Test that extract_all returns results sorted by start_line."""
        extractor = ConcreteExtractor()
        results = extractor.extract_all(python_tree, python_source_code, "/test.py")

        # Results should be sorted by start_line
        for i in range(len(results) - 1):
            assert results[i].location.start_line <= results[i + 1].location.start_line


class TestHelperMethods:
    """Tests for helper methods in BaseCodeExtractor."""

    def test_get_node_text(self, python_tree, python_source_code):
        """Test _get_node_text extracts correct text."""
        extractor = ConcreteExtractor()
        root = python_tree.root_node

        # Get text of first child (should be a function or newline)
        for child in root.children:
            if child.type == "function_definition":
                text = extractor._get_node_text(child, python_source_code)
                assert "def" in text
                break

    def test_get_node_text_handles_encoding(self):
        """Test _get_node_text handles UTF-8 encoding."""
        extractor = ConcreteExtractor()

        # Create mock node
        mock_node = MagicMock()
        mock_node.start_byte = 0
        mock_node.end_byte = 5

        source = "hello".encode('utf-8')
        text = extractor._get_node_text(mock_node, source)
        assert text == "hello"

    def test_create_location(self, python_tree):
        """Test _create_location creates correct ASTNodeLocation."""
        extractor = ConcreteExtractor()
        node = python_tree.root_node

        location = extractor._create_location(node)

        assert isinstance(location, ASTNodeLocation)
        assert location.start_line == node.start_point[0]
        assert location.end_line == node.end_point[0]
        assert location.start_byte == node.start_byte
        assert location.end_byte == node.end_byte

    def test_is_private_name_single_underscore(self):
        """Test _is_private_name returns True for single underscore prefix."""
        extractor = ConcreteExtractor()
        assert extractor._is_private_name("_private") is True
        assert extractor._is_private_name("_internal_method") is True

    def test_is_private_name_double_underscore(self):
        """Test _is_private_name returns False for dunder methods."""
        extractor = ConcreteExtractor()
        assert extractor._is_private_name("__init__") is False
        assert extractor._is_private_name("__str__") is False

    def test_is_private_name_public(self):
        """Test _is_private_name returns False for public names."""
        extractor = ConcreteExtractor()
        assert extractor._is_private_name("public") is False
        assert extractor._is_private_name("method") is False


class TestAbstractMethods:
    """Tests to verify abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseCodeExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCodeExtractor()

    def test_must_implement_language_name(self):
        """Test that language_name must be implemented."""

        class IncompleteExtractor(BaseCodeExtractor):
            @property
            def supported_node_types(self) -> tuple[str, ...]:
                return ("function",)

            def extract_functions(self, tree, source, file_path):
                return []

            def extract_classes(self, tree, source, file_path):
                return []

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_must_implement_supported_node_types(self):
        """Test that supported_node_types must be implemented."""

        class IncompleteExtractor(BaseCodeExtractor):
            @property
            def language_name(self) -> str:
                return "test"

            def extract_functions(self, tree, source, file_path):
                return []

            def extract_classes(self, tree, source, file_path):
                return []

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_must_implement_extract_functions(self):
        """Test that extract_functions must be implemented."""

        class IncompleteExtractor(BaseCodeExtractor):
            @property
            def language_name(self) -> str:
                return "test"

            @property
            def supported_node_types(self) -> tuple[str, ...]:
                return ("function",)

            def extract_classes(self, tree, source, file_path):
                return []

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_must_implement_extract_classes(self):
        """Test that extract_classes must be implemented."""

        class IncompleteExtractor(BaseCodeExtractor):
            @property
            def language_name(self) -> str:
                return "test"

            @property
            def supported_node_types(self) -> tuple[str, ...]:
                return ("function",)

            def extract_functions(self, tree, source, file_path):
                return []

        with pytest.raises(TypeError):
            IncompleteExtractor()
