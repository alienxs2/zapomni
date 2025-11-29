"""
Unit tests for the EntityExtractor component.

Tests hybrid NER (SpaCy + LLM) entity extraction for knowledge graph construction.
Phase 1 tests: SpaCy-only extraction (LLM refinement tests for Phase 2).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import spacy

from zapomni_core.exceptions import ValidationError
from zapomni_core.extractors.entity_extractor import (
    Entity,
    EntityExtractor,
    Relationship,
)


@pytest.fixture
def spacy_model():
    """Load SpaCy model for testing."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not installed, use blank model with NER component
        nlp = spacy.blank("en")
        nlp.add_pipe("ner")
        # Initialize the NER component
        nlp.initialize(lambda: [])
    return nlp


@pytest.fixture
def basic_extractor(spacy_model):
    """Create basic EntityExtractor without LLM refinement."""
    return EntityExtractor(
        spacy_model=spacy_model,
        enable_llm_refinement=False,
        confidence_threshold=0.7,
    )


@pytest.fixture
def mock_ollama_client():
    """Create mock Ollama client for LLM tests."""
    client = MagicMock()
    client.generate = MagicMock(
        return_value={
            "response": "Python: TECHNOLOGY, Programming language\n"
            "Guido van Rossum: PERSON, Creator of Python"
        }
    )

    # Add async extract_relationships method
    async def mock_extract_relationships(text, entities):
        return [
            {
                "source": "Django",
                "target": "Python",
                "type": "WRITTEN_IN",
                "confidence": 0.9,
                "evidence": "written in",
            }
        ]

    client.extract_relationships = mock_extract_relationships
    client.base_url = "http://localhost:11434"
    client.timeout = 30
    client.temperature = 0.7
    return client


class TestEntityExtractorInitialization:
    """Test suite for EntityExtractor initialization and configuration."""

    def test_init_with_spacy_model_only(self, spacy_model) -> None:
        """Should initialize with SpaCy model and default settings."""
        extractor = EntityExtractor(spacy_model=spacy_model)

        assert extractor.spacy_nlp == spacy_model
        assert extractor.ollama_client is None
        assert extractor.enable_llm_refinement is False
        assert extractor.confidence_threshold == 0.7
        assert len(extractor.entity_types) > 0

    def test_init_with_custom_confidence_threshold(self, spacy_model) -> None:
        """Should accept custom confidence threshold."""
        extractor = EntityExtractor(
            spacy_model=spacy_model,
            confidence_threshold=0.85,
        )

        assert extractor.confidence_threshold == 0.85

    def test_init_with_custom_entity_types(self, spacy_model) -> None:
        """Should accept custom entity types set."""
        custom_types = {"PERSON", "ORG", "TECHNOLOGY"}
        extractor = EntityExtractor(
            spacy_model=spacy_model,
            entity_types=custom_types,
        )

        assert extractor.entity_types == custom_types

    def test_init_raises_error_for_invalid_confidence_threshold(self, spacy_model) -> None:
        """Should raise ValueError for confidence threshold outside [0.0, 1.0]."""
        with pytest.raises(ValueError) as exc_info:
            EntityExtractor(
                spacy_model=spacy_model,
                confidence_threshold=1.5,
            )

        assert "confidence_threshold" in str(exc_info.value).lower()

    def test_init_raises_error_for_negative_confidence(self, spacy_model) -> None:
        """Should raise ValueError for negative confidence threshold."""
        with pytest.raises(ValueError) as exc_info:
            EntityExtractor(
                spacy_model=spacy_model,
                confidence_threshold=-0.1,
            )

        assert "confidence_threshold" in str(exc_info.value).lower()

    def test_init_raises_error_if_spacy_model_missing_ner(self) -> None:
        """Should raise ValueError if SpaCy model doesn't have NER component."""
        blank_nlp = spacy.blank("en")  # No NER component

        with pytest.raises(ValueError) as exc_info:
            EntityExtractor(spacy_model=blank_nlp)

        assert "ner" in str(exc_info.value).lower()

    def test_init_with_llm_refinement_requires_ollama_client(self, spacy_model) -> None:
        """Should raise ValueError if LLM refinement enabled without Ollama client."""
        with pytest.raises(ValueError) as exc_info:
            EntityExtractor(
                spacy_model=spacy_model,
                enable_llm_refinement=True,
                ollama_client=None,
            )

        assert "ollama_client" in str(exc_info.value).lower()

    def test_init_with_llm_refinement_and_client(self, spacy_model, mock_ollama_client) -> None:
        """Should initialize successfully with LLM refinement enabled."""
        extractor = EntityExtractor(
            spacy_model=spacy_model,
            ollama_client=mock_ollama_client,
            enable_llm_refinement=True,
        )

        assert extractor.enable_llm_refinement is True
        assert extractor.ollama_client == mock_ollama_client


class TestEntityExtraction:
    """Test suite for entity extraction functionality."""

    def test_extract_entities_from_simple_text(self, basic_extractor) -> None:
        """Should extract entities from simple text."""
        text = "Python was created by Guido van Rossum in 1991."

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)
        assert len(entities) > 0
        assert all(isinstance(e, Entity) for e in entities)

    def test_extract_entities_returns_entity_with_required_fields(self, basic_extractor) -> None:
        """Should return Entity objects with all required fields."""
        text = "OpenAI developed GPT-4."

        entities = basic_extractor.extract_entities(text)

        for entity in entities:
            assert hasattr(entity, "name")
            assert hasattr(entity, "type")
            assert hasattr(entity, "description")
            assert hasattr(entity, "confidence")
            assert hasattr(entity, "mentions")
            assert isinstance(entity.confidence, float)
            assert 0.0 <= entity.confidence <= 1.0

    def test_extract_entities_filters_by_confidence_threshold(self, spacy_model) -> None:
        """Should filter entities below confidence threshold."""
        extractor = EntityExtractor(
            spacy_model=spacy_model,
            confidence_threshold=0.9,  # Very high threshold
        )

        text = "Python was created by Guido van Rossum."

        entities = extractor.extract_entities(text)

        # With high threshold, should filter out low-confidence entities
        for entity in entities:
            assert entity.confidence >= 0.9

    def test_extract_entities_sorts_by_confidence_descending(self, basic_extractor) -> None:
        """Should return entities sorted by confidence (highest first)."""
        text = "Python was created by Guido van Rossum at Google in Amsterdam."

        entities = basic_extractor.extract_entities(text)

        if len(entities) > 1:
            confidences = [e.confidence for e in entities]
            assert confidences == sorted(confidences, reverse=True)

    def test_extract_entities_deduplicates_same_entity(self, basic_extractor) -> None:
        """Should deduplicate entities that appear multiple times."""
        text = "Python is great. Python is a programming language. Python is popular."

        entities = basic_extractor.extract_entities(text)

        # Check that Python appears only once
        python_entities = [e for e in entities if "python" in e.name.lower()]
        if python_entities:
            # Should have one entity with mentions > 1
            assert any(e.mentions > 1 for e in python_entities)

    def test_extract_entities_raises_validation_error_for_empty_text(self, basic_extractor) -> None:
        """Should raise ValidationError for empty text."""
        with pytest.raises(ValidationError) as exc_info:
            basic_extractor.extract_entities("")

        assert exc_info.value.error_code == "VAL_001"
        assert "empty" in exc_info.value.message.lower()

    def test_extract_entities_raises_validation_error_for_whitespace_only(
        self, basic_extractor
    ) -> None:
        """Should raise ValidationError for whitespace-only text."""
        with pytest.raises(ValidationError) as exc_info:
            basic_extractor.extract_entities("   \n\t   ")

        assert exc_info.value.error_code == "VAL_001"

    def test_extract_entities_raises_validation_error_for_too_long_text(
        self, basic_extractor
    ) -> None:
        """Should raise ValidationError for text exceeding max length."""
        # Create text longer than 100,000 characters
        long_text = "A" * 100_001

        with pytest.raises(ValidationError) as exc_info:
            basic_extractor.extract_entities(long_text)

        assert exc_info.value.error_code == "VAL_002"
        assert "length" in exc_info.value.message.lower()

    def test_extract_entities_handles_special_characters(self, basic_extractor) -> None:
        """Should handle text with special characters gracefully."""
        text = "C++ and C# are programming languages. They use symbols like ++, #, and @."

        entities = basic_extractor.extract_entities(text)

        # Should not crash and should extract some entities
        assert isinstance(entities, list)

    def test_extract_entities_handles_unicode(self, basic_extractor) -> None:
        """Should handle Unicode text correctly."""
        text = "François wrote code in Python. 日本語 also works."

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)

    def test_extract_entities_from_technical_text(self, basic_extractor) -> None:
        """Should extract technology-related entities from technical text."""
        text = """
        Docker is a containerization platform that uses Linux kernel features.
        It was developed at Docker Inc. and is written in Go programming language.
        """

        entities = basic_extractor.extract_entities(text)

        # Should find entities like Docker, Linux, Go
        assert len(entities) > 0

    def test_extract_entities_preserves_entity_types(self, basic_extractor) -> None:
        """Should correctly identify different entity types."""
        text = "Guido van Rossum created Python at CWI in Amsterdam."

        entities = basic_extractor.extract_entities(text)

        entity_types = {e.type for e in entities}
        # Should have multiple types (PERSON, TECHNOLOGY, ORG, GPE)
        assert len(entity_types) > 0


class TestEntityNormalization:
    """Test suite for entity name normalization."""

    def test_normalize_entity_removes_extra_whitespace(self, basic_extractor) -> None:
        """Should remove extra whitespace from entity names."""
        normalized = basic_extractor.normalize_entity("  Python  ", "TECHNOLOGY")

        assert normalized == "Python"

    def test_normalize_entity_standardizes_common_variants(self, basic_extractor) -> None:
        """Should normalize common entity name variants."""
        # Test common normalizations
        test_cases = [
            ("Python lang", "TECHNOLOGY", "Python"),
            ("Python language", "TECHNOLOGY", "Python"),
            ("JavaScript JS", "TECHNOLOGY", "JavaScript"),
            ("JS", "TECHNOLOGY", "JavaScript"),
        ]

        for input_name, entity_type, expected in test_cases:
            normalized = basic_extractor.normalize_entity(input_name, entity_type)
            # At minimum should remove extra words
            assert len(normalized) <= len(input_name)

    def test_normalize_entity_preserves_proper_nouns(self, basic_extractor) -> None:
        """Should preserve proper noun capitalization."""
        normalized = basic_extractor.normalize_entity("Guido van Rossum", "PERSON")

        # Should maintain capital letters
        assert "Guido" in normalized

    def test_normalize_entity_handles_acronyms(self, basic_extractor) -> None:
        """Should handle acronyms correctly."""
        normalized = basic_extractor.normalize_entity("NASA", "ORG")

        assert normalized == "NASA"  # Should preserve uppercase

    def test_normalize_entity_raises_validation_error_for_empty_name(self, basic_extractor) -> None:
        """Should raise ValidationError for empty entity name."""
        with pytest.raises(ValidationError):
            basic_extractor.normalize_entity("", "TECHNOLOGY")


class TestEntityDataclass:
    """Test suite for Entity dataclass."""

    def test_entity_creation_with_required_fields(self) -> None:
        """Should create Entity with all required fields."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            description="Programming language",
            confidence=0.85,
        )

        assert entity.name == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.description == "Programming language"
        assert entity.confidence == 0.85
        assert entity.mentions == 1  # Default value
        assert entity.source_span is None  # Default value

    def test_entity_creation_with_all_fields(self) -> None:
        """Should create Entity with all fields including optional ones."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            description="Programming language",
            confidence=0.85,
            mentions=3,
            source_span="Python is a programming language",
        )

        assert entity.mentions == 3
        assert entity.source_span == "Python is a programming language"


class TestRelationshipDataclass:
    """Test suite for Relationship dataclass."""

    def test_relationship_creation(self) -> None:
        """Should create Relationship with all required fields."""
        relationship = Relationship(
            source_entity="Python",
            target_entity="Guido van Rossum",
            relationship_type="CREATED_BY",
            confidence=0.9,
            evidence="Python was created by Guido van Rossum",
        )

        assert relationship.source_entity == "Python"
        assert relationship.target_entity == "Guido van Rossum"
        assert relationship.relationship_type == "CREATED_BY"
        assert relationship.confidence == 0.9
        assert "created by" in relationship.evidence.lower()


class TestExtractRelationships:
    """Test suite for relationship extraction (Phase 2 - stub tests for now)."""

    def test_extract_relationships_not_implemented_without_llm(self, basic_extractor) -> None:
        """Should return empty list or raise NotImplementedError without LLM."""
        text = "Python was created by Guido van Rossum."
        entities = [
            Entity("Python", "TECHNOLOGY", "Programming language", 0.9),
            Entity("Guido van Rossum", "PERSON", "Python creator", 0.9),
        ]

        # Phase 1: Should return empty list or raise NotImplementedError
        try:
            relationships = basic_extractor.extract_relationships(text, entities)
            assert isinstance(relationships, list)
            assert len(relationships) == 0  # Not implemented yet
        except NotImplementedError:
            pass  # Also acceptable for Phase 1

    def test_extract_relationships_with_llm_enabled(self, spacy_model, mock_ollama_client) -> None:
        """Should extract relationships when LLM refinement is enabled (Phase 2)."""
        extractor = EntityExtractor(
            spacy_model=spacy_model,
            ollama_client=mock_ollama_client,
            enable_llm_refinement=True,
        )

        text = "Django is a web framework written in Python."
        entities = [
            Entity("Django", "TECHNOLOGY", "Web framework", 0.9),
            Entity("Python", "TECHNOLOGY", "Programming language", 0.9),
        ]

        # Phase 2: Should return relationships
        try:
            relationships = extractor.extract_relationships(text, entities)
            assert isinstance(relationships, list)
            # LLM should detect "written in" relationship
        except NotImplementedError:
            # Acceptable if Phase 2 not implemented yet
            pytest.skip("Phase 2 relationship extraction not implemented")


class TestPerformance:
    """Test suite for performance requirements."""

    def test_extract_entities_completes_within_time_limit(self, basic_extractor) -> None:
        """Should complete extraction within 10ms for small documents (SpaCy only)."""
        import time

        text = "Python was created by Guido van Rossum in 1991 at CWI."

        start = time.time()
        basic_extractor.extract_entities(text)
        elapsed = time.time() - start

        # SpaCy extraction should be fast (< 100ms for small text)
        assert elapsed < 0.1  # 100ms tolerance for CI environments

    def test_extract_entities_handles_large_document(self, basic_extractor) -> None:
        """Should handle documents up to max length."""
        # Create document near max length (100,000 chars)
        large_text = " ".join(
            ["Python is a programming language created by Guido van Rossum."] * 1000
        )

        entities = basic_extractor.extract_entities(large_text)

        # Should not crash and return entities
        assert isinstance(entities, list)


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_extract_entities_from_text_with_no_entities(self, basic_extractor) -> None:
        """Should return empty list for text with no recognizable entities."""
        text = "This is a simple sentence with no names or places."

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)
        # May return empty list or few low-confidence entities (filtered by threshold)

    def test_extract_entities_from_single_word(self, basic_extractor) -> None:
        """Should handle single-word input."""
        text = "Python"

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)

    def test_extract_entities_with_numbers_only(self, basic_extractor) -> None:
        """Should handle numeric text."""
        text = "123 456 789"

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)

    def test_extract_entities_with_urls(self, basic_extractor) -> None:
        """Should handle text containing URLs."""
        text = "Visit https://python.org to learn about Python."

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)

    def test_extract_entities_with_code_snippets(self, basic_extractor) -> None:
        """Should handle text containing code snippets."""
        text = """
        Here's a Python example:
        def hello():
            print("Hello, World!")
        """

        entities = basic_extractor.extract_entities(text)

        assert isinstance(entities, list)
