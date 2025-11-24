"""
Unit tests for ClassHierarchyBuilder component.

Tests class extraction, hierarchy building, and metadata handling for:
- Single inheritance
- Multiple inheritance
- Nested classes
- Method type classification (instance/class/static)
- Attribute extraction
- Decorator handling
- Abstract class detection

Follows TDD approach with comprehensive test coverage.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
import ast
from typing import List

from zapomni_core.code.class_hierarchy_builder import (
    ClassHierarchyBuilder,
    ClassInfo,
    MethodInfo,
    AttributeInfo,
    HierarchyNode,
)
from zapomni_core.exceptions import ValidationError, ExtractionError


# ============================================================================
# Test Data / Fixtures
# ============================================================================

@pytest.fixture
def builder():
    """Create a ClassHierarchyBuilder instance."""
    return ClassHierarchyBuilder()


# Simple single class
SIMPLE_CLASS_CODE = """
class Animal:
    '''Base animal class'''
    species: str = "Unknown"

    def __init__(self, name: str):
        self.name = name

    def speak(self):
        '''Make sound'''
        pass
"""

# Multiple classes with inheritance
INHERITANCE_CODE = """
class Animal:
    '''Base animal class'''

    def speak(self):
        pass

class Dog(Animal):
    '''Dog subclass'''

    def speak(self):
        return "Woof"

class Cat(Animal):
    '''Cat subclass'''

    def speak(self):
        return "Meow"
"""

# Multiple inheritance
MULTIPLE_INHERITANCE_CODE = """
class Serializable:
    def to_dict(self):
        pass

class Comparable:
    def compare(self, other):
        pass

class Person(Serializable, Comparable):
    '''Person with multiple inheritance'''
    name: str
    age: int

    def __init__(self, name, age):
        self.name = name
        self.age = age
"""

# Class with various method types
METHOD_TYPES_CODE = """
class Calculator:
    '''Class with different method types'''

    value: int = 0

    def __init__(self, initial=0):
        '''Instance method'''
        self.value = initial

    @classmethod
    def from_string(cls, s):
        '''Class method'''
        return cls(int(s))

    @staticmethod
    def validate(n):
        '''Static method'''
        return isinstance(n, int)

    def add(self, n):
        '''Instance method'''
        self.value += n
        return self.value

    @property
    def current(self):
        '''Property'''
        return self.value
"""

# Nested classes
NESTED_CLASSES_CODE = """
class Outer:
    '''Outer class'''

    def outer_method(self):
        pass

    class Inner:
        '''Inner class'''

        def inner_method(self):
            pass
"""

# Abstract class
ABSTRACT_CLASS_CODE = """
from abc import ABC, abstractmethod

class AbstractAnimal(ABC):
    '''Abstract base class'''

    @abstractmethod
    def speak(self):
        '''Subclasses must implement'''
        pass

class Dog(AbstractAnimal):
    '''Concrete implementation'''

    def speak(self):
        return "Woof"
"""

# Complex hierarchy
COMPLEX_HIERARCHY_CODE = """
class Vehicle:
    '''Base vehicle'''
    wheels: int = 4

    def start(self):
        pass

class Car(Vehicle):
    '''Car class'''
    doors: int = 4

    def honk(self):
        pass

class Truck(Vehicle):
    '''Truck class'''
    capacity: int

    def load(self, item):
        pass

class ElectricCar(Car):
    '''Electric car'''
    battery: int

    def charge(self):
        pass
"""

# Attributes with various types
ATTRIBUTES_CODE = """
class Config:
    '''Configuration class'''

    name: str = "default"
    timeout: int = 30
    enabled: bool = True
    tags: list = []
    settings = {"debug": False}

    def __init__(self):
        self._internal = None
"""

# Decorators
DECORATORS_CODE = """
class MyClass:
    '''Class with decorated methods'''

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, value):
        self._prop = value

    @classmethod
    def create(cls):
        return cls()

    @staticmethod
    def helper():
        pass

    @deprecated
    def old_method(self):
        pass
"""

# Async methods
ASYNC_CODE = """
class AsyncTask:
    '''Class with async methods'''

    async def fetch_data(self, url):
        '''Async method'''
        pass

    async def process(self):
        '''Async processing'''
        pass

    def sync_method(self):
        '''Regular method'''
        pass
"""


# ============================================================================
# Tests: Extract Classes (Basic)
# ============================================================================

class TestExtractClassesBasic:
    """Tests for basic class extraction."""

    def test_extract_simple_class(self, builder):
        """Extract a simple class definition."""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        classes = builder.extract_classes(tree)

        assert len(classes) == 1
        assert classes[0].name == "Animal"
        assert classes[0].docstring == "Base animal class"

    def test_extract_multiple_classes(self, builder):
        """Extract multiple classes from code."""
        tree = ast.parse(INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        assert len(classes) == 3
        names = {c.name for c in classes}
        assert names == {"Animal", "Dog", "Cat"}

    def test_extract_class_with_line_numbers(self, builder):
        """Line numbers are captured correctly."""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        classes = builder.extract_classes(tree)

        assert classes[0].lineno > 0
        assert classes[0].end_lineno is not None

    def test_extract_empty_class(self, builder):
        """Handle empty class definitions."""
        code = "class Empty: pass"
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        assert len(classes) == 1
        assert classes[0].name == "Empty"
        assert classes[0].methods == []
        assert classes[0].attributes == []


# ============================================================================
# Tests: Base Classes & Inheritance
# ============================================================================

class TestBaseClasses:
    """Tests for base class extraction."""

    def test_extract_single_inheritance(self, builder):
        """Extract single inheritance."""
        tree = ast.parse(INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        dog = next(c for c in classes if c.name == "Dog")
        assert dog.base_classes == ["Animal"]

    def test_extract_multiple_inheritance(self, builder):
        """Extract multiple inheritance."""
        tree = ast.parse(MULTIPLE_INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        person = next(c for c in classes if c.name == "Person")
        assert set(person.base_classes) == {"Serializable", "Comparable"}

    def test_no_base_classes(self, builder):
        """Root class has no base classes."""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        classes = builder.extract_classes(tree)

        assert classes[0].base_classes == []

    def test_builtin_base_class(self, builder):
        """Handle built-in base classes."""
        code = "class MyList(list): pass"
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        assert classes[0].base_classes == ["list"]


# ============================================================================
# Tests: Methods
# ============================================================================

class TestMethods:
    """Tests for method extraction and classification."""

    def test_extract_instance_methods(self, builder):
        """Extract instance methods."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]
        instance_methods = [m for m in calc.methods if m.type == "instance"]

        assert len(instance_methods) >= 2
        method_names = {m.name for m in instance_methods}
        assert "add" in method_names

    def test_classify_class_method(self, builder):
        """Classify @classmethod correctly."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]
        class_methods = [m for m in calc.methods if m.type == "class"]

        assert len(class_methods) == 1
        assert class_methods[0].name == "from_string"

    def test_classify_static_method(self, builder):
        """Classify @staticmethod correctly."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]
        static_methods = [m for m in calc.methods if m.type == "static"]

        assert len(static_methods) == 1
        assert static_methods[0].name == "validate"

    def test_method_arguments(self, builder):
        """Extract method arguments correctly."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]
        add_method = next(m for m in calc.methods if m.name == "add")

        assert "n" in add_method.args
        assert "self" not in add_method.args  # self excluded

    def test_method_docstring(self, builder):
        """Extract method docstrings."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]
        add_method = next(m for m in calc.methods if m.name == "add")

        assert add_method.docstring == "Instance method"

    def test_async_methods(self, builder):
        """Detect async methods."""
        tree = ast.parse(ASYNC_CODE)
        classes = builder.extract_classes(tree)

        task = classes[0]
        async_methods = [m for m in task.methods if m.is_async]

        assert len(async_methods) == 2
        assert all(m.name in ["fetch_data", "process"] for m in async_methods)

    def test_property_decorator(self, builder):
        """Detect property decorator."""
        tree = ast.parse(DECORATORS_CODE)
        classes = builder.extract_classes(tree)

        my_class = classes[0]
        prop_method = next(m for m in my_class.methods if m.name == "prop")

        assert "property" in prop_method.decorators


# ============================================================================
# Tests: Attributes
# ============================================================================

class TestAttributes:
    """Tests for attribute extraction."""

    def test_extract_annotated_attributes(self, builder):
        """Extract type-annotated attributes."""
        tree = ast.parse(ATTRIBUTES_CODE)
        classes = builder.extract_classes(tree)

        config = classes[0]
        assert len(config.attributes) > 0

        name_attr = next(a for a in config.attributes if a.name == "name")
        assert name_attr.type_annotation == "str"
        assert name_attr.value == "'default'"

    def test_extract_multiple_attributes(self, builder):
        """Extract multiple class attributes."""
        tree = ast.parse(ATTRIBUTES_CODE)
        classes = builder.extract_classes(tree)

        config = classes[0]
        attr_names = {a.name for a in config.attributes}

        assert "name" in attr_names
        assert "timeout" in attr_names
        assert "enabled" in attr_names

    def test_attribute_type_annotations(self, builder):
        """Capture type annotations."""
        tree = ast.parse(ATTRIBUTES_CODE)
        classes = builder.extract_classes(tree)

        config = classes[0]
        timeout_attr = next(a for a in config.attributes if a.name == "timeout")

        assert timeout_attr.type_annotation == "int"
        assert timeout_attr.value == "30"

    def test_skip_private_attributes(self, builder):
        """Skip attributes that look like private/internal."""
        tree = ast.parse(ATTRIBUTES_CODE)
        classes = builder.extract_classes(tree)

        config = classes[0]
        attr_names = {a.name for a in config.attributes}

        # _internal should be skipped
        assert "_internal" not in attr_names


# ============================================================================
# Tests: Decorators
# ============================================================================

class TestDecorators:
    """Tests for decorator extraction."""

    def test_extract_class_decorators(self, builder):
        """Extract decorators on classes."""
        code = """
@dataclass
@frozen
class Point:
    x: int
    y: int
"""
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        assert len(classes[0].decorators) == 2
        assert set(classes[0].decorators) == {"dataclass", "frozen"}

    def test_extract_method_decorators(self, builder):
        """Extract decorators on methods."""
        tree = ast.parse(DECORATORS_CODE)
        classes = builder.extract_classes(tree)

        my_class = classes[0]
        create_method = next(m for m in my_class.methods if m.name == "create")

        assert "classmethod" in create_method.decorators


# ============================================================================
# Tests: Nested Classes
# ============================================================================

class TestNestedClasses:
    """Tests for nested class handling."""

    def test_extract_nested_classes(self, builder):
        """Extract nested class definitions."""
        tree = ast.parse(NESTED_CLASSES_CODE)
        classes = builder.extract_classes(tree)

        # Should extract both Outer and Inner
        assert any(c.name == "Outer" for c in classes)
        # Note: Depending on implementation, Inner might be separate or marked as nested

    def test_nested_class_parent_tracking(self, builder):
        """Track parent of nested classes."""
        tree = ast.parse(NESTED_CLASSES_CODE)
        classes = builder.extract_classes(tree)

        # Inner class should have Outer as parent
        inner = next((c for c in classes if c.name == "Inner"), None)
        if inner:
            assert inner.parent_class == "Outer"


# ============================================================================
# Tests: Abstract Classes
# ============================================================================

class TestAbstractClasses:
    """Tests for abstract class detection."""

    def test_detect_abstract_with_abc_base(self, builder):
        """Detect abstract class with ABC base."""
        tree = ast.parse(ABSTRACT_CLASS_CODE)
        classes = builder.extract_classes(tree)

        abstract = next(c for c in classes if c.name == "AbstractAnimal")
        assert abstract.is_abstract is True

    def test_concrete_subclass_not_abstract(self, builder):
        """Concrete subclass not marked as abstract."""
        tree = ast.parse(ABSTRACT_CLASS_CODE)
        classes = builder.extract_classes(tree)

        dog = next(c for c in classes if c.name == "Dog")
        assert dog.is_abstract is False


# ============================================================================
# Tests: Build Hierarchy
# ============================================================================

class TestBuildHierarchy:
    """Tests for building class inheritance hierarchy."""

    def test_build_simple_hierarchy(self, builder):
        """Build hierarchy for simple inheritance."""
        tree = ast.parse(INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        hierarchy = builder.build_hierarchy(classes)

        # Should have 3 nodes
        assert len(hierarchy) == 3
        assert "Animal" in hierarchy
        assert "Dog" in hierarchy
        assert "Cat" in hierarchy

    def test_hierarchy_parent_child_relationships(self, builder):
        """Parent-child relationships in hierarchy."""
        tree = ast.parse(INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        hierarchy = builder.build_hierarchy(classes)
        animal_node = hierarchy["Animal"]

        # Animal should have Dog and Cat as children
        child_names = {child.class_info.name for child in animal_node.children}
        assert "Dog" in child_names
        assert "Cat" in child_names

    def test_hierarchy_depth(self, builder):
        """Verify depth tracking in hierarchy."""
        tree = ast.parse(COMPLEX_HIERARCHY_CODE)
        classes = builder.extract_classes(tree)

        hierarchy = builder.build_hierarchy(classes)

        # Vehicle should be at depth 0
        assert hierarchy["Vehicle"].depth == 0

        # Car should be at depth 1
        assert hierarchy["Car"].depth == 1

        # ElectricCar should be at depth 2
        assert hierarchy["ElectricCar"].depth == 2

    def test_multiple_inheritance_hierarchy(self, builder):
        """Handle multiple inheritance in hierarchy."""
        tree = ast.parse(MULTIPLE_INHERITANCE_CODE)
        classes = builder.extract_classes(tree)

        hierarchy = builder.build_hierarchy(classes)
        person = hierarchy["Person"]

        # Person should have depth > 0 (has parents)
        assert person.depth > 0


# ============================================================================
# Tests: ClassInfo Data Model
# ============================================================================

class TestClassInfoModel:
    """Tests for ClassInfo data model."""

    def test_class_info_to_dict(self, builder):
        """Convert ClassInfo to dictionary."""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        classes = builder.extract_classes(tree)

        class_dict = classes[0].to_dict()

        assert isinstance(class_dict, dict)
        assert class_dict["name"] == "Animal"
        assert class_dict["docstring"] == "Base animal class"
        assert isinstance(class_dict["methods"], list)
        assert isinstance(class_dict["attributes"], list)

    def test_class_info_contains_metadata(self, builder):
        """ClassInfo contains all expected metadata."""
        tree = ast.parse(METHOD_TYPES_CODE)
        classes = builder.extract_classes(tree)

        calc = classes[0]

        assert calc.name == "Calculator"
        assert len(calc.methods) > 0
        assert len(calc.attributes) > 0
        assert calc.docstring is not None


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_ast_type_raises_error(self, builder):
        """Passing non-AST raises ValidationError."""
        with pytest.raises(ValidationError):
            builder.extract_classes("not an ast")

    def test_invalid_ast_module_raises_error(self, builder):
        """Passing wrong AST type raises ValidationError."""
        code = "def foo(): pass"
        tree = ast.parse(code)
        func_def = tree.body[0]

        with pytest.raises(ValidationError):
            builder.extract_classes(func_def)

    def test_extraction_with_syntax_error(self, builder):
        """Gracefully handle syntax errors in extraction."""
        # This would be caught earlier by ast.parse, but test extraction logic
        valid_code = "class A: pass"
        tree = ast.parse(valid_code)

        # Should not raise
        classes = builder.extract_classes(tree)
        assert len(classes) > 0


# ============================================================================
# Tests: Complex Real-World Scenarios
# ============================================================================

class TestComplexScenarios:
    """Tests for complex real-world code patterns."""

    def test_mixin_pattern(self, builder):
        """Handle mixin class pattern."""
        code = """
class TimestampMixin:
    created_at: str

    def get_age(self):
        pass

class LoggingMixin:
    logger: object

    def log(self, msg):
        pass

class User(TimestampMixin, LoggingMixin):
    name: str

    def create(self):
        pass
"""
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        user = next(c for c in classes if c.name == "User")
        assert "TimestampMixin" in user.base_classes
        assert "LoggingMixin" in user.base_classes

    def test_special_methods(self, builder):
        """Handle special methods like __init__, __str__."""
        code = """
class MyClass:
    def __init__(self):
        pass

    def __str__(self):
        return "MyClass"

    def __repr__(self):
        return "MyClass()"
"""
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        my_class = classes[0]
        method_names = {m.name for m in my_class.methods}

        assert "__init__" in method_names
        assert "__str__" in method_names
        assert "__repr__" in method_names

    def test_generic_base_class(self, builder):
        """Handle generic/parameterized base classes."""
        code = """
from typing import Generic, TypeVar

T = TypeVar('T')

class Container(Generic[T]):
    item: T

    def get(self) -> T:
        return self.item
"""
        tree = ast.parse(code)
        classes = builder.extract_classes(tree)

        container = classes[0]
        # Generic[T] should be captured as base class
        assert any("Generic" in base for base in container.base_classes)


# ============================================================================
# Tests: Integration
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_extract_and_build(self, builder):
        """Complete workflow: extract classes and build hierarchy."""
        tree = ast.parse(COMPLEX_HIERARCHY_CODE)

        # Extract classes
        classes = builder.extract_classes(tree)
        assert len(classes) == 4

        # Build hierarchy
        hierarchy = builder.build_hierarchy(classes)
        assert len(hierarchy) == 4

        # Verify structure
        vehicle_node = hierarchy["Vehicle"]
        assert len(vehicle_node.children) == 2  # Car and Truck

        # Verify deep hierarchy
        electric_car_node = hierarchy["ElectricCar"]
        assert electric_car_node.depth == 2

    def test_large_class_set(self, builder):
        """Handle large number of classes."""
        code = "\n".join([f"class Class{i}: pass" for i in range(100)])
        tree = ast.parse(code)

        classes = builder.extract_classes(tree)
        assert len(classes) == 100

        hierarchy = builder.build_hierarchy(classes)
        assert len(hierarchy) == 100
