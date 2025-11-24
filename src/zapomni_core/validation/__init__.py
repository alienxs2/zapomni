"""
Validation module for zapomni_core.

Provides input validation functionality for text, metadata, queries,
and pagination parameters.

Currently provides:
- InputValidator: Centralized input validation for all Zapomni inputs

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from .input_validator import InputValidator

__all__ = ["InputValidator"]
