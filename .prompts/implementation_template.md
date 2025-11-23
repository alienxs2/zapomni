# Implementation Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Implement {feature_name} feature using Test-Driven Development

PHASE: {TESTS_FIRST | CODE_GREEN | REFACTOR}

FUNCTION SPECS:
{list_of_function_level_spec_files}

TDD WORKFLOW:

## STEP 1: WRITE TESTS (RED phase)

For each function in specs:
1. Read function spec: {spec_file}
2. Create test file: `tests/{type}/test_{module}_{function}.py`
3. Write tests:
   - Happy path (success scenario)
   - Each edge case from spec
   - All error cases (exceptions)
4. Run tests → MUST FAIL (no code yet)
5. Verify coverage target: 100% functions have tests

## STEP 2: WRITE CODE (GREEN phase)

For each function:
1. Write MINIMAL code to make tests pass
2. Run tests → should turn GREEN
3. Check coverage: >= 90%
4. NO refactoring yet

## STEP 3: REFACTOR

1. Improve code quality (DRY, readability)
2. Tests MUST stay GREEN during refactoring
3. Run tests after each refactor iteration
4. Final check: all tests green, coverage >= 90%

GIT WORKFLOW:

1. Create branch: `feature/{feature_name}`
2. Commit tests: `test({feature}): Add comprehensive test suite`
3. Commit code: `feat({feature}): Implement {feature_name}`
4. Commit refactor: `refactor({feature}): Improve code quality`
5. Push and create PR

QUALITY CHECKS:

✅ All tests pass (100%)
✅ Coverage >= 90%
✅ Type hints (mypy clean)
✅ Formatted (black, isort)
✅ No linting errors

ФОРМАТ ОТЧЁТА:

```markdown
## IMPLEMENTATION COMPLETE

**Feature:** {feature_name}
**Branch:** feature/{feature_name}
**Commits:** {count}

**Tests:**
- Total: {count}
- Passing: {count} (100%)
- Coverage: {percentage}%

**Files Created:**
- {list}

**Files Modified:**
- {list}

**Ready for Code Review:** Yes
```

---

НАЧИНАЙ IMPLEMENTATION.
