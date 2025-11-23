# Code Review Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Code review for PR #{pr_number}

PR INFO:
- Feature: {feature_name}
- Branch: {branch_name}
- Files changed: {count}
- Author: {agent_name}

REVIEW CHECKLIST:

## 1. Code Quality

**Style:**
- ✅ Black formatted (100 chars)
- ✅ isort applied
- ✅ Consistent naming conventions

**Readability:**
- ✅ Functions < 50 lines
- ✅ Clear variable names
- ✅ No magic numbers/strings
- ✅ DRY principle (no duplication)

**Documentation:**
- ✅ Docstrings present (Google style)
- ✅ Complex logic commented
- ✅ Type hints 100%

## 2. Testing

**Coverage:**
- ✅ Overall >= 90%
- ✅ All functions tested
- ✅ Edge cases covered (from specs)

**Quality:**
- ✅ Test names descriptive: `test_{function}_{scenario}_{expected}`
- ✅ Tests isolated (no shared state)
- ✅ Mocking used appropriately

## 3. Security

**Input Validation:**
- ✅ All user inputs validated
- ✅ No SQL injection vectors
- ✅ No command injection
- ✅ No path traversal

**Error Handling:**
- ✅ Exceptions caught appropriately
- ✅ No sensitive data in error messages
- ✅ Proper logging (no secrets logged)

## 4. Alignment with Specs

**Spec Compliance:**
- ✅ All function signatures match specs exactly
- ✅ All edge cases from specs handled
- ✅ All test scenarios from specs implemented

**Completeness:**
- ✅ All required functions implemented
- ✅ No extra functions (scope creep)

## 5. Performance

**Efficiency:**
- ✅ No obvious performance issues (N^2 loops, etc)
- ✅ Database queries optimized
- ✅ Resource cleanup (connections closed)

ФОРМАТ ОТЧЁТА:

```markdown
# Code Review - PR #{pr_number}

**Feature:** {feature_name}
**Reviewer:** Code Review Agent
**Date:** {timestamp}

## ✅ APPROVED ASPECTS

- Code quality: Excellent formatting, clear naming
- Tests: 95% coverage, all edge cases tested
- Security: Input validation comprehensive

## ⚠️ MINOR ISSUES (nice to fix)

### Issue 1: Magic Number
- **Location:** `src/module/file.py:45`
- **Current:** `if x > 100:`
- **Suggestion:** `if x > MAX_ALLOWED_SIZE:`
- **Priority:** Low

## ❌ BLOCKING ISSUES (must fix)

### Issue 1: Missing Input Validation
- **Location:** `src/module/file.py:67`
- **Issue:** Function accepts `data: dict` but doesn't validate required keys
- **Impact:** Will crash on invalid input
- **Fix:** Add validation as per spec edge case #2
- **Priority:** Critical

## METRICS

- Files reviewed: {count}
- Lines changed: +{added} -{removed}
- Tests: {count} ({coverage}% coverage)
- Issues: Critical {X}, Minor {Y}

## VERDICT

- [ ] ✅ APPROVE (merge ready)
- [ ] ⚠️ APPROVE WITH COMMENTS (minor issues, can merge)
- [X] ❌ REQUEST CHANGES (blocking issues, cannot merge)

**Reasoning:** {explanation}
```

---

НАЧИНАЙ CODE REVIEW.
