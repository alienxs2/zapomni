# Verification Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Multi-document verification

ASSIGNED DOCUMENTS:
{list_of_spec_files}

VERIFICATION CHECKLIST:

## 1. Internal Consistency (каждый документ отдельно)

For each document:
- ✅ No contradictions within document
- ✅ All cross-references valid
- ✅ Diagrams match text descriptions
- ✅ Code examples syntactically correct

## 2. Cross-Document Consistency (между assigned docs)

Check:
- ✅ API contracts match
  - If doc A exports interface X, doc B imports X correctly
- ✅ Data models aligned
  - Same structure described identically
- ✅ Dependencies correct
  - If A depends on B, B provides it
- ✅ No circular dependencies

## 3. Steering Alignment

Verify against:
- /home/dev/zapomni/.spec-workflow/steering/product.md
  - Features match vision
- /home/dev/zapomni/.spec-workflow/steering/tech.md
  - Technologies used match stack
- /home/dev/zapomni/.spec-workflow/steering/structure.md
  - Organization matches conventions

## 4. Technical Feasibility

- ✅ Can be implemented with chosen tech stack (FalkorDB, Ollama, Python)
- ✅ Performance targets realistic
- ✅ No architectural impossibilities

## 5. Completeness

- ✅ All features from product.md covered
- ✅ All edge cases enumerated
- ✅ Error handling specified

ФОРМАТ ОТЧЁТА:

Save to: `verification_reports/{level}/agent{N}_report.md`

```markdown
# Verification Report - Agent {N}

**Documents Verified:** {list}
**Date:** {timestamp}

## ✅ APPROVED ASPECTS

- Document {N}:
  - {что хорошо}

- Documents {N} & {M}:
  - {что согласовано}

## ⚠️ WARNINGS (non-critical)

### Warning 1: {title}
- **Location:** {doc}:{section}
- **Issue:** {description}
- **Suggestion:** {how to fix}
- **Priority:** Low/Medium

## ❌ CRITICAL ISSUES (blocking)

### Issue 1: {title}
- **Location:** {doc}:{section}
- **Type:** Contradiction / Missing / Infeasible
- **Description:** {detailed explanation}
- **Impact:** {what breaks if not fixed}
- **Recommendation:** {how to fix}
- **Priority:** High/Critical

## CROSS-DOCUMENT FINDINGS

### Finding 1: {title}
- **Documents:** {doc1} <-> {doc2}
- **Issue:** {description}
- **Fix:** {recommendation}

## METRICS

- Documents analyzed: {count}
- Issues found: Critical {X}, Warnings {Y}
- Consistency score: {percentage}%

## FINAL VERDICT

- [ ] APPROVE (zero critical issues)
- [ ] APPROVE WITH WARNINGS (< 3 warnings, zero critical)
- [X] REJECT (>= 1 critical issue)

**Justification:** {reasoning}
```

---

НАЧИНАЙ VERIFICATION.
