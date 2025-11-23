# Reconciliation Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Reconcile synthesis recommendations with steering documents

INPUT:
- Synthesis report: verification_reports/{level}/synthesis_report.md

STEERING DOCUMENTS:
- /home/dev/zapomni/.spec-workflow/steering/product.md
- /home/dev/zapomni/.spec-workflow/steering/tech.md
- /home/dev/zapomni/.spec-workflow/steering/structure.md

PROCESS:

## Step 1: Read Synthesis Report

Extract all confirmed critical issues and recommendations.

## Step 2: Steering Alignment Check

For each recommended fix:

**Check against product.md:**
- Does it align with product vision?
- Does it preserve intended features?
- Does it support user goals?

**Check against tech.md:**
- Is it technically sound with chosen stack?
- Does it respect architectural decisions?
- Are performance impacts acceptable?

**Check against structure.md:**
- Does it follow coding conventions?
- Does it match organizational patterns?
- Is it consistent with project structure?

## Step 3: Validate or Adjust

For each recommendation:
- ✅ APPROVED: Aligns with all steering docs
- ⚠️ APPROVED WITH ADJUSTMENT: Needs minor tweak for alignment
- ❌ REJECTED: Conflicts with steering vision

## Step 4: Technical Feasibility

Ensure all approved recommendations are implementable with:
- Python 3.10+
- FalkorDB
- Ollama
- MCP protocol

ФОРМАТ ОТЧЁТА:

Save to: `verification_reports/{level}/reconciliation_report.md`

```markdown
# Reconciliation Report - {Level}

**Date:** {timestamp}
**Input:** Synthesis report + Steering documents

## STEERING ALIGNMENT CHECK

### Issue 1: {title}
**Synthesis Recommendation:** {original recommendation}

**Steering Check:**
- product.md: ✅ / ⚠️ / ❌ {reason}
- tech.md: ✅ / ⚠️ / ❌ {reason}
- structure.md: ✅ / ⚠️ / ❌ {reason}

**Validation:** ✅ APPROVED / ⚠️ APPROVED WITH ADJUSTMENT / ❌ REJECTED
**Adjusted Recommendation:** {if adjusted, new recommendation}

### Issue 2: {title}
[Same format for each issue]

## TECHNICAL FEASIBILITY

All recommended fixes feasible: ✅ Yes / ❌ No (explain)

## FINAL RECOMMENDATIONS

1. **Issue 1:** {approved fix} ✅
2. **Issue 2:** {approved fix with adjustment} ✅
3. **Issue 3:** {rejected, reason} ❌

## DECISION

Proceed to refinement phase: ✅ Yes / ❌ No (needs user decision)
```

---

НАЧИНАЙ RECONCILIATION.
