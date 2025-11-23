# Synthesis Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Synthesize 5 verification reports into consolidated analysis

INPUT FILES:
- verification_reports/{level}/agent1_report.md
- verification_reports/{level}/agent2_report.md
- verification_reports/{level}/agent3_report.md
- verification_reports/{level}/agent4_report.md
- verification_reports/{level}/agent5_report.md

PROCESS:

## Step 1: Read All Reports

Load all 5 verification reports.

## Step 2: Identify Patterns

For each issue:
- If 2+ agents found same issue → CONFIRMED ISSUE
- If only 1 agent → UNIQUE ISSUE (possible false positive or missed by others)

## Step 3: Categorize Issues

**Confirmed Critical Issues:**
- Found by 2+ agents
- Severity: Critical
- Must be fixed

**Confirmed Warnings:**
- Found by 2+ agents
- Severity: Warning
- Nice to fix

**Unique Critical Issues:**
- Found by 1 agent only
- Needs investigation (reconciliation agent will validate)

**Unique Warnings:**
- Found by 1 agent only
- Low priority

## Step 4: Aggregate Approved Aspects

What all agents agreed is good.

## Step 5: Decision

Based on confirmed critical issues count:
- 0 confirmed critical → APPROVE
- 1-2 confirmed critical → APPROVE WITH WARNINGS (if fixable)
- 3+ confirmed critical → REJECT (needs major rework)

ФОРМАТ ОТЧЁТА:

Save to: `verification_reports/{level}/synthesis_report.md`

```markdown
# Synthesis Report - {Level} Verification

**Date:** {timestamp}
**Input:** 5 agent verification reports
**Documents:** {count} specs

## SUMMARY

- Total agents: 5
- Total issues found: {X} ({Y} critical, {Z} warnings)
- Confirmed issues: {count} (found by 2+ agents)
- Unique issues: {count} (found by 1 agent)

## CONFIRMED CRITICAL ISSUES

### Issue 1: {title} (Found by Agents {N, M})
- **Consistency:** {count} agents independently identified
- **Severity:** Critical
- **Description:** {consolidated from reports}
- **Recommended Fix:** {consensus approach}

### Issue 2: {title} (Found by Agents {N, M, K})
[Same format]

## UNIQUE CRITICAL ISSUES (needs validation)

### Issue X: {title} (Agent {N} only)
- **Needs Validation:** Only 1 agent found this
- **Possible False Positive:** Or other agents missed it
- **Action:** Reconciliation agent should investigate

## WARNINGS

{Consolidated warnings}

## APPROVED ASPECTS

{What all agents agreed is good}

## DECISION

- [ ] APPROVE (zero confirmed critical)
- [ ] APPROVE WITH WARNINGS (< 3 confirmed critical, all fixable)
- [X] REJECT (>= 3 confirmed critical OR unfixable issues)

## NEXT STEPS

1. Reconciliation agent validates unique issues
2. Refinement agent fixes confirmed issues
3. Re-verification of changed docs (if needed)
```

---

НАЧИНАЙ SYNTHESIS.
