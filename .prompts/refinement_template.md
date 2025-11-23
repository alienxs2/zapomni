# Refinement Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Apply approved fixes to specification documents

INPUT:
- Reconciliation report: verification_reports/{level}/reconciliation_report.md

PROCESS:

## Step 1: Read Reconciliation Report

Extract all approved recommendations (✅).

## Step 2: Apply Fixes

For each approved recommendation:
1. Identify affected spec file(s)
2. Locate exact section to change
3. Apply fix as recommended
4. Track change (file, line numbers, before/after)

## Step 3: Consistency Update

If change affects multiple specs:
- Update all related specs for consistency
- Ensure terminology updated everywhere
- Check cross-references still valid

## Step 4: Verification Prep

Identify which specs changed:
- Major changes → needs re-verification
- Minor changes → skip re-verification (user decision)

ФОРМАТ ОТЧЁТА:

Save to: `verification_reports/{level}/refinement_report.md`

```markdown
# Refinement Actions - {Level}

**Date:** {timestamp}
**Input:** Reconciliation report

## CHANGES MADE

### Changed Document: {filename}

**Section:** {section_name}

**Before:**
```
{original text/code}
```

**After:**
```
{updated text/code}
```

**Rationale:** {why this change, reference to issue}

### Changed Document: {another_file}
[Same format]

## WARNINGS ADDRESSED

- Fixed naming inconsistency: "{old}" → "{new}"
- Added usage examples to {doc}
- Updated cross-references in {docs}

## FILES MODIFIED

- {file1} (lines {ranges})
- {file2} (lines {ranges})

## RE-VERIFICATION NEEDED

**Major Changes (require re-verification):**
- {file1}, {file2}

**Minor Changes (skip re-verification):**
- {file3}, {file4}

## STATUS

Refinement complete. Ready for re-verification (if needed) or user approval.
```

---

НАЧИНАЙ REFINEMENT.
