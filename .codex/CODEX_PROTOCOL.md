# Codex CLI Protocol for Claude Code Integration

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-23
**Purpose:** Standard protocol for Claude Code CLI to delegate tasks to Codex CLI

---

## 🎯 Core Principle

**Claude Code = Manager (Planning, Coordination, Monitoring)**
**Codex CLI = Worker (Implementation, Testing, Review)**

All coding tasks delegated to Codex CLI. Claude Code only monitors and reports results.

---

## ⚠️ CRITICAL: Sandbox Modes

**THREE modes (use correct one!):**

```bash
# 1. READ-ONLY (default) - CANNOT create files
codex exec "analyze code"
# Use for: Analysis only

# 2. WORKSPACE-WRITE (recommended) - CAN create files in workdir
codex exec "implement component" --sandbox workspace-write
# Use for: Implementation, tests, code generation

# 3. DANGER-FULL-ACCESS (caution) - CAN write anywhere
codex exec "system task" --sandbox danger-full-access
# Use for: System-level tasks only
```

**❌ COMMON MISTAKE:**
```bash
codex exec "implement X" --sandbox full  # ERROR: invalid value
```

**✅ CORRECT:**
```bash
codex exec "implement X" --sandbox workspace-write  # CORRECT
```

---

## 📊 Enhanced Protocol with Progress Tracking

### CODEX PROMPT TEMPLATE (Standard)

```bash
codex exec "
⚠️ PROGRESS TRACKING (MANDATORY):

Progress file: /tmp/codex_progress_{task_id}.txt
Update on EVERY step:
echo \"[\$(date +%H:%M:%S)] [STATUS] Message\" >> /tmp/codex_progress_{task_id}.txt

Required status updates:
- [STARTED] Reading specifications
- [PROGRESS] Writing tests (X/Y done)
- [COMPLETED] Tests written (X tests)
- [STARTED] Implementing {Component}
- [PROGRESS] Implementation (X lines / method Y)
- [COMPLETED] Code complete (X lines)
- [STARTED] Running pytest
- [PROGRESS] Tests (X/Y passing)
- [COMPLETED] All tests GREEN (X/Y, Z% coverage)
- [FINISHED] Task complete

⚠️ FINAL REPORT (MANDATORY):

Report file: /tmp/codex_report_{task_id}.md

Format:
\`\`\`markdown
# Codex Implementation Report

**Component:** {ComponentName}
**Status:** ✅ SUCCESS / ❌ FAILED
**Duration:** Xm Ys
**Timestamp:** $(date +\"%Y-%m-%d %H:%M:%S\")

## Files Created
- path/file.py (X lines) - Purpose

## Test Results
- Tests written: X
- Tests passing: X/Y ({%})
- Coverage: {%}
- Test execution time: Xs

## Code Quality
- Type hints: ✅ YES / ❌ NO
- Docstrings: ✅ YES / ❌ NO
- Formatted (black/isort): ✅ YES / ❌ NO
- Follows spec: ✅ YES / ❌ NO

## Issues Encountered
1. Issue description
   - Resolution: How fixed
2. [Or: None]

## Ready For
- ✅ Code review by Claude Code
- ✅ Merge to main (after review)
- ❌ Needs fixes: [list issues]

## Notes
[Any additional context]
\`\`\`

---

NOW IMPLEMENT TASK:

{actual_task_description}

" --sandbox workspace-write
```

---

## 🔄 Claude Code Monitoring Loop

**Every 30 seconds:**

```bash
# Check progress
tail -3 /tmp/codex_progress_{task_id}.txt

# Show user:
"Codex: [13:05:30] [PROGRESS] Implementation (250/350 lines)"
```

**When completed:**

```bash
# Read final report
cat /tmp/codex_report_{task_id}.md

# Show user structured summary
```

**Cleanup:**

```bash
# After successful completion
rm /tmp/codex_progress_{task_id}.txt
rm /tmp/codex_report_{task_id}.md
```

---

## 📋 Task Types & Templates

### TASK TYPE 1: Implementation (TDD)

```bash
codex exec "
TASK: Implement {ComponentName} using TDD

SPEC: .spec-workflow/specs/level2/{component_name}_component.md

PROGRESS: /tmp/codex_progress_impl_{component}.txt
REPORT: /tmp/codex_report_impl_{component}.md

TDD WORKFLOW:
1. Read spec (update progress)
2. Write tests FIRST - tests/unit/test_{component}.py
3. Run pytest → RED (update progress)
4. Implement - src/{module}/{component}.py
5. Run pytest → GREEN (update progress)
6. Format (black, isort)
7. Write final report

REQUIREMENTS:
- Type hints 100%
- Docstrings (Google style)
- Follow spec exactly
- NO git commits

UPDATE PROGRESS & REPORT!
START.
" --sandbox workspace-write
```

---

### TASK TYPE 2: Code Review

```bash
codex exec "
TASK: Code Review for {ComponentName}

FILES TO REVIEW:
- {file1}
- {file2}

PROGRESS: /tmp/codex_progress_review_{component}.txt
REPORT: /tmp/codex_report_review_{component}.md

REVIEW CHECKLIST:
1. Read files (update progress)
2. Check spec compliance
3. Check code quality (type hints, docstrings, DRY)
4. Check security (validation, injection, errors)
5. Check tests (coverage, edge cases)
6. Write review report

REPORT FORMAT:
## Code Review Report

Component: {Name}
Reviewer: Codex CLI (GPT-5.1)

### ✅ APPROVED
[what's good]

### ⚠️ WARNINGS
[non-critical issues]

### ❌ BLOCKING
[must fix before merge]

VERDICT: APPROVE / REQUEST_CHANGES / REJECT

UPDATE PROGRESS & REPORT!
START REVIEW.
" --sandbox read-only
```

---

### TASK TYPE 3: Fix Issues

```bash
codex exec "
TASK: Fix issues in {ComponentName}

ISSUES (from code review):
1. {issue_description}
2. {issue_description}

PROGRESS: /tmp/codex_progress_fix_{component}.txt
REPORT: /tmp/codex_report_fix_{component}.md

WORKFLOW:
1. Read files
2. Fix each issue (update progress per issue)
3. Run tests
4. Report what fixed

UPDATE PROGRESS & REPORT!
START FIXES.
" --sandbox workspace-write
```

---

## 🎯 Standard Workflow (Sequential)

```
1. IMPLEMENT
   └─ Codex: Create tests + code
   └─ Claude: Monitor progress
   └─ Result: /tmp/codex_report_impl_X.md

2. REVIEW
   └─ Codex: Review code
   └─ Claude: Parse review report
   └─ Result: /tmp/codex_report_review_X.md

3. FIX (if needed)
   └─ Codex: Fix issues
   └─ Claude: Verify fixes
   └─ Result: /tmp/codex_report_fix_X.md

4. MERGE
   └─ Claude: Git commit + push (after user approval)
```

---

## 📈 Monitoring Best Practices

### DO:
- ✅ Monitor progress file every 30 seconds
- ✅ Show user latest 3 lines from progress
- ✅ Read final report when complete
- ✅ Check files created (git status)
- ✅ Cleanup /tmp files after success

### DON'T:
- ❌ Say "будет мониторить" and forget
- ❌ Use BashOutput (context pollution)
- ❌ Check too frequently (< 30 sec)
- ❌ Leave /tmp files after completion

---

## 🔧 Error Handling

### If Codex Hangs (no progress updates > 2 min):

```bash
# Check if stuck
tail -1 /tmp/codex_progress_{task}.txt
# If timestamp > 2 min old:

# Kill Codex
pkill -f "codex exec.*{task_id}"

# Report to user
"Codex timeout detected - killed process"
```

### If Codex Fails:

```bash
# Read report
cat /tmp/codex_report_{task}.md | grep "Status:"
# If FAILED:

# Show user the issues
cat /tmp/codex_report_{task}.md | grep -A 10 "Issues Encountered"

# Decide: retry or escalate
```

---

## 📚 Example: Full Implementation Cycle

```bash
# STEP 1: Delegate implementation to Codex
codex exec "
PROGRESS: /tmp/codex_progress_impl_ollama.txt
REPORT: /tmp/codex_report_impl_ollama.md

TASK: Implement OllamaEmbedder
SPEC: .spec-workflow/specs/level2/ollama_embedder_component.md

TDD: tests first, then code
UPDATE PROGRESS ON EVERY STEP!
" --sandbox workspace-write &

# Claude monitors (every 30s):
tail -3 /tmp/codex_progress_impl_ollama.txt
# Shows: "[13:10:00] [PROGRESS] Tests (15/20 written)"

# STEP 2: When complete, review
codex exec "
PROGRESS: /tmp/codex_progress_review_ollama.txt
REPORT: /tmp/codex_report_review_ollama.md

REVIEW: src/zapomni_core/embeddings/ollama_embedder.py
CHECK: spec compliance, quality, security, tests
" --sandbox read-only &

# Claude monitors review

# STEP 3: If issues, fix
codex exec "
PROGRESS: /tmp/codex_progress_fix_ollama.txt
REPORT: /tmp/codex_report_fix_ollama.md

FIX: {issues from review}
" --sandbox workspace-write &

# STEP 4: Claude commits (after user approval)
git add src/zapomni_core/embeddings/
git commit -m "feat(embeddings): Add OllamaEmbedder component"
```

---

## 🎓 Quick Reference Card

| Task | Sandbox Mode | Progress File | Report File |
|------|--------------|---------------|-------------|
| Implement | workspace-write | /tmp/codex_progress_impl_{X}.txt | /tmp/codex_report_impl_{X}.md |
| Review | read-only | /tmp/codex_progress_review_{X}.txt | /tmp/codex_report_review_{X}.md |
| Fix | workspace-write | /tmp/codex_progress_fix_{X}.txt | /tmp/codex_report_fix_{X}.md |
| Analyze | read-only | /tmp/codex_progress_analyze_{X}.txt | /tmp/codex_report_analyze_{X}.md |

**Monitoring command:**
```bash
tail -f /tmp/codex_progress_*.txt
```

---

## ✅ Checklist Before Codex Call

- [ ] Spec file exists and complete
- [ ] Dependencies installed (pip install done)
- [ ] Git branch created (feature/codex-{task})
- [ ] Progress file path in prompt
- [ ] Report file path in prompt
- [ ] Correct sandbox mode (workspace-write for implementation)
- [ ] Monitoring plan (tail every 30s)

---

## 🚀 Benefits

1. **Real-time visibility** - See what Codex doing NOW
2. **Structured reporting** - Machine-readable summaries
3. **No context pollution** - Progress in files, not chat
4. **Easy debugging** - Progress log shows where stuck
5. **Automated workflow** - Consistent task → review → fix cycle

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Project:** Zapomni

*This protocol enables efficient Claude Code + Codex CLI coordination with full visibility and structured communication.*
