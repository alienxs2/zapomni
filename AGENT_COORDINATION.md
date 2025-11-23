# Agent Coordination & Management

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Date:** 2025-11-22

Related: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md) | [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md)

---

## 📋 Overview

Этот документ описывает как координировать до 5 параллельных AI агентов для разработки Zapomni, включая:
- Типы агентов и их роли
- Prompt templates для каждого типа
- Стратегии параллелизации
- Управление зависимостями
- Dashboard integration
- Troubleshooting и recovery

### Why Agent Coordination Matters

**Challenges:**
- Multiple agents working simultaneously
- Avoiding conflicts (git, files, resources)
- Ensuring consistency across outputs
- Managing dependencies (sequential vs parallel)
- Timeout handling and retries

**Solutions:**
- Clear agent roles and responsibilities
- Standardized prompt templates
- Dependency mapping
- Real-time monitoring via dashboard
- Automated conflict resolution

---

## 🤖 Agent Types & Roles

### 1. Spec Creation Agents

**Purpose:** Create specification documents at all 3 levels

**Responsibilities:**
- Read steering documents (product.md, tech.md, structure.md)
- Create spec documents following SPEC_METHODOLOGY.md templates
- Ensure completeness and consistency
- Follow naming conventions

**Trigger:** Manual (launched by coordinator for each spec batch)

**Execution Mode:** Can run in parallel (up to 5 agents)

**Example Tasks:**
- "Create module-level spec for zapomni_mcp"
- "Create component specs for MCP tools (batch of 3)"
- "Create function specs for search_memory module (batch of 5)"

**Output:**
- Markdown files in `.spec-workflow/specs/levelN/`
- Status report with completion metrics

**Dependencies:**
- Must have steering documents available
- For Level 2/3: requires Level 1/2 specs completed

**Prompt Template:** See Section "Spec Creation Prompt Template"

---

### 2. Verification Agents

**Purpose:** Multi-agent verification of specifications

**Responsibilities:**
- Read assigned specs (overlapping pattern)
- Check internal consistency
- Check cross-document consistency
- Check steering alignment
- Check technical feasibility
- Generate detailed verification report

**Trigger:** After spec creation batch completes

**Execution Mode:** Always 5 agents in parallel (overlapping verification)

**Example Tasks:**
- "Agent 1: Verify specs [1, 2, 3]"
- "Agent 2: Verify specs [3, 4, 5]"
- "Agent 3: Verify specs [5, 6, 7]"
- "Agent 4: Verify specs [2, 4, 6]"
- "Agent 5: Verify specs [1, 3, 5, 7]"

**Output:**
- Verification report in `verification_reports/levelN/agentN_report.md`
- Issues categorized: CRITICAL, WARNING, APPROVED

**Dependencies:**
- Requires specs to be created
- All 5 agents must complete before synthesis

**Prompt Template:** See Section "Verification Prompt Template"

---

### 3. Synthesis Agent

**Purpose:** Aggregate and analyze 5 verification reports

**Responsibilities:**
- Read all 5 verification reports
- Identify patterns (multiple agents found same issue)
- Categorize confirmed vs unique issues
- Prioritize issues by severity
- Create consolidated synthesis report

**Trigger:** After all 5 verification agents complete

**Execution Mode:** Single agent, sequential

**Example Task:**
- "Synthesize verification reports for Level 1 specs"

**Output:**
- Synthesis report: `verification_reports/levelN/synthesis_report.md`
- Decision: APPROVE / APPROVE_WITH_WARNINGS / REJECT

**Dependencies:**
- Requires all 5 verification reports
- Blocks reconciliation until complete

**Prompt Template:** See Section "Synthesis Prompt Template"

---

### 4. Reconciliation Agent

**Purpose:** Validate synthesis recommendations against steering documents

**Responsibilities:**
- Read synthesis report
- Check recommendations align with product.md (vision)
- Check technical soundness per tech.md (stack)
- Check conventions per structure.md (patterns)
- Adjust recommendations if needed
- Create reconciliation report

**Trigger:** After synthesis completes

**Execution Mode:** Single agent, sequential

**Example Task:**
- "Reconcile synthesis report with steering documents"

**Output:**
- Reconciliation report: `verification_reports/levelN/reconciliation_report.md`
- Validated recommendations list

**Dependencies:**
- Requires synthesis report
- Requires steering documents
- Blocks refinement until complete

**Prompt Template:** See Section "Reconciliation Prompt Template"

---

### 5. Refinement Agent

**Purpose:** Apply approved fixes to specs

**Responsibilities:**
- Read reconciliation report
- Edit affected spec documents
- Track all changes made
- Prepare changed specs for re-verification
- Generate refinement summary

**Trigger:** After reconciliation approves fixes

**Execution Mode:** Single agent, sequential

**Example Task:**
- "Refine specs based on reconciliation recommendations"

**Output:**
- Updated spec files
- Refinement report: `verification_reports/levelN/refinement_report.md`
- List of changed files for re-verification

**Dependencies:**
- Requires reconciliation report with approved fixes
- Requires write access to spec files

**Prompt Template:** See Section "Refinement Prompt Template"

---

### 6. Implementation Agents (TDD)

**Purpose:** Write tests and code for features

**Responsibilities:**
- Read function-level specs
- Write tests FIRST (RED phase)
- Write minimal code (GREEN phase)
- Refactor code (REFACTOR phase)
- Commit to feature branch
- Create PR

**Trigger:** After all specs approved, during implementation phase

**Execution Mode:** Parallel (up to 5 agents, feature-based)

**Example Tasks:**
- "Agent 1: Implement add_memory feature (tests + code)"
- "Agent 2: Implement search_memory feature (tests + code)"
- "Agent 3: Implement get_stats feature (tests + code)"

**Output:**
- Test files in `tests/`
- Source code in `src/`
- Git commits on feature branches
- Pull request created

**Dependencies:**
- Requires function-level specs
- Requires foundation code (db_client, utils)
- Each feature may depend on specific modules

**Prompt Template:** See Section "Implementation Prompt Template"

---

### 7. Code Review Agents

**Purpose:** Review code quality, security, tests

**Responsibilities:**
- Read changed files in PR
- Check code quality (style, DRY, readability)
- Check tests (coverage, edge cases)
- Check security (injections, validation)
- Check alignment with specs
- Generate review report

**Trigger:** After implementation agent creates PR

**Execution Mode:** One review agent per PR (sequential reviews)

**Example Task:**
- "Review PR #15 (add_memory feature)"

**Output:**
- Code review report
- Verdict: APPROVE / REQUEST_CHANGES / REJECT

**Dependencies:**
- Requires PR to be created
- Blocks merge until approved

**Prompt Template:** See Section "Code Review Prompt Template"

---

## 📝 Prompt Templates (Complete)

### Spec Creation Prompt Template

```markdown
ЗАДАЧА: Создать спецификацию {level}-level для {component_name}

КОНТЕКСТ ПРОЕКТА:
- Проект: Zapomni - local-first MCP memory system
- Author: Goncharenko Anton aka alienxs2
- License: MIT
- GitHub: https://github.com/alienxs2/zapomni

STEERING DOCUMENTS (прочитать обязательно):
1. /home/dev/zapomni/.spec-workflow/steering/product.md
2. /home/dev/zapomni/.spec-workflow/steering/tech.md
3. /home/dev/zapomni/.spec-workflow/steering/structure.md

{if level > 1}
PARENT SPECIFICATIONS (прочитать обязательно):
{parent_spec_files}
{endif}

МЕТОДОЛОГИЯ:
Прочитай: /home/dev/zapomni/SPEC_METHODOLOGY.md
Используй соответствующий template для {level}-level spec.

СОЗДАТЬ ФАЙЛ: `.spec-workflow/specs/{level}/{component_name}.md`

ТРЕБОВАНИЯ:

1. **Completeness:**
   - Все секции template заполнены
   - {if level==1: API interfaces defined}
   - {if level==2: All public methods documented}
   - {if level==3: All edge cases (min 3) + test scenarios (min 5)}

2. **Consistency:**
   - Terminology совпадает с parent specs
   - Alignment со steering documents
   - Data types consistent

3. **Quality:**
   - Examples provided
   - Design decisions explained
   - Non-functional requirements specified

ФОРМАТ ОТЧЁТА:

## SPEC CREATED

**File:** `.spec-workflow/specs/{level}/{component_name}.md`
**Lines:** [count] lines
**Size:** [words] words

**Sections Completed:**
- ✅ Overview
- ✅ {level-specific sections}
- ✅ References

**Key Decisions:**
1. {decision 1}
2. {decision 2}

**Dependencies Identified:**
- {dependency 1}
- {dependency 2}

**Ready for Verification:** Yes

---

НАЧИНАЙ СОЗДАНИЕ СПЕЦИФИКАЦИИ.
```

---

### Verification Prompt Template

```markdown
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
```

---

### Synthesis Prompt Template

```markdown
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
```

---

### Reconciliation Prompt Template

```markdown
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
```

---

### Refinement Prompt Template

```markdown
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
```

---

### Implementation Prompt Template

```markdown
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
```

---

### Code Review Prompt Template

```markdown
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
```

---

## 🔄 Parallelization Strategies

### Rule: Max 5 Agents Concurrent

Система Claude Code поддерживает до 5 параллельных агентов одновременно.

### When To Parallelize

**✅ GOOD - Independent Tasks:**
```
# Spec creation - documents independent
Agent 1: Create spec for module A
Agent 2: Create spec for module B
Agent 3: Create spec for module C

# Verification - always 5 agents
Agents 1-5: Verify overlapping document sets

# Features - isolated features
Agent 1: Implement add_memory
Agent 2: Implement search_memory
Agent 3: Implement get_stats
```

**❌ BAD - Dependent Tasks:**
```
# Sequential dependency
Agent 1: Create module spec
Agent 2: Create component spec (NEEDS module spec done first)

# Shared resource conflict
Agent 1: Edit file.py
Agent 2: Edit file.py (GIT CONFLICT!)
```

### Dependency Management

**Foundation First:**
```
Wave 0 (Sequential):
  Agent 1: Setup infrastructure (db_client, utils)

Wave 1 (Parallel):
  Agent 1: Feature A (uses foundation)
  Agent 2: Feature B (uses foundation)
  Agent 3: Feature C (uses foundation)
```

**Feature Dependencies:**
```
Wave 1 (Parallel):
  Agent 1: search_memory (basic)
  Agent 2: add_memory (independent)
  Agent 3: get_stats (independent)

Wave 2 (After Wave 1 merged):
  Agent 4: hybrid_search (depends on search_memory)
  Agent 5: caching (depends on search_memory)
```

### Git Branch Strategy

**Avoid Conflicts:**
```
main
├── foundation/db-client (Agent 1, merged)
├── foundation/utils (Agent 1, merged)
├── feature/add-memory (Agent 2, active)
├── feature/search (Agent 3, active)
└── feature/stats (Agent 4, active)
```

**Merge Order:**
1. Foundation first
2. Independent features (any order)
3. Dependent features (after dependencies merged)

---

## 📊 Dashboard Integration

### Real-Time Monitoring

**Dashboard Backend (FastAPI) tracks:**
- Agent status (idle, working, completed, failed)
- Current task description
- Progress percentage (if available)
- Duration (how long agent has been running)
- Logs (real-time via WebSocket)

**Dashboard Frontend displays:**
```
┌─────────┬──────────┬────────────────────────┬──────────┬──────────┐
│ Agent   │ Status   │ Current Task           │ Progress │ Duration │
├─────────┼──────────┼────────────────────────┼──────────┼──────────┤
│ Agent 1 │ Working  │ Create module spec     │ 65%      │ 4m 23s   │
│ Agent 2 │ Idle     │ -                      │ -        │ -        │
│ Agent 3 │ Working  │ Verification check     │ 85%      │ 7m 12s   │
│ Agent 4 │ Failed   │ Synthesis failed       │ -        │ 9m 45s ⚠️│
│ Agent 5 │ Completed│ Component spec         │ 100%     │ 6m 34s ✅│
└─────────┴──────────┴────────────────────────┴──────────┴──────────┘

Alerts:
⚠️ Agent 4 failed - needs attention
⏰ Agent 3 approaching timeout (10min limit)
```

### Status Updates

**Agent reports status via:**
- File writes (status files in `.agent_status/`)
- Dashboard API calls (HTTP POST to `/api/agents/{id}/status`)
- Log streaming (WebSocket to `/ws/agents/{id}/logs`)

**Coordinator (Claude Code) updates dashboard:**
```python
# When launching agent
dashboard.update_agent_status(
    agent_id="agent_1",
    status="working",
    task="Create module spec for zapomni_mcp",
    started_at=datetime.now()
)

# When agent completes
dashboard.update_agent_status(
    agent_id="agent_1",
    status="completed",
    task="Create module spec for zapomni_mcp",
    completed_at=datetime.now(),
    result="success"
)
```

---

## 🔧 Troubleshooting & Recovery

### Common Issues

#### Issue 1: Agent Timeout

**Symptom:** Agent takes > 10 minutes

**Causes:**
- Task too complex
- Network issues
- Infinite loop in processing

**Recovery:**
1. Kill agent (timeout automatic)
2. Review task complexity
3. Break into smaller sub-tasks
4. Retry with adjusted prompt

#### Issue 2: Agent Returns Incomplete Output

**Symptom:** Output file missing sections

**Causes:**
- Token limit reached
- Misunderstood prompt
- Missing context

**Recovery:**
1. Review agent output
2. Identify missing parts
3. Create補充 task: "Complete sections X, Y, Z"
4. Merge outputs

#### Issue 3: Git Merge Conflict

**Symptom:** Two agents modified same file

**Causes:**
- Poor parallelization planning
- Shared code not in foundation

**Recovery:**
1. Identify conflicting changes
2. Launch reconciliation agent:
   ```
   TASK: Resolve merge conflict
   Files: file.py
   Branch A changes: [...]
   Branch B changes: [...]
   Resolve conflict maintaining both intents.
   ```
3. Manual review if complex

#### Issue 4: Verification Infinite Loop

**Symptom:** After 3 refinement cycles, still have critical issues

**Causes:**
- Fundamental spec problem
- Conflicting requirements

**Recovery:**
1. Escalate to user
2. User decides:
   - Accept specs as-is
   - Manual intervention
   - Pivot approach

### Retry Strategy

**Automatic Retry (1x):**
```python
try:
    result = launch_agent(task)
except AgentTimeout:
    # Retry once with same prompt
    result = launch_agent(task, retry=True)
except AgentError:
    # Escalate to user
    notify_user(error)
```

**Manual Retry:**
- User reviews error
- Adjusts approach
- Re-launches manually

---

## 📚 Best Practices

### 1. Clear Task Descriptions

**❌ Vague:**
```
"Fix the specs"
```

**✅ Specific:**
```
"Refine zapomni_mcp_module.md:
- Change process() signature from str to dict[str, Any]
- Add example in Section 5
- Update cross-reference to zapomni_core in Section 3"
```

### 2. Provide Full Context

**Always include:**
- Project context (Zapomni, MIT, author)
- Steering documents to read
- Parent specs (for Level 2/3)
- Expected output format

### 3. One Agent, One Responsibility

**❌ Don't:**
```
"Create spec AND verify it AND refine it"
```

**✅ Do:**
```
Agent 1: "Create spec"
Agent 2: "Verify spec"
Agent 3: "Refine spec"
```

### 4. Monitor Progress

- Check dashboard every 2-3 minutes
- If agent > 8 minutes, prepare for timeout
- Review logs for errors

### 5. Celebrate Completions

- Update todo list immediately
- Log metrics (lines, time, quality)
- Share progress with user

---

## 📖 References

- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Overall plan
- [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md) - Spec creation details
- [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) - Operational guidelines

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**GitHub:** https://github.com/alienxs2/zapomni

*This document defines agent coordination for Zapomni development.*
