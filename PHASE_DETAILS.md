# Phase-by-Phase Implementation Guide

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Date:** 2025-11-23

Related: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md) | [AGENT_COORDINATION.md](AGENT_COORDINATION.md) | [QUALITY_FRAMEWORK.md](QUALITY_FRAMEWORK.md)

---

## 📋 Overview

Этот документ предоставляет детальный пошаговый guide для каждой из 10 фаз разработки Zapomni (Phase 0-9).

Для каждой фазы описаны:
- Цели и deliverables
- Пошаговый процесс выполнения
- Конкретные команды и действия для агентов
- Quality gate критерии
- Expected timeline
- Troubleshooting

**Total Timeline:** ~8 weeks
**Total Phases:** 10 (Phase 0-9)

---

## 🚀 Phase 0: Setup & Preparation

**Status:** ✅ CURRENT (in progress)
**Duration:** 1-2 days
**Agent Invocations:** ~5-10

### Goals

Создать foundation для всей разработки:
- Planning documentation complete (5 docs)
- Project structure setup
- Steering documents updated
- Git initialized
- Dashboard ready (optional)

### Deliverables

✅ 5 planning documents:
1. DEVELOPMENT_PLAN.md
2. SPEC_METHODOLOGY.md
3. AGENT_COORDINATION.md
4. QUALITY_FRAMEWORK.md
5. PHASE_DETAILS.md (this document)

✅ Updated steering documents (MIT license, correct author)
✅ MIT LICENSE file
✅ Project directory structure
✅ Git repository initialized
✅ Agent prompt templates created
✅ Dashboard setup (basic)

### Step-by-Step Process

#### Step 1: Complete Planning Documents ✅ DONE

**Agent Tasks:**
- Create 5 planning docs (all completed)

**Completion Criteria:**
- ✅ All 5 docs created and saved
- ✅ Each doc >= 3000 words
- ✅ Comprehensive coverage

**Status:** Complete

---

#### Step 2: Update Steering Documents

**Agent Task:**
```
TASK: Update steering documents with MIT license and author

FILES TO UPDATE:
- .spec-workflow/steering/product.md
- .spec-workflow/steering/tech.md
- .spec-workflow/steering/structure.md

CHANGES:
1. Replace "Tony" → "Goncharenko Anton aka alienxs2" (everywhere)
2. Replace "Apache 2.0" → "MIT" (in license sections)
3. Add/update copyright notice: "Copyright (c) 2025 Goncharenko Anton aka alienxs2"
4. Ensure consistency across all 3 files

VERIFICATION:
- Search for "Tony" - should find 0 results
- Search for "Apache" - should find 0 results
- Search for "Goncharenko Anton aka alienxs2" - should find in all 3 files
```

**Expected Output:**
- 3 updated files
- Verification report

**Commands:**
```bash
# Agent will use Edit tool to update files
# Then verify changes:
grep -r "Tony" .spec-workflow/steering/
# Should return nothing

grep -r "Apache" .spec-workflow/steering/
# Should return nothing (or only in context, not as license)

grep -r "Goncharenko Anton" .spec-workflow/steering/
# Should return matches in all 3 files
```

**Duration:** 15-20 minutes

---

#### Step 3: Create LICENSE File

**Agent Task (or manual):**
```
CREATE FILE: /home/dev/zapomni/LICENSE

CONTENT:
```
MIT License

Copyright (c) 2025 Goncharenko Anton aka alienxs2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
```

**Verification:**
```bash
cat /home/dev/zapomni/LICENSE
# Should show MIT license with correct author
```

**Duration:** 5 minutes

---

#### Step 4: Setup Project Directory Structure

**Commands:**
```bash
cd /home/dev/zapomni

# Create spec directories
mkdir -p .spec-workflow/specs/level1
mkdir -p .spec-workflow/specs/level2
mkdir -p .spec-workflow/specs/level3

# Create source directories
mkdir -p src/zapomni_mcp/tools
mkdir -p src/zapomni_mcp/schemas
mkdir -p src/zapomni_core/processors
mkdir -p src/zapomni_core/embeddings
mkdir -p src/zapomni_core/search
mkdir -p src/zapomni_core/extractors
mkdir -p src/zapomni_core/chunking
mkdir -p src/zapomni_core/tasks
mkdir -p src/zapomni_core/utils
mkdir -p src/zapomni_db/falkordb
mkdir -p src/zapomni_db/redis_cache

# Create test directories
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e
mkdir -p tests/performance
mkdir -p tests/fixtures

# Create verification reports directory
mkdir -p verification_reports/level1
mkdir -p verification_reports/level2
mkdir -p verification_reports/level3

# Create docs directory
mkdir -p docs/api
mkdir -p docs/guides
mkdir -p docs/architecture
mkdir -p docs/benchmarks

# Create scripts directory
mkdir -p scripts

# Create dashboard directories (optional)
mkdir -p dashboard/backend
mkdir -p dashboard/frontend

# Create prompts directory
mkdir -p .prompts

# Create .gitkeep files in empty directories
find . -type d -empty -exec touch {}/.gitkeep \;
```

**Verify:**
```bash
tree -L 3 -d /home/dev/zapomni
# Should show all created directories
```

**Duration:** 10 minutes

---

#### Step 5: Create Agent Prompt Templates

**Agent Task:**
```
CREATE FILES in .prompts/ directory:

1. .prompts/spec_creation_template.md
2. .prompts/verification_template.md
3. .prompts/synthesis_template.md
4. .prompts/reconciliation_template.md
5. .prompts/refinement_template.md
6. .prompts/implementation_template.md
7. .prompts/code_review_template.md

CONTENT: Copy from AGENT_COORDINATION.md Section "Prompt Templates"

Each template should be:
- Standalone markdown file
- Ready to use (fill-in-the-blanks style)
- Clear placeholders for variables
```

**Source:** All templates are in AGENT_COORDINATION.md lines 269-989

**Verification:**
```bash
ls -la .prompts/
# Should show 7 template files

wc -l .prompts/*.md
# Each should have 50-200 lines
```

**Duration:** 20 minutes

---

#### Step 6: Git Initialization & First Commit

**Commands:**
```bash
cd /home/dev/zapomni

# Check if git already initialized
if [ ! -d .git ]; then
  git init
fi

# Create .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDE
.vscode/
.idea/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# Build
dist/
build/
*.egg-info/

# Dashboard
dashboard/frontend/node_modules/
dashboard/frontend/dist/

# Agent status (temporary files)
.agent_status/

# OS
.DS_Store
Thumbs.db

# Keep empty directories
!.gitkeep
EOF

# Add all files
git add .

# Initial commit
git commit -m "chore: Initial project setup - Phase 0 complete

- Planning documentation (5 comprehensive docs)
- Steering documents updated (MIT license, Goncharenko Anton aka alienxs2)
- MIT LICENSE file created
- Project structure complete (src/, tests/, docs/, .spec-workflow/)
- Agent prompt templates (7 templates)
- Git initialized with .gitignore

Phase 0 deliverables complete ✅

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Check remote
git remote -v
# If not set, add:
# git remote add origin https://github.com/alienxs2/zapomni.git

# Push (if remote exists and user wants to)
# git push -u origin main
```

**Duration:** 10 minutes

---

#### Step 7: User Review & Approval ⏸️ CHECKPOINT

**Checkpoint:**
- User reviews all planning documents (5 files)
- User reviews project structure
- User reviews steering document updates
- User approves to proceed to Phase 1

**User Action Required:**
```
User says: "Phase 0 approved" or "Proceed to Phase 1"
```

**What to Review:**
1. **Planning Documents:**
   - DEVELOPMENT_PLAN.md - comprehensive?
   - SPEC_METHODOLOGY.md - clear process?
   - AGENT_COORDINATION.md - good templates?
   - QUALITY_FRAMEWORK.md - realistic standards?
   - PHASE_DETAILS.md - helpful guide?

2. **Project Structure:**
   - Directory layout makes sense?
   - All needed folders created?

3. **Steering Documents:**
   - Author updated everywhere?
   - License changed to MIT?

**Duration:** User decides (15-30 minutes review time)

---

### Quality Gate: Phase 0 Complete

**Checklist:**
- ✅ All 5 planning documents created
- ✅ Steering documents updated
- ✅ LICENSE file created
- ✅ Project structure complete
- ✅ Git initialized
- ✅ Agent templates created
- ✅ User approval obtained

**Next Phase:** Phase 1 (Module-Level Specs)

**Transition:** Once user approves, begin Phase 1 immediately.

---

## 📝 Phase 1: Module-Level Specifications

**Duration:** 2-3 days
**Agent Invocations:** ~14-20
**Quality Gate:** Gate 1 (Module Specs Approved)

### Goals

Create 7 high-level module specifications that define architecture and boundaries.

### Deliverables

7 module-level specs in `.spec-workflow/specs/level1/`:
1. zapomni_mcp_module.md
2. zapomni_core_module.md
3. zapomni_db_module.md
4. cross_module_interfaces.md
5. data_flow_architecture.md
6. error_handling_strategy.md
7. configuration_management.md

### Step-by-Step Process

#### Step 1: Create Module Spec Plan

**Before Starting:**
```
Review:
- product.md (features we need to support)
- tech.md (architecture decisions)
- structure.md (module organization)

Identify modules needed:
1. zapomni_mcp - MCP protocol adapter
2. zapomni_core - processing engine
3. zapomni_db - database clients
4. Plus 4 cross-cutting docs
```

**Agent Task:**
```
TASK: Plan module breakdown

Read steering documents and create list of:
- Main modules (3-4)
- Cross-cutting concerns (3-4)

Total: 7 module-level specs

Output: Plan document listing all 7 specs with 1-sentence purpose each
```

**Duration:** 15 minutes

---

#### Step 2: Create Module Specs (Sequential or 2-3 Parallel)

**Option A: Sequential (safer, slower)**
```
Agent 1: Create zapomni_mcp_module.md
  Wait for completion
Agent 2: Create zapomni_core_module.md
  Wait for completion
Agent 3: Create zapomni_db_module.md
  ...etc
```

**Option B: Parallel (faster, requires coordination)**
```
Agent 1: Create zapomni_mcp_module.md
Agent 2: Create zapomni_core_module.md
Agent 3: Create zapomni_db_module.md

(Launch 3 agents in parallel)

After completion, sequential:
Agent 4: Create cross_module_interfaces.md (needs 1-3 done)
Agent 5: Create data_flow_architecture.md
Agent 6: Create error_handling_strategy.md
Agent 7: Create configuration_management.md
```

**Agent Prompt (use template from .prompts/spec_creation_template.md):**

```markdown
ЗАДАЧА: Создать спецификацию Level 1 (Module) для zapomni_mcp

КОНТЕКСТ ПРОЕКТА:
- Проект: Zapomni - local-first MCP memory system
- Author: Goncharenko Anton aka alienxs2
- License: MIT
- GitHub: https://github.com/alienxs2/zapomni

STEERING DOCUMENTS (прочитать обязательно):
1. /home/dev/zapomni/.spec-workflow/steering/product.md
2. /home/dev/zapomni/.spec-workflow/steering/tech.md
3. /home/dev/zapomni/.spec-workflow/steering/structure.md

МЕТОДОЛОГИЯ:
Прочитай: /home/dev/zapomni/SPEC_METHODOLOGY.md
Используй template для Level 1 (Module) spec из Section "Level 1: Module-Level Specifications"

СОЗДАТЬ ФАЙЛ: `.spec-workflow/specs/level1/zapomni_mcp_module.md`

ТРЕБОВАНИЯ:
1. **Completeness:** Все секции template заполнены
2. **API Interfaces:** Определены основные классы и их public API
3. **Dependencies:** Все зависимости перечислены
4. **Design Decisions:** Ключевые решения задокументированы
5. **Length:** 1500-2500 words

ФОРМАТ ОТЧЁТА:
## SPEC CREATED
**File:** `.spec-workflow/specs/level1/zapomni_mcp_module.md`
**Lines:** [count] lines
**Size:** [words] words
**Sections Completed:** [list]
**Ready for Verification:** Yes
```

**Repeat for each module** (adjust module name and file path)

**Expected Time per Module:** 30-45 minutes
**Total for 7 modules:** 3-5 hours (depends on parallel vs sequential)

---

#### Step 3: Verification (Always 5 Agents in Parallel)

**Overlapping Verification Pattern:**

**Agent Assignment:**
```
Agent 1: Verify specs [1, 2, 3]
Agent 2: Verify specs [3, 4, 5]
Agent 3: Verify specs [5, 6, 7]
Agent 4: Verify specs [2, 4, 6]
Agent 5: Verify specs [1, 3, 5, 7]
```

**Coverage:**
```
Spec 1: Agents 1, 5 → 2x
Spec 2: Agents 1, 4 → 2x
Spec 3: Agents 1, 2, 5 → 3x ⭐ (highest coverage)
Spec 4: Agents 2, 4 → 2x
Spec 5: Agents 2, 3, 5 → 3x ⭐
Spec 6: Agents 3, 4 → 2x
Spec 7: Agents 3, 5 → 2x
```

**Launch 5 Agents Simultaneously:**

Use prompt template from `.prompts/verification_template.md`

**Agent 1 Prompt:**
```markdown
ЗАДАЧА: Multi-document verification

ASSIGNED DOCUMENTS:
- .spec-workflow/specs/level1/zapomni_mcp_module.md
- .spec-workflow/specs/level1/zapomni_core_module.md
- .spec-workflow/specs/level1/zapomni_db_module.md

VERIFICATION CHECKLIST:
[Full checklist from template]

Save report to: `verification_reports/level1/agent1_report.md`
```

**Repeat for Agents 2-5** with their assigned documents

**Expected Time:** 1-1.5 hours (parallel execution)

---

#### Step 4: Synthesis

**Agent Task:**
```markdown
ЗАДАЧА: Synthesize 5 verification reports

INPUT FILES:
- verification_reports/level1/agent1_report.md
- verification_reports/level1/agent2_report.md
- verification_reports/level1/agent3_report.md
- verification_reports/level1/agent4_report.md
- verification_reports/level1/agent5_report.md

PROCESS:
1. Read all 5 reports
2. Identify patterns (2+ agents = confirmed issue)
3. Categorize: Confirmed Critical, Confirmed Warnings, Unique Issues
4. Aggregate approved aspects
5. Make decision: APPROVE / APPROVE_WITH_WARNINGS / REJECT

Save to: verification_reports/level1/synthesis_report.md
```

Use template: `.prompts/synthesis_template.md`

**Expected Time:** 30 minutes

**Expected Outcomes:**
- **Best Case:** 0 confirmed critical issues → APPROVE
- **Common Case:** 1-3 confirmed critical issues → REJECT, needs refinement
- **Worst Case:** 5+ issues → Major rework needed

---

#### Step 5: Reconciliation

**Agent Task:**
```markdown
ЗАДАЧА: Reconcile synthesis recommendations with steering documents

INPUT:
- verification_reports/level1/synthesis_report.md

STEERING DOCUMENTS:
- .spec-workflow/steering/product.md
- .spec-workflow/steering/tech.md
- .spec-workflow/steering/structure.md

PROCESS:
For each recommended fix:
- Check alignment with product.md (vision)
- Check technical soundness per tech.md
- Check conventions per structure.md
- Validate or adjust recommendation

Save to: verification_reports/level1/reconciliation_report.md
```

Use template: `.prompts/reconciliation_template.md`

**Expected Time:** 20 minutes

**Expected Outcomes:**
- Most recommendations ✅ APPROVED
- Some ⚠️ APPROVED WITH ADJUSTMENT
- Rare ❌ REJECTED (conflicts with vision)

---

#### Step 6: Refinement

**Agent Task:**
```markdown
ЗАДАЧА: Apply approved fixes to specs

INPUT:
- verification_reports/level1/reconciliation_report.md

PROCESS:
1. Read reconciliation report
2. Extract approved recommendations
3. Apply fixes to affected spec files
4. Track all changes
5. Identify which specs need re-verification

Save to: verification_reports/level1/refinement_report.md
```

Use template: `.prompts/refinement_template.md`

**Expected Time:** 30-45 minutes

**Iteration Limit:** Maximum 3 refinement cycles

**Cycle 1:** Initial verification → usually finds 2-5 issues
**Cycle 2:** Re-verification → should find 0-2 issues
**Cycle 3:** Final check → should find 0 issues

**If still issues after 3 cycles:** Escalate to user

---

#### Step 7: Re-Verification (If Needed)

**Decision:**
- Minor changes (typos, examples) → **SKIP** re-verification
- Major changes (API changes, new sections) → **RE-VERIFY**

**If Re-verification Needed:**
```
Only verify CHANGED specs
Use subset of agents (e.g., Agents 1, 2 who verified those docs)
Same process: Verification → Synthesis → Reconciliation → Refinement

Maximum 3 total cycles (including initial)
```

**Expected Time:** 1-2 hours (if needed)

---

#### Step 8: User Review & Approval ⏸️ CHECKPOINT

**User Reviews:**
1. Read 7 module specs
2. Check verification reports (synthesis, reconciliation)
3. Verify architecture makes sense
4. Approve or request changes

**User Action:**
```
User says: "Module specs approved" or "Approve Gate 1"
```

**What User Reviews:**
- Architecture sound?
- Modules well-separated?
- No circular dependencies?
- Matches product vision?
- Technical stack correct?

**Duration:** User decides (1-2 hours review time)

---

### Quality Gate 1: Module Specs Approved

**Criteria:**
- ✅ All 7 module specs created
- ✅ Zero CRITICAL issues in final reports
- ✅ Maximum 3 WARNING issues (non-blocking)
- ✅ 100% alignment with steering documents
- ✅ User approval obtained

**Metrics:**
- Documents consistency score: 100%
- Technical feasibility: Confirmed
- Coverage: All features from product.md represented

**Exit:** User approval → Proceed to Phase 2

---

### Troubleshooting Phase 1

#### Issue: Verification Finds Many Critical Issues (5+)

**Cause:** Specs too vague or contradictory

**Solution:**
1. Review steering documents - are they clear?
2. May need to refine product.md or tech.md first
3. Restart spec creation with clearer guidance

---

#### Issue: Circular Dependencies Detected

**Cause:** Poor module separation

**Solution:**
1. Introduce abstraction layer (Protocol, ABC)
2. Reorganize module responsibilities
3. May need to add/remove modules

Example:
```
Problem: Module A needs B, B needs A

Solution: Extract common interface to Module C
- A depends on C (interface)
- B implements C
- No circular dependency
```

---

#### Issue: Specs Don't Cover All Features from product.md

**Cause:** Incomplete module breakdown

**Solution:**
1. Create feature coverage matrix
2. Map each product.md feature to module
3. Add missing modules or expand existing

---

## 🔧 Phase 2: Component-Level Specifications

**Duration:** 4-5 days
**Agent Invocations:** ~40-50
**Quality Gate:** Gate 2

### Goals

Break down each module into components (classes/services) with detailed APIs.

### Deliverables

15-20 component-level specs in `.spec-workflow/specs/level2/`:

**Examples:**
- MCP Module (4 components):
  1. mcp_server_component.md
  2. add_memory_tool_component.md
  3. search_memory_tool_component.md
  4. get_stats_tool_component.md

- Core Module (8 components):
  1. text_processor_component.md
  2. ollama_embedder_component.md
  3. vector_search_component.md
  4. hybrid_search_component.md
  5. entity_extractor_component.md
  6. relationship_extractor_component.md
  7. semantic_cache_component.md
  8. task_manager_component.md

- DB Module (3 components):
  1. falkordb_client_component.md
  2. redis_cache_client_component.md
  3. schema_manager_component.md

**Total:** ~15-20 component specs

### Step-by-Step Process

#### Step 1: Component Breakdown Planning

**Agent Task:**
```
ЗАДАЧА: Create component breakdown for each module

INPUT:
- All 7 Level 1 module specs

PROCESS:
For each module:
1. Identify main responsibilities
2. Break into 3-5 components (classes/services)
3. Define each component's purpose (1 sentence)

OUTPUT:
Component breakdown document listing:
- Module name
- Component names (3-5 per module)
- Component purpose
- Parent module reference

Total components: 15-20
```

**Expected Time:** 30 minutes

**Example Output:**
```markdown
# Component Breakdown - Phase 2

## Module: zapomni_mcp

### Components (4):
1. **MCPServer** - Main MCP stdio server, tool routing
2. **AddMemoryTool** - Implements add_memory MCP tool
3. **SearchMemoryTool** - Implements search_memory MCP tool
4. **GetStatsTool** - Implements get_stats MCP tool

## Module: zapomni_core

### Components (8):
1. **TextProcessor** - Document chunking and preprocessing
2. **OllamaEmbedder** - Embedding generation via Ollama
3. **VectorSearch** - HNSW similarity search
4. **HybridSearch** - BM25 + vector fusion
5. **EntityExtractor** - SpaCy + LLM entity extraction
6. **RelationshipExtractor** - LLM relationship detection
7. **SemanticCache** - Redis-based embedding cache
8. **TaskManager** - Background async jobs

## Module: zapomni_db

### Components (3):
1. **FalkorDBClient** - Database operations wrapper
2. **RedisCacheClient** - Cache operations wrapper
3. **SchemaManager** - Database schema initialization

**Total:** 15 components
```

---

#### Step 2: Create Component Specs (Parallel Batches)

**Parallelization Strategy:**

**Batch 1 (3 agents):**
```
Agent 1: MCP components (4 specs)
Agent 2: Core components batch 1 (3 specs)
Agent 3: Core components batch 2 (3 specs)
```

**Batch 2 (2 agents):**
```
Agent 4: Core components batch 3 (2 specs)
Agent 5: DB components (3 specs)
```

**Agent Prompt (use template from .prompts/spec_creation_template.md):**

```markdown
ЗАДАЧА: Создать спецификацию Level 2 (Component) для AddMemoryTool

КОНТЕКСТ ПРОЕКТА:
- Проект: Zapomni
- Author: Goncharenko Anton aka alienxs2
- License: MIT

STEERING DOCUMENTS:
1. /home/dev/zapomni/.spec-workflow/steering/product.md
2. /home/dev/zapomni/.spec-workflow/steering/tech.md
3. /home/dev/zapomni/.spec-workflow/steering/structure.md

PARENT SPECIFICATION (обязательно прочитать):
- /home/dev/zapomni/.spec-workflow/specs/level1/zapomni_mcp_module.md

МЕТОДОЛОГИЯ:
Прочитай: /home/dev/zapomni/SPEC_METHODOLOGY.md
Используй template для Level 2 (Component) spec

СОЗДАТЬ ФАЙЛ: `.spec-workflow/specs/level2/add_memory_tool_component.md`

ТРЕБОВАНИЯ:
1. **Class Definition:** Full signature with all public methods
2. **Dependencies:** All component dependencies listed
3. **State Management:** Attributes and lifecycle documented
4. **Public Methods:** Each method fully documented
5. **Error Handling:** Exceptions defined
6. **Usage Examples:** Basic and advanced examples
7. **Length:** 1000-1500 words

ФОРМАТ ОТЧЁТА:
[Same as Level 1]
```

**Expected Time per Component:** 25-35 minutes
**Total for 15-20 components:** 6-10 hours (with parallelization: 2-3 days)

**Tips for Agents:**
- Read parent module spec carefully
- Ensure APIs match what module spec promised
- Define clear responsibilities (Single Responsibility Principle)
- Document all dependencies explicitly

---

#### Step 3: Verification (5 Agents, Overlapping)

**Same process as Level 1, but with 15-20 documents**

**Agent Assignment Example:**
```
Agent 1: Verify components [1, 2, 3, 4, 5]
Agent 2: Verify components [4, 5, 6, 7, 8]
Agent 3: Verify components [8, 9, 10, 11, 12]
Agent 4: Verify components [2, 5, 8, 11, 14]
Agent 5: Verify components [1, 4, 7, 10, 13, 15]
```

**Ensure Overlap:** Each doc verified by 2-3 agents

**Verification Focus:**
- ✅ APIs match module spec interfaces
- ✅ Dependencies are available (defined in other components or modules)
- ✅ No circular dependencies between components
- ✅ State management clear
- ✅ All public methods have signatures

**Expected Time:** 1.5-2 hours

---

#### Step 4-7: Synthesis → Reconciliation → Refinement → Re-verification

**Same process as Phase 1**

**Expected Issues:**
- Interface mismatches (component A expects X, component B provides Y)
- Missing dependencies
- Circular dependencies
- Vague method signatures

**Iteration Cycles:** 1-2 (components more concrete than modules, fewer ambiguities)

**Total Time for Verification Loop:** 3-5 hours

---

#### Step 8: User Review & Approval ⏸️ CHECKPOINT

**User Reviews:**
1. Sample 3-5 component specs (read in detail)
2. Skim others for completeness
3. Check dependency graph visualization (if available)
4. Verify no circular dependencies
5. Approve

**User Action:**
```
User says: "Component specs approved" or "Approve Gate 2"
```

**Duration:** 1-2 hours review time

---

### Quality Gate 2: Component Specs Approved

**Criteria:**
- ✅ All 15-20 component specs created
- ✅ Each component traceable to parent module
- ✅ Public API fully defined (all methods with signatures)
- ✅ Dependency graph validated (acyclic)
- ✅ User approval obtained

**Metrics:**
- API coverage: 100% of public methods documented
- Dependency graph: Valid (no cycles)
- Consistency score: 100%

**Exit:** User approval → Proceed to Phase 3

---

### Troubleshooting Phase 2

#### Issue: API Mismatch Between Components

**Example:**
```
ComponentA.process() returns str
ComponentB.handle() expects dict
```

**Solution:**
1. Identify correct type from business logic
2. Update one component's signature
3. Re-verify affected components

---

#### Issue: Circular Dependency Between Components

**Example:**
```
ComponentA depends on ComponentB
ComponentB depends on ComponentA
```

**Solution:**
1. Extract common interface/protocol
2. Both components depend on interface
3. Implement interface in one component
4. Inject via dependency injection

---

## 📋 Phase 3: Function-Level Specifications

**Duration:** 7-10 days
**Agent Invocations:** ~80-100
**Quality Gate:** Gate 3

### Goals

Document EVERY public function/method with maximum detail for TDD.

### Deliverables

40-50 function-level specs in `.spec-workflow/specs/level3/`:

**Examples:**
- add_memory_tool_execute_function.md
- search_memory_tool_execute_function.md
- text_processor_chunk_function.md
- ollama_embedder_generate_function.md
- vector_search_query_function.md
- ...etc (40-50 total)

### Step-by-Step Process

#### Step 1: Function Enumeration

**Agent Task:**
```
ЗАДАЧА: List all public functions from component specs

INPUT:
- All Level 2 component specs

PROCESS:
For each component:
1. Extract all public methods
2. Create function spec plan

OUTPUT:
Function list document with:
- Component name
- Function name
- Parameters (brief)
- Returns (brief)
- Purpose (1 sentence)

Total: 40-50 functions
```

**Expected Time:** 45 minutes

---

#### Step 2: Create Function Specs (Parallel Batches)

**Parallelization Strategy:**

**Wave 1 (5 agents):**
```
Agent 1: Functions 1-10
Agent 2: Functions 11-20
Agent 3: Functions 21-30
Agent 4: Functions 31-40
Agent 5: Functions 41-50
```

Each agent creates 8-10 function specs

**Agent Prompt:**
```markdown
ЗАДАЧА: Создать Level 3 (Function) spec для AddMemoryTool.execute

КОНТЕКСТ ПРОЕКТА: [same as before]

PARENT SPECIFICATION:
- .spec-workflow/specs/level2/add_memory_tool_component.md

МЕТОДОЛОГИЯ:
Используй template для Level 3 (Function) spec

СОЗДАТЬ ФАЙЛ: `.spec-workflow/specs/level3/add_memory_tool_execute_function.md`

ТРЕБОВАНИЯ:
1. **Function Signature:** Complete with type hints
2. **Parameters:** EACH param fully specified (type, constraints, examples)
3. **Return Value:** Structure, fields, examples
4. **Exceptions:** All exceptions with when/why raised
5. **Algorithm:** Pseudocode
6. **Edge Cases:** Minimum 3 per function
7. **Test Scenarios:** Minimum 5 per function
8. **Performance:** Latency/throughput expectations
9. **Length:** 500-800 words

CRITICAL:
- Every edge case must have clear handling
- Every test scenario must be executable
- Examples for every parameter (valid + invalid)
```

**Expected Time per Function:** 15-25 minutes
**Total for 40-50 functions:** 10-20 hours (with parallelization: 2-4 hours per wave)

**Waves:** 2-3 waves of 5 agents each → 1-2 days total

---

#### Step 3: Verification (5 Agents, Overlapping)

**Same process, now with 40-50 documents**

**Verification Focus:**
- ✅ Function signature matches component spec
- ✅ All parameters typed and constrained
- ✅ Edge cases enumerated (min 3)
- ✅ Test scenarios defined (min 5)
- ✅ Algorithm clear enough to implement
- ✅ Can write tests directly from spec

**Expected Time:** 2-3 hours

---

#### Step 4-7: Synthesis → Reconciliation → Refinement

**Expected Issues:**
- Missing edge cases
- Vague validation rules
- Incomplete exception documentation
- Missing test scenarios

**Iteration Cycles:** 1-2

**Total Time:** 4-6 hours

---

#### Step 8: User Review & Approval ⏸️ CHECKPOINT

**User Reviews:**
1. Sample 5-8 function specs (read in detail)
2. Verify edge cases make sense
3. Check test scenarios are executable
4. Approve

**Duration:** 1-2 hours

---

### Quality Gate 3: Function Specs Approved

**Criteria:**
- ✅ All 40-50 function specs created
- ✅ Every public function documented
- ✅ Edge cases (min 3 per function) → ~120+ edge cases total
- ✅ Test scenarios (min 5 per function) → ~200+ scenarios total
- ✅ User approval obtained

**Metrics:**
- Functions documented: 100%
- Edge cases per function: Avg >= 3
- Test scenarios defined: Total >= 200

**Exit:** User approval → Proceed to Phase 4 (TDD - Tests First!)

---

### Troubleshooting Phase 3

#### Issue: Can't Identify 3 Edge Cases for Simple Function

**Example:**
```python
def get_config() -> dict:
    """Return configuration dictionary."""
    return self.config
```

**Solution:**
Edge cases can be subtle:
1. Config not yet initialized (None)
2. Config corrupted (invalid structure)
3. Thread safety (concurrent access)

Even simple functions have edge cases - think harder!

---

#### Issue: Test Scenarios Too Vague

**Example:**
❌ "Test that function works"

**Solution:**
✅ Be specific:
```
1. test_add_memory_success_simple_text()
   - Input: "Python is great" (14 chars)
   - Expected: Returns memory_id (UUID format)
   - Verify: Database contains entry with text

2. test_add_memory_empty_text_raises()
   - Input: "" (empty string)
   - Expected: Raises ValidationError("text cannot be empty")

3. test_add_memory_max_length_success()
   - Input: "x" * 100000 (100K chars, at limit)
   - Expected: Success, chunks created

4. test_add_memory_exceeds_max_raises()
   - Input: "x" * 100001 (over limit)
   - Expected: Raises ValidationError("exceeds max length")

5. test_add_memory_ollama_offline_raises()
   - Mock: Ollama unavailable (ConnectionError)
   - Expected: Raises EmbeddingError("Ollama unavailable")
```

---

## 🧪 Phase 4: Test Development (TDD - RED Phase)

**Duration:** 3-5 days
**Agent Invocations:** ~10-15
**Quality Gate:** Gate 4

### Goals

Write ALL tests FIRST (before any code) based on function specs.

### Deliverables

Complete test suite (>= 200 tests):
- Unit tests: ~140 tests (70%)
- Integration tests: ~50 tests (25%)
- E2E tests: ~10 tests (5%)

All tests MUST FAIL (RED) - no code written yet!

### Step-by-Step Process

#### Step 1: Test Planning

**Agent Task:**
```
ЗАДАЧА: Create test plan from function specs

INPUT:
- All Level 3 function specs (40-50 specs)
- Test scenarios from each spec (5+ per function = 200+ scenarios)

PROCESS:
1. Extract all test scenarios
2. Categorize: Unit / Integration / E2E
3. Group by module/component
4. Create test file structure

OUTPUT:
Test plan document with:
- Test file paths
- Test function names
- Test categories
- Estimated count (200+)
```

**Expected Output:**
```markdown
# Test Plan - Phase 4

## Unit Tests (140 tests, 70%)

### tests/unit/test_add_memory_tool.py (15 tests)
1. test_add_memory_success_simple_text()
2. test_add_memory_empty_text_raises()
...

### tests/unit/test_text_processor.py (12 tests)
...

## Integration Tests (50 tests, 25%)

### tests/integration/test_mcp_to_core.py (10 tests)
...

## E2E Tests (10 tests, 5%)

### tests/e2e/test_full_workflow.py (6 tests)
...

**Total:** 200 tests
```

**Expected Time:** 1 hour

---

#### Step 2: Create Test Fixtures & Utilities

**Agent Task:**
```
ЗАДАЧА: Create test fixtures and utilities

PROCESS:
1. Create pytest fixtures for common setup:
   - Mock FalkorDB client
   - Mock Ollama client
   - Temporary directories
   - Sample data
2. Create test utilities:
   - Helper functions
   - Assertion helpers
   - Data generators

OUTPUT:
- tests/fixtures/db_fixtures.py
- tests/fixtures/ollama_fixtures.py
- tests/fixtures/data_fixtures.py
- tests/utils/helpers.py
```

**Expected Time:** 2-3 hours

---

#### Step 3: Write Tests (Sequential - Important!)

**Why Sequential?**
- Tests are interdependent (share fixtures)
- Need consistent patterns
- Avoid duplicate test logic

**Agent Task (1 agent, or 2-3 agents with careful coordination):**
```
ЗАДАЧА: Write all tests based on function specs

INPUT:
- All Level 3 function specs
- Test plan
- Fixtures

PROCESS:
For each test scenario in specs:
1. Create test function with descriptive name
2. Setup test data (using fixtures)
3. Execute function (will fail - no code yet)
4. Assert expected behavior
5. Document test with docstring

CRITICAL:
- Follow naming: test_{function}_{scenario}_{expected}
- Use fixtures (don't repeat setup code)
- Clear assertions (specific, not vague)
- Tests MUST FAIL (no implementation exists)

OUTPUT:
- All test files in tests/ directory
- All tests written (>= 200)
```

**Example Test:**
```python
def test_add_memory_success_simple_text(db_mock, ollama_mock):
    """Test add_memory with simple text should return valid memory ID."""
    # Setup
    tool = AddMemoryTool(db=db_mock, ollama=ollama_mock)
    text = "Python is a great programming language"

    # Execute
    result = tool.execute({"content": text})

    # Assert
    assert result["isError"] is False
    assert "content" in result
    assert len(result["content"]) > 0
    memory_id = extract_memory_id(result["content"][0]["text"])
    assert is_valid_uuid(memory_id)

    # Verify mock calls
    ollama_mock.embed.assert_called_once()
    db_mock.add_memory.assert_called_once()
```

**Expected Time per Test:** 5-10 minutes
**Total for 200 tests:** 16-33 hours (sequential: 3-5 days)

**Strategies to Speed Up:**
- Use AI code generation (carefully!)
- Reuse test patterns
- Good fixtures reduce setup time

---

#### Step 4: Run Tests (Verify RED Phase)

**Commands:**
```bash
cd /home/dev/zapomni

# Install pytest and dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/

# Expected output:
# ================== FAILURES ==================
# test_add_memory_success ... FAILED (AddMemoryTool not found)
# test_search_memory_success ... FAILED (SearchMemoryTool not found)
# ...
# ========== 200 failed in 12.34s ===========
```

**Verification:**
- ✅ All tests FAIL
- ✅ Tests fail for right reason (not broken tests, just no code)
- ✅ No syntax errors in tests
- ✅ Fixtures work

**If tests PASS:** Something wrong! No code exists yet, tests should fail.

**Expected Time:** 30 minutes

---

#### Step 5: Test Quality Review

**Agent Task:**
```
ЗАДАЧА: Review test quality

CRITERIA:
1. **Naming:** All tests follow convention
2. **Coverage:** Every function has tests (100%)
3. **Edge Cases:** All edge cases from specs tested
4. **Assertions:** Clear and specific
5. **Fixtures:** Reused properly (no duplication)
6. **Documentation:** Test docstrings clear

OUTPUT:
Test quality report with metrics:
- Tests written: count
- Functions covered: percentage
- Edge cases tested: percentage
- Quality score: 0-100
```

**Expected Time:** 1-2 hours

---

#### Step 6: User Review & Approval ⏸️ CHECKPOINT

**User Reviews:**
1. Sample 10-15 tests (read code)
2. Verify tests make sense
3. Check coverage (all functions have tests?)
4. Approve test suite

**User Action:**
```
User says: "Tests approved" or "Approve Gate 4"
```

**Duration:** 1 hour

---

### Quality Gate 4: Tests Complete

**Criteria:**
- ✅ All tests written (>= 200)
- ✅ All tests FAIL (RED phase confirmed)
- ✅ 100% functions have tests
- ✅ Test fixtures ready
- ✅ User approval obtained

**Metrics:**
- Tests written: >= 200
- Test-to-function ratio: >= 5:1
- Test status: 100% FAIL (expected)
- Test quality score: >= 90%

**Exit:** User approval → Proceed to Phase 5 (Implementation - Foundation)

---

### Troubleshooting Phase 4

#### Issue: Tests Pass Without Code

**Cause:** Tests not actually testing anything, or using stubs

**Solution:**
```python
# ❌ BAD - Test will pass even without code
def test_add_memory():
    assert True  # Always passes!

# ✅ GOOD - Test calls real function
def test_add_memory():
    tool = AddMemoryTool(...)
    result = tool.execute({"content": "test"})
    # Will fail: AddMemoryTool doesn't exist yet
```

---

#### Issue: Test Coverage Shows 0%

**Cause:** No code exists yet, coverage is of *code*, not *functions*

**Solution:**
This is expected! Coverage will be measured after code is written (Phase 5-7).

For now, verify coverage of *functions having tests*:
```bash
# Count functions in specs
grep -r "def " .spec-workflow/specs/level3/ | wc -l
# Output: 45

# Count test files for those functions
find tests/ -name "test_*.py" | wc -l
# Output: 45

# 45/45 = 100% function coverage ✅
```

---

## 🏗️ Phase 5: Foundation Development

**Duration:** 3-4 days
**Agent Invocations:** 5-7 (sequential)
**Quality Gate:** Gate 5a

### Goals

Implement shared infrastructure that all features depend on.

### Deliverables

Foundation code merged to `main`:
1. Project structure (pyproject.toml, __init__.py files)
2. Database client (FalkorDBClient)
3. Embedding wrapper (OllamaEmbedder)
4. Configuration management (Pydantic Settings)
5. Logging setup (structured logging)
6. Utilities (text processing, validation)
7. Test fixtures (shared for all tests)

All foundation tests GREEN ✅

### Step-by-Step Process

#### Step 1: Project Structure Setup

**Agent Task:**
```
TASK: Setup Python project structure

DELIVERABLES:
1. pyproject.toml (dependencies, build config)
2. __init__.py files in all packages
3. README.md (basic setup instructions)
4. requirements.txt (or poetry.lock)

DEPENDENCIES (from tech.md):
- Python 3.10+
- FalkorDB client (falkordb==1.1.0)
- Ollama client (ollama-python==0.2.0)
- Pydantic (pydantic==2.5.0)
- Pytest (pytest==7.4.0)
- [full list from tech.md]

STRUCTURE:
/home/dev/zapomni/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── zapomni_mcp/
│   │   ├── __init__.py
│   │   └── tools/
│   │       └── __init__.py
│   ├── zapomni_core/
│   │   ├── __init__.py
│   │   └── [subpackages]
│   └── zapomni_db/
│       ├── __init__.py
│       └── [subpackages]
└── tests/
    └── __init__.py
```

**Expected Time:** 1-2 hours

**Commit:**
```bash
git add .
git commit -m "feat(foundation): Setup Python project structure

- Add pyproject.toml with dependencies
- Create package __init__.py files
- Add basic README with setup instructions

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Step 2: Database Client Implementation

**Agent Task:**
```
TASK: Implement FalkorDBClient following TDD

SPEC:
- .spec-workflow/specs/level2/falkordb_client_component.md
- .spec-workflow/specs/level3/falkordb_*.md (all related functions)

TESTS:
- tests/integration/test_falkordb_client.py (already written in Phase 4)

PROCESS:
1. Read specs
2. Read tests (understand expected behavior)
3. Implement minimal code to make tests pass (GREEN)
4. Run tests → should turn GREEN
5. Refactor for quality
6. Ensure tests stay GREEN

OUTPUT:
- src/zapomni_db/falkordb/client.py
- All FalkorDB tests GREEN

TDD CYCLE:
RED (tests exist, failing) → GREEN (implement) → REFACTOR (improve)
```

**Expected Time:** 4-6 hours

**Commit:**
```bash
git add src/zapomni_db/falkordb/
git commit -m "feat(db): Implement FalkorDBClient

- Database connection management
- Graph operations (create, query, update)
- Schema initialization
- Error handling and retries

Tests: 15/15 GREEN ✅
Coverage: 95%

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Step 3: Embedding Wrapper Implementation

**Agent Task:**
```
TASK: Implement OllamaEmbedder following TDD

SPEC:
- .spec-workflow/specs/level2/ollama_embedder_component.md
- .spec-workflow/specs/level3/ollama_*.md

TESTS:
- tests/unit/test_ollama_embedder.py
- tests/integration/test_ollama_client.py

PROCESS:
[Same TDD cycle]

OUTPUT:
- src/zapomni_core/embeddings/ollama.py
- All Ollama tests GREEN
```

**Expected Time:** 3-5 hours

**Commit:**
```bash
git commit -m "feat(embeddings): Implement OllamaEmbedder

- Embedding generation via Ollama API
- Batch processing support
- Error handling (offline, timeout)
- Caching (optional)

Tests: 12/12 GREEN ✅
Coverage: 93%

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Step 4: Configuration Management

**Agent Task:**
```
TASK: Implement configuration management with Pydantic Settings

SPEC:
- .spec-workflow/specs/level1/configuration_management.md
- .spec-workflow/specs/level2/config_*.md

OUTPUT:
- src/zapomni_core/config.py

FEATURES:
- Environment variable loading (.env support)
- Validation (Pydantic)
- Defaults for all settings
- Type hints

EXAMPLE:
```python
from pydantic_settings import BaseSettings

class ZapomniConfig(BaseSettings):
    # Database
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # Performance
    chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"
        env_prefix = "ZAPOMNI_"
```
```

**Expected Time:** 2-3 hours

---

#### Step 5: Logging Setup

**Agent Task:**
```
TASK: Setup structured logging

OUTPUT:
- src/zapomni_core/logging_config.py

FEATURES:
- Structured logging (JSON format)
- Log to stderr (MCP requirement)
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Context injection (request_id, etc)

EXAMPLE:
```python
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)
    )

logger = structlog.get_logger()
logger.info("add_memory_called", text_length=len(text), user="xyz")
```
```

**Expected Time:** 1-2 hours

---

#### Step 6: Utilities Implementation

**Agent Task:**
```
TASK: Implement utility functions

SPECS:
- Various function specs for utilities

OUTPUT:
- src/zapomni_core/utils/text.py (text processing)
- src/zapomni_core/utils/validation.py (input validation)
- src/zapomni_core/utils/uuid.py (UUID generation)

FUNCTIONS:
- clean_text(text: str) -> str
- validate_uuid(uuid_str: str) -> bool
- generate_memory_id() -> str
- truncate_text(text: str, max_len: int) -> str
- ...etc
```

**Expected Time:** 3-4 hours

---

#### Step 7: Foundation Integration Testing

**Commands:**
```bash
# Run all foundation tests
pytest tests/unit/test_falkordb_client.py
pytest tests/unit/test_ollama_embedder.py
pytest tests/unit/test_config.py
pytest tests/unit/test_utils.py

# Run integration tests
pytest tests/integration/test_db_integration.py
pytest tests/integration/test_ollama_integration.py

# All should be GREEN
```

**Expected Result:**
```
================== 45 passed in 8.34s ==================
Coverage: 94%
```

---

#### Step 8: Merge Foundation to Main

**Process:**
```bash
# Ensure all tests green
pytest tests/

# Ensure linting clean
black src/ tests/
isort src/ tests/
mypy src/

# Merge to main (if on branch)
git checkout main
git merge feature/foundation
git push origin main
```

**All agents wait here** - foundation MUST be merged before features can start

---

### Quality Gate 5a: Foundation Complete

**Criteria:**
- ✅ Infrastructure code implemented
- ✅ Foundation tests passing (GREEN)
- ✅ Coverage >= 90% for foundation
- ✅ Merged to `main`
- ✅ All agents synchronized on new baseline

**Exit:** Foundation merged → Proceed to Phase 6 (Features Wave 1)

---

### Troubleshooting Phase 5

#### Issue: FalkorDB Connection Fails in Tests

**Cause:** FalkorDB not running

**Solution:**
```bash
# Start FalkorDB via Docker
docker run -d -p 6379:6379 falkordb/falkordb:latest

# Verify connection
redis-cli -h localhost -p 6379 PING
# Should return: PONG
```

---

#### Issue: Ollama Tests Fail (Model Not Found)

**Cause:** Ollama model not pulled

**Solution:**
```bash
# Pull embedding model
ollama pull nomic-embed-text

# Verify
ollama list
# Should show: nomic-embed-text
```

---

## ⚡ Phase 6: Features Wave 1 (Parallel)

**Duration:** 3-4 days
**Agent Invocations:** 6 (3 implementation + 3 review)
**Quality Gate:** Gate 5b

### Goals

Implement 3 core features in parallel, each on separate branch.

### Features

1. **add_memory** - Document ingestion with chunking and embedding
2. **search_memory** - Hybrid search (BM25 + vector)
3. **get_stats** - System statistics and monitoring

### Step-by-Step Process

#### Step 1: Feature Branch Creation

**Commands:**
```bash
# Create branches (can be done by agents or manually)
git checkout main
git checkout -b feature/add-memory

git checkout main
git checkout -b feature/search-memory

git checkout main
git checkout -b feature/get-stats
```

---

#### Step 2: Parallel Implementation (3 Agents Simultaneously)

**Agent 1: Implement add_memory**

```markdown
TASK: Implement add_memory feature using TDD

BRANCH: feature/add-memory

SPECS:
- .spec-workflow/specs/level2/add_memory_tool_component.md
- .spec-workflow/specs/level3/add_memory_*.md (all related functions)
- .spec-workflow/specs/level2/text_processor_component.md

TESTS:
- tests/unit/test_add_memory_tool.py (already written)
- tests/integration/test_add_memory_integration.py

TDD PROCESS:
1. Checkout branch: feature/add-memory
2. Read specs
3. Read tests (verify understanding)
4. Implement code to make tests GREEN
5. Refactor for quality
6. Ensure coverage >= 90%
7. Commit and push
8. Create PR

DELIVERABLES:
- src/zapomni_mcp/tools/add_memory.py (MCP tool)
- src/zapomni_core/processors/text_processor.py (processing logic)
- All add_memory tests GREEN
- Coverage >= 90%

QUALITY CHECKS:
- Black formatted ✅
- isort applied ✅
- mypy clean ✅
- All tests pass ✅
```

**Agent 2: Implement search_memory** (similar prompt, different feature)

**Agent 3: Implement get_stats** (similar prompt, different feature)

**Expected Time per Feature:** 6-10 hours
**Total (parallel):** 6-10 hours (same time, 3 agents working simultaneously)

---

#### Step 3: Feature PRs Created

Each agent creates PR:

```bash
# Agent 1 (add_memory)
git checkout feature/add-memory
git add src/zapomni_mcp/tools/add_memory.py src/zapomni_core/processors/
git commit -m "feat(mcp): Implement add_memory tool

- Add AddMemoryTool MCP tool implementation
- Add TextProcessor for document chunking
- Integrate with FalkorDB and Ollama
- Handle errors and edge cases

Tests: 25/25 GREEN ✅
Coverage: 94%

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin feature/add-memory

# Create PR using gh CLI
gh pr create --title "feat(mcp): Implement add_memory tool" \
  --body "$(cat <<'EOF'
## Summary
- Implements add_memory MCP tool for document ingestion
- Text chunking with configurable size/overlap
- Embedding generation via Ollama
- Storage in FalkorDB graph

## Tests
- 25/25 tests passing ✅
- Coverage: 94%

## Checklist
- [x] All tests green
- [x] Coverage >= 90%
- [x] Black formatted
- [x] mypy clean
- [x] Follows specs exactly

🤖 Generated with Claude Code
EOF
)"
```

**Repeat for search_memory and get_stats**

---

#### Step 4: Code Review (3 Review Agents in Parallel)

**Agent 4: Review add_memory PR**
**Agent 5: Review search_memory PR**
**Agent 6: Review get_stats PR**

**Review Agent Prompt:**
```markdown
TASK: Code review for PR #X (add_memory feature)

PR INFO:
- Feature: add_memory
- Branch: feature/add-memory
- Files changed: 12
- Author: Agent 1

REVIEW CHECKLIST:
[Full checklist from AGENT_COORDINATION.md Code Review Template]

1. Code Quality
2. Testing (coverage, edge cases)
3. Security (input validation, no injection)
4. Alignment with Specs
5. Performance

OUTPUT:
Code review report with verdict: APPROVE / REQUEST_CHANGES / REJECT
```

Use template: `.prompts/code_review_template.md`

**Expected Time per Review:** 30-45 minutes
**Total (parallel):** 30-45 minutes

---

#### Step 5: Address Review Feedback (If Needed)

**If REQUEST_CHANGES:**
```
Implementation agent fixes issues
Re-runs tests
Pushes updates
Review agent re-reviews
```

**Iteration:** 1-2 cycles typical

---

#### Step 6: Merge Features to Main

**After APPROVE:**
```bash
# Merge add_memory
git checkout main
git merge feature/add-memory --squash
git commit -m "feat(mcp): Add add_memory tool [squashed]"
git push origin main

# Delete branch
git branch -d feature/add-memory
git push origin --delete feature/add-memory

# Repeat for search_memory and get_stats
```

**Order:** Can merge in any order (features independent)

---

#### Step 7: Integration Testing (All 3 Features Together)

**Commands:**
```bash
# Run all tests (unit + integration)
pytest tests/

# Run integration test with all 3 features
pytest tests/integration/test_all_features.py

# Expected:
# - add_memory works ✅
# - search_memory works ✅
# - get_stats works ✅
# - Features work together ✅
```

**If integration fails:** Debug, fix, re-test

---

### Quality Gate 5b: Wave 1 Complete

**Criteria:**
- ✅ All 3 features merged to `main`
- ✅ All tests GREEN (100% passing)
- ✅ No merge conflicts
- ✅ Integration tests passing
- ✅ Coverage >= 90% overall

**Metrics:**
- Features complete: 3/5 (60%)
- Tests passing: 100%
- Coverage: >= 90%

**Exit:** All features merged → Proceed to Phase 7 (Features Wave 2)

---

### Troubleshooting Phase 6

#### Issue: Merge Conflict Between Features

**Cause:** Two features modified same file (shouldn't happen with good isolation, but can)

**Solution:**
```bash
# Reconciliation agent resolves conflict
git checkout feature/search-memory
git merge main

# Conflict in src/zapomni_mcp/server.py
# Agent reads both changes, merges intelligently
# Commits resolution

git add src/zapomni_mcp/server.py
git commit -m "fix: Resolve merge conflict with add_memory"
git push
```

---

#### Issue: Integration Test Fails (Features Don't Work Together)

**Example:**
```
add_memory stores data with ID format: "mem_12345"
search_memory expects ID format: "12345" (UUID)
```

**Solution:**
1. Identify root cause (ID format mismatch)
2. Fix one feature (change ID format to UUID)
3. Update tests
4. Re-run integration tests
5. Merge fix

---

## 🚀 Phase 7: Features Wave 2 (Parallel)

**Duration:** 2-3 days
**Agent Invocations:** 4 (2 implementation + 2 review)

### Features

4. **hybrid_search** - Enhanced search with cross-encoder reranking
5. **caching_layer** - Semantic caching for performance

### Process

**Same as Phase 6**, but only 2 features:

1. Create branches
2. Agent 7: Implement hybrid_search
3. Agent 8: Implement caching_layer
4. Create PRs
5. Agent 9: Review hybrid_search
6. Agent 10: Review caching_layer
7. Address feedback
8. Merge to main
9. Integration testing

**Expected Time:** 4-6 hours implementation + 2-3 hours review/integration

---

### Quality Gate 5c: Wave 2 Complete

**Criteria:**
- ✅ All 5 features merged to `main`
- ✅ All tests GREEN
- ✅ Performance SLAs met:
  - add_memory: < 500ms (P95)
  - search_memory: < 200ms (P95)
  - Cache hit rate: > 60%

**Exit:** All features merged + performance validated → Proceed to Phase 8

---

## 🔗 Phase 8: Integration & Testing

**Duration:** 2-3 days
**Agent Invocations:** 3-5
**Quality Gate:** Gate 6a

### Goals

Validate entire system works end-to-end, test with real services, benchmark performance.

### Activities

1. Full integration testing
2. Performance testing
3. End-to-end workflow testing
4. MCP integration testing
5. Bug fixing

### Step-by-Step Process

#### Step 1: Full Integration Testing

**Agent Task:**
```
TASK: Run full integration test suite

SETUP:
1. Start FalkorDB: docker-compose up -d falkordb
2. Start Ollama: ollama serve
3. Pull model: ollama pull nomic-embed-text

TESTS:
pytest tests/integration/ -v

VERIFY:
- All integration tests pass
- Database operations work
- Ollama integration works
- MCP protocol works
```

**Expected Result:**
```
================== 50 passed in 45.23s ==================
```

**Expected Time:** 1-2 hours

---

#### Step 2: Performance Testing

**Agent Task:**
```
TASK: Run performance benchmarks

TESTS:
pytest tests/performance/ --benchmark-only

BENCHMARKS:
1. add_memory latency (target: < 500ms P95)
2. search_memory latency (target: < 200ms P95)
3. Memory usage (target: < 2GB for 100K items)
4. Cache hit rate (target: > 60%)

DATASET:
Generate 10,000 test documents
```

**Commands:**
```bash
# Run benchmarks
pytest tests/performance/bench_add_memory.py --benchmark-only
pytest tests/performance/bench_search.py --benchmark-only
pytest tests/performance/bench_memory_usage.py
```

**Expected Output:**
```
Benchmark: add_memory_latency
  Mean:   342ms
  P95:    467ms  ✅ (target: < 500ms)
  P99:    512ms

Benchmark: search_memory_latency
  Mean:   145ms
  P95:    178ms  ✅ (target: < 200ms)
  P99:    234ms

Memory usage: 1.8GB  ✅ (target: < 2GB)
Cache hit rate: 67%  ✅ (target: > 60%)
```

**If SLAs Not Met:** Profile, optimize, re-test

**Expected Time:** 2-4 hours

---

#### Step 3: End-to-End Workflow Testing

**Agent Task:**
```
TASK: Test complete user workflows

SCENARIOS:
1. Fresh install → add 100 memories → search → verify results
2. Add memories with metadata → filter search → verify
3. Error scenarios (DB offline, Ollama offline) → verify graceful handling

TEST:
pytest tests/e2e/test_full_workflow.py -v
```

**Example E2E Test:**
```python
def test_complete_workflow(clean_db):
    """Test full workflow: setup → add → search → stats."""
    # 1. Initialize MCP server
    server = MCPServer(config=test_config)

    # 2. Add memories
    for i in range(10):
        result = server.call_tool("add_memory", {
            "content": f"Test document {i} about Python programming"
        })
        assert result["isError"] is False

    # 3. Search
    results = server.call_tool("search_memory", {
        "query": "Python",
        "limit": 5
    })
    assert len(results["memories"]) == 5

    # 4. Get stats
    stats = server.call_tool("get_stats", {})
    assert stats["total_memories"] == 10
```

**Expected Time:** 2-3 hours

---

#### Step 4: MCP Integration Testing with Claude CLI

**Manual Testing (User or Agent with access):**

```bash
# 1. Configure Claude CLI to use Zapomni MCP server
cat > ~/.config/claude/mcp_config.json <<EOF
{
  "servers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "cwd": "/home/dev/zapomni"
    }
  }
}
EOF

# 2. Start Claude CLI
claude

# 3. Test add_memory
User: "Add a memory: Python is a high-level programming language"
Claude: [uses add_memory tool] Memory added successfully: mem_abc123

# 4. Test search_memory
User: "Search for memories about Python"
Claude: [uses search_memory tool] Found 1 memory: "Python is a high-level programming language"

# 5. Test get_stats
User: "Show me memory statistics"
Claude: [uses get_stats tool] Total memories: 1, Total chunks: 3
```

**Verification:**
- ✅ MCP tools appear in Claude CLI
- ✅ add_memory works
- ✅ search_memory works
- ✅ get_stats works
- ✅ Error messages clear

**Expected Time:** 1 hour

---

#### Step 5: Bug Fixing & Polish

**Process:**
```
If any issues found in Steps 1-4:
1. Create bug report
2. Prioritize (critical, high, medium, low)
3. Fix critical bugs immediately
4. Fix high bugs before Gate 6a
5. Medium/low can wait for post-MVP
```

**Bug Fix Workflow:**
```bash
# Create branch
git checkout -b fix/bug-description

# Fix bug
[agent edits code]

# Add test for bug (regression test)
# Commit
git commit -m "fix(component): Fix bug description

- Root cause: [explanation]
- Fix: [what changed]
- Test added: [regression test]

Closes #issue-number"

# Merge
git checkout main
git merge fix/bug-description
```

**Expected Bugs:** 2-5 (normal for first integration)

**Expected Time:** 2-6 hours (depends on bug complexity)

---

### Quality Gate 6a: Integration Complete

**Criteria:**
- ✅ All integration tests GREEN
- ✅ All E2E tests GREEN
- ✅ Performance benchmarks met (all SLAs)
- ✅ MCP integration functional
- ✅ Zero critical bugs remaining
- ✅ Coverage >= 90%

**Exit:** All tests passing + performance validated → Proceed to Phase 9

---

### Troubleshooting Phase 8

#### Issue: Performance Below SLA

**Example:** add_memory P95 = 650ms (target: < 500ms)

**Solution:**
```bash
# Profile code
python -m cProfile -o profile.stats tests/performance/bench_add_memory.py

# Analyze
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Identify bottleneck (e.g., Ollama embedding slow)
# Optimize (e.g., batch embeddings, cache)
# Re-benchmark
```

---

#### Issue: MCP Integration Fails

**Example:** Claude CLI doesn't see Zapomni tools

**Solution:**
```bash
# Check MCP server starts
python -m zapomni_mcp.server
# Should output: MCP server ready on stdio

# Check MCP config
cat ~/.config/claude/mcp_config.json
# Verify paths correct

# Check Claude CLI logs
claude --verbose
# Look for Zapomni connection errors
```

---

## 📚 Phase 9: Documentation & Polish

**Duration:** 2-3 days
**Agent Invocations:** 4-5
**Quality Gate:** Gate 6b (MVP READY!)

### Goals

Create comprehensive documentation, test installation, prepare release.

### Deliverables

1. README.md (30-min quick start)
2. API documentation
3. User guide
4. Troubleshooting guide
5. CHANGELOG.md
6. Docker Compose for easy setup
7. Example scripts
8. GitHub release v0.1.0-mvp

### Step-by-Step Process

#### Step 1: README.md (Quick Start Guide)

**Agent Task:**
```
TASK: Create comprehensive README.md

STRUCTURE:
1. Project overview (what is Zapomni)
2. Features (3 core MCP tools)
3. Quick Start (30-min setup)
   - Prerequisites
   - Installation steps
   - Configuration
   - First use
4. Architecture overview
5. Contributing
6. License (MIT)
7. Author attribution

CRITICAL:
- Setup instructions must work on clean Ubuntu/macOS
- Test installation yourself (if possible) or provide script
- Target: 30 minutes from zero to working system

LENGTH: 800-1200 words
```

**Example README Quick Start:**
```markdown
## Quick Start (30 minutes)

### Prerequisites
- Python 3.10+
- Docker (for FalkorDB)
- Ollama

### Installation

1. **Install Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
```

2. **Start FalkorDB:**
```bash
docker run -d -p 6379:6379 --name zapomni-db falkordb/falkordb:latest
```

3. **Install Zapomni:**
```bash
git clone https://github.com/alienxs2/zapomni.git
cd zapomni
pip install -e .
```

4. **Configure Claude CLI:**
```bash
cat > ~/.config/claude/mcp_config.json <<EOF
{
  "servers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"]
    }
  }
}
EOF
```

5. **Test:**
```bash
claude
> Add a memory: Python is amazing
> Search for Python
```

✅ Done! You now have a local AI memory system.
```

**Expected Time:** 2-3 hours

---

#### Step 2: API Documentation (Auto-Generated)

**Commands:**
```bash
# Generate API docs from docstrings
pip install pdoc3

pdoc --html --output-dir docs/api src/zapomni_mcp src/zapomni_core src/zapomni_db

# Result: HTML docs in docs/api/
```

**Agent Task:**
```
TASK: Review and enhance generated API docs

1. Generate initial docs (pdoc)
2. Add overview pages for each module
3. Add usage examples to key classes
4. Ensure all public APIs documented
```

**Expected Time:** 2-3 hours

---

#### Step 3: User Guide

**Agent Task:**
```
TASK: Create user guide

FILE: docs/guides/user_guide.md

CONTENT:
1. Introduction (what is Zapomni)
2. Core Concepts (memories, embeddings, search)
3. Using MCP Tools
   - add_memory detailed guide
   - search_memory detailed guide
   - get_stats detailed guide
4. Common Workflows
   - Building a knowledge base
   - Searching effectively
   - Using metadata
5. Advanced Topics
   - Configuration options
   - Performance tuning
6. FAQ

LENGTH: 2000-3000 words
```

**Expected Time:** 3-4 hours

---

#### Step 4: Troubleshooting Guide

**Agent Task:**
```
TASK: Create troubleshooting guide

FILE: docs/guides/troubleshooting.md

CONTENT:
Common Issues and Solutions:
1. Installation Issues
   - FalkorDB connection failed
   - Ollama not found
   - Python version mismatch
2. Runtime Issues
   - add_memory slow
   - search_memory returns no results
   - MCP tools not appearing in Claude
3. Performance Issues
   - High memory usage
   - Slow search
4. Error Messages
   - "Database connection failed"
   - "Ollama unavailable"
   - Each error with cause + solution

LENGTH: 1500-2000 words
```

**Expected Time:** 2-3 hours

---

#### Step 5: CHANGELOG.md

**Agent Task:**
```
TASK: Create CHANGELOG.md

FORMAT: Keep a Changelog (keepachangelog.com)

FILE: CHANGELOG.md

CONTENT:
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-mvp] - 2025-11-XX

### Added
- Initial MVP release
- MCP tools: add_memory, search_memory, get_stats
- FalkorDB integration for graph storage
- Ollama integration for embeddings
- Hybrid search (BM25 + vector)
- Semantic caching layer
- Complete test suite (200+ tests, 90%+ coverage)
- Comprehensive documentation

### Features
- 100% local execution (no external APIs)
- Sub-500ms add_memory performance
- Sub-200ms search_memory performance
- 60%+ cache hit rate

[0.1.0-mvp]: https://github.com/alienxs2/zapomni/releases/tag/v0.1.0-mvp
```

**Expected Time:** 30 minutes

---

#### Step 6: Docker Compose Setup

**Agent Task:**
```
TASK: Create docker-compose.yml for easy setup

FILE: docker-compose.yml

CONTENT:
version: '3.8'

services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data
    restart: unless-stopped

  # Optional: Add Ollama container if user wants fully containerized
  # (though Ollama typically runs on host)

volumes:
  falkordb_data:

ALSO CREATE:
- .env.example (example environment variables)
- docker-compose.override.yml.example (for local customization)
```

**Usage:**
```bash
docker-compose up -d
# FalkorDB ready!
```

**Expected Time:** 1 hour

---

#### Step 7: Example Scripts

**Agent Task:**
```
TASK: Create example usage scripts

FILES:
1. examples/basic_usage.py
   - Simple add and search example
   - Commented heavily

2. examples/build_knowledge_base.py
   - Load 100 documents from files
   - Demonstrate batch processing

3. examples/advanced_search.py
   - Metadata filtering
   - Result reranking
   - Complex queries

Each script:
- Fully working (tested)
- Well-commented
- Clear output
```

**Expected Time:** 2-3 hours

---

#### Step 8: Installation Testing (Critical!)

**Agent Task (or User):**
```
TASK: Test installation on clean environment

SETUP:
1. Create fresh Ubuntu 22.04 VM (or Docker container)
2. No dependencies pre-installed

PROCESS:
Follow README.md Quick Start exactly
Time each step
Document any issues

SUCCESS CRITERIA:
- Complete in <= 30 minutes
- All 3 MCP tools work
- No errors encountered
```

**If installation fails or exceeds 30 min:** Fix docs, simplify steps, re-test

**Expected Time:** 1-2 hours (including fixes)

---

#### Step 9: GitHub Release

**Process:**
```bash
# 1. Tag release
git tag -a v0.1.0-mvp -m "Release MVP v0.1.0

First production-ready release of Zapomni.

Features:
- 3 MCP tools (add_memory, search_memory, get_stats)
- 100% local execution
- FalkorDB + Ollama integration
- 90%+ test coverage
- Comprehensive documentation

Author: Goncharenko Anton aka alienxs2
License: MIT"

git push origin v0.1.0-mvp

# 2. Create GitHub release
gh release create v0.1.0-mvp \
  --title "Zapomni MVP v0.1.0" \
  --notes "$(cat <<'EOF'
# Zapomni MVP - First Release 🎉

Local-first MCP memory system for AI agents.

## Features

✅ **3 Core MCP Tools:**
- `add_memory` - Store information with automatic chunking
- `search_memory` - Hybrid search (BM25 + vector)
- `get_stats` - System statistics

✅ **100% Local:**
- No external API calls
- Complete privacy
- Zero recurring costs

✅ **High Performance:**
- < 500ms add_memory (P95)
- < 200ms search_memory (P95)
- 60%+ cache hit rate

✅ **Production Ready:**
- 200+ tests, 90%+ coverage
- Type-safe (mypy strict)
- Comprehensive docs

## Installation

See [README.md](README.md) for full setup guide.

Quick start: 30 minutes from zero to working system.

## Author

**Goncharenko Anton aka alienxs2**

## License

MIT

---

🤖 Built with Claude Code
EOF
)"
```

**Expected Time:** 30 minutes

---

### Quality Gate 6b: MVP Ready for Production

**Final Checklist:**
- ✅ Documentation complete (README, API docs, user guide, troubleshooting)
- ✅ Installation tested (< 30 min on clean system)
- ✅ MCP integration working with Claude CLI
- ✅ Docker Compose setup
- ✅ Example scripts working
- ✅ CHANGELOG created
- ✅ GitHub release tagged (v0.1.0-mvp)
- ✅ User acceptance testing passed
- ✅ **USER FINAL APPROVAL** → **MVP COMPLETE** 🎉

**Exit:** User approval → **ZAPOMNI MVP COMPLETE!**

---

### Troubleshooting Phase 9

#### Issue: Installation Takes > 30 Minutes

**Causes:**
- Too many manual steps
- Dependencies slow to install
- Unclear instructions

**Solutions:**
- Create installation script (automate steps)
- Pre-build Docker image (skip build time)
- Improve documentation (clearer steps)

---

#### Issue: Documentation Unclear

**Solution:**
- Get fresh eyes (someone unfamiliar with project)
- Test documentation (follow it exactly)
- Improve based on feedback

---

## 📊 Summary Timeline

| Phase | Duration | Agents | Deliverable | Status |
|-------|----------|--------|-------------|--------|
| 0 | 1-2 days | 5-10 | Planning docs, setup | ✅ COMPLETE |
| 1 | 2-3 days | 14-20 | 7 module specs | ⏳ NEXT |
| 2 | 4-5 days | 40-50 | 15-20 component specs | ⏸️ Pending |
| 3 | 7-10 days | 80-100 | 40-50 function specs | ⏸️ Pending |
| 4 | 3-5 days | 10-15 | Complete test suite | ⏸️ Pending |
| 5 | 3-4 days | 5-7 | Foundation code | ⏸️ Pending |
| 6 | 3-4 days | 6 | 3 core features | ⏸️ Pending |
| 7 | 2-3 days | 4 | 2 enhanced features | ⏸️ Pending |
| 8 | 2-3 days | 3-5 | Integration validated | ⏸️ Pending |
| 9 | 2-3 days | 4-5 | MVP released | ⏸️ Pending |
| **Total** | **~8 weeks** | **175-210** | **Production MVP** | ⏸️ In Progress |

---

## 🔄 Phase Transitions

### General Pattern

1. ✅ Complete current phase deliverables
2. ✅ Quality Gate check
3. ⏸️ User approval
4. ✅ Setup for next phase
5. ▶️ Start next phase

### Example: Phase 1 → Phase 2

1. **Phase 1 complete:** 7 module specs verified ✅
2. **Quality Gate 1:** All criteria met ✅
3. **User approves:** "Proceed to Phase 2" ⏸️
4. **Setup:** Create component spec plan (which components for each module)
5. **Start Phase 2:** Agent creates first component spec ▶️

---

## 🛠️ Common Commands Reference

### Development

```bash
# Setup
cd /home/dev/zapomni
pip install -e ".[dev]"

# Testing
pytest tests/                    # All tests
pytest tests/unit/              # Unit only
pytest tests/integration/       # Integration only
pytest --cov=src --cov-report=html  # With coverage

# Code Quality
black src/ tests/               # Format
isort src/ tests/               # Sort imports
mypy src/                       # Type check
pylint src/                     # Lint

# Git
git status
git add .
git commit -m "feat: description"
git push origin main

# Branches
git checkout -b feature/name
git checkout main
git merge feature/name
```

### Services

```bash
# FalkorDB
docker run -d -p 6379:6379 falkordb/falkordb:latest
docker ps | grep falkordb
docker logs <container-id>

# Ollama
ollama serve                    # Start server
ollama pull nomic-embed-text    # Pull model
ollama list                     # List models
```

### Verification

```bash
# Check specs
ls -la .spec-workflow/specs/level1/
wc -l .spec-workflow/specs/**/*.md

# Check tests
find tests/ -name "test_*.py" | wc -l
pytest --collect-only | grep "test session starts"

# Check coverage
coverage report
coverage html
open htmlcov/index.html
```

---

## 🔧 Troubleshooting Guide

### By Phase

**Phase 0-3 (Specs):**
- **Issue:** Verification finds critical issues after 2 cycles
- **Solution:** User review, possible requirement clarification, refine steering docs

**Phase 4 (Tests):**
- **Issue:** Can't write tests from specs (specs too vague)
- **Solution:** Go back to Phase 3, refine function specs with more detail

**Phase 5-7 (Implementation):**
- **Issue:** Git merge conflicts
- **Solution:** Reconciliation agent or manual resolution
- **Prevention:** Better feature isolation, foundation-first approach

**Phase 8 (Integration):**
- **Issue:** Integration tests fail
- **Solution:** Debug integration points, fix interfaces, ensure services running

**Phase 9 (Documentation):**
- **Issue:** Installation doesn't work in 30 min
- **Solution:** Improve setup scripts, simplify process, create automation

---

### General Issues

#### Agent Timeout

**Symptom:** Agent takes > 10 minutes

**Solution:**
1. Kill agent (automatic timeout)
2. Review task complexity
3. Break into smaller sub-tasks
4. Retry with adjusted prompt

---

#### Tests Failing Unexpectedly

**Symptom:** Tests were green, now red (no code changed)

**Cause:** External service down (FalkorDB, Ollama)

**Solution:**
```bash
# Check FalkorDB
docker ps | grep falkordb
# Restart if needed
docker restart <container-id>

# Check Ollama
curl http://localhost:11434/api/tags
# Restart if needed
pkill ollama && ollama serve
```

---

#### Coverage Below Target

**Symptom:** Coverage = 85% (target: 90%)

**Solution:**
```bash
# Find uncovered lines
coverage report --show-missing

# Add tests for uncovered lines
# Focus on:
# - Error paths (exceptions)
# - Edge cases
# - Conditional branches
```

---

## 📖 References

### Internal Documents
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Master plan
- [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md) - 3-level spec process
- [AGENT_COORDINATION.md](AGENT_COORDINATION.md) - Agent management
- [QUALITY_FRAMEWORK.md](QUALITY_FRAMEWORK.md) - Testing & quality

### Steering Documents
- [product.md](.spec-workflow/steering/product.md) - Vision & features
- [tech.md](.spec-workflow/steering/tech.md) - Architecture & stack
- [structure.md](.spec-workflow/steering/structure.md) - Project structure

---

## 📝 Changelog

**Version 1.0 - 2025-11-23**
- Initial phase-by-phase implementation guide created
- Detailed step-by-step process for all 10 phases (Phase 0-9)
- Concrete commands and agent tasks for each step
- Quality gate criteria for each phase
- Troubleshooting guides
- Timeline estimates (8 weeks total)
- Command reference for common operations

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**GitHub:** https://github.com/alienxs2/zapomni

*This document provides detailed implementation guidance for each phase of Zapomni development.*

---

**Document Statistics:**
- **Total Word Count:** ~12,500 words
- **Total Phases:** 10 (Phase 0-9)
- **Total Steps:** ~80 detailed steps
- **Total Commands:** 100+ concrete commands
- **Reading Time:** ~60 minutes
