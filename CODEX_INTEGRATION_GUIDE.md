# Codex CLI Integration Guide for Claude Code CLI

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Date:** 2025-11-23
**Purpose:** Enable Claude Code CLI to delegate tasks to Codex CLI

---

## 📋 Overview

This guide explains how Claude Code CLI can programmatically call and coordinate with OpenAI's Codex CLI to delegate coding tasks, enabling multi-AI-agent workflows.

### Why Integrate?

**Strengths of Each CLI:**
- **Claude Code CLI (Anthropic):** Planning, architecture, reasoning, multi-agent coordination
- **Codex CLI (OpenAI GPT-5):** One-shot code generation, bulk implementation, rapid prototyping

**Together:** Best of both worlds - Claude plans and coordinates, Codex executes implementation.

---

## ✅ Prerequisites Check

### 1. Verify Codex CLI Installation

```bash
# Check if Codex is installed
which codex
# Expected: /home/dev/.nvm/versions/node/vX.X.X/bin/codex (or similar)

# Check version
codex --version
# Expected: codex-cli 0.63.0 or later
```

### 2. Verify Codex Authentication

```bash
# Check login status
codex login --status 2>&1 | grep -i "logged in"

# If not logged in:
codex login
# Follow prompts to sign in with ChatGPT (Plus/Pro/Team account required)
```

### 3. Test Basic Codex Execution

```bash
# Non-interactive test
codex exec "echo 'Codex CLI test successful'"

# Expected: Should execute and return output
```

---

## ⚠️ CRITICAL: Sandbox Mode

**Codex CLI has THREE sandbox modes:**

### 1. Read-Only Sandbox (DEFAULT - Cannot Write)

```bash
# DEFAULT - Codex can only READ files
codex exec "analyze component"

# Output: sandbox: read-only
# Result: Codex reads and analyzes, CANNOT create/modify files
```

**Use for:** Analysis, code review, questions, diagnostics

### 2. Workspace-Write Sandbox (RECOMMENDED for Implementation)

```bash
# WORKSPACE WRITE ACCESS - Can write in workdir
codex exec "implement component" --sandbox workspace-write

# Output: sandbox: workspace-write [workdir, /tmp, $TMPDIR]
# Result: Codex CAN create/modify files in workspace
```

**Use for:** Implementation, file creation, code generation
**Restrictions:** Can only write in workdir + /tmp (safe)

### 3. Danger-Full-Access (Use with Caution)

```bash
# FULL SYSTEM ACCESS - Can write anywhere
codex exec "system task" --sandbox danger-full-access

# Output: sandbox: danger-full-access
# Result: Codex can modify ANY files on system
```

**Use for:** System-level tasks requiring full access
**Warning:** Use with extreme caution!

### ❌ COMMON MISTAKES

```bash
# ❌ WRONG - Missing sandbox flag (read-only default)
codex exec "Implement SemanticChunker"
# Codex will analyze but create NO files

# ❌ WRONG - Invalid flag value
codex exec "Implement SemanticChunker" --sandbox full
# Error: invalid value 'full' (possible values: read-only, workspace-write, danger-full-access)

# ✅ CORRECT - workspace-write for implementation
codex exec "Implement SemanticChunker" --sandbox workspace-write
```

**Always use `--sandbox workspace-write` for implementation tasks!**

---

## 📊 Active Monitoring Strategy

**When delegating to Codex, ACTIVELY MONITOR progress:**

```python
import subprocess
import time

# Launch Codex in background
process = subprocess.Popen(
    ["codex", "exec", "task", "--sandbox", "workspace-write"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# ACTIVE MONITORING (every 30 seconds)
while process.poll() is None:
    time.sleep(30)

    # Check files created
    files = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True
    ).stdout

    print(f"[{time.time()}] Files changed:\n{files}")

    # Check if Codex is stuck (timeout > 10 min)
    if time.time() - start_time > 600:
        print("⚠️ Timeout - killing Codex")
        process.kill()
        break

# Get final output
stdout, stderr = process.communicate()
print(f"✅ Codex completed")
print(stdout)
```

**DON'T:** Say "будет мониторить" and forget
**DO:** Actually monitor every 30 seconds with checks

---

## 🚀 Integration Methods

### Method 1: Direct Subprocess Call (WITH FULL SANDBOX)

**When to use:** Single task delegation, simple coordination

**Python Code (for agents):**

```python
import subprocess
import json
from pathlib import Path

def call_codex(
    prompt: str,
    workdir: str = "/home/dev/zapomni",
    timeout: int = 300
) -> dict:
    """
    Call Codex CLI non-interactively.

    Args:
        prompt: Task description for Codex
        workdir: Working directory for Codex
        timeout: Timeout in seconds (default 5 minutes)

    Returns:
        dict with 'success', 'output', 'error' keys
    """
    try:
        result = subprocess.run(
            ["codex", "exec", prompt, "--sandbox", "full"],  # CRITICAL: --sandbox full
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Codex execution exceeded {timeout}s timeout",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "returncode": -1
        }

# Example usage:
result = call_codex(
    prompt="Implement function from spec: .spec-workflow/specs/level3/example.md",
    workdir="/home/dev/zapomni",
    timeout=600  # 10 minutes for complex tasks
)

if result["success"]:
    print(f"✅ Codex completed: {result['output']}")
else:
    print(f"❌ Codex failed: {result['error']}")
```

---

### Method 2: Git Branch Isolation (Recommended for Parallel Work)

**When to use:** Avoid file conflicts when both CLIs work simultaneously

**Setup:**

```python
def delegate_to_codex_isolated(
    task_name: str,
    prompt: str,
    base_branch: str = "main"
) -> dict:
    """
    Delegate task to Codex in isolated git branch.

    Returns:
        dict with 'success', 'branch', 'files_changed', 'output'
    """
    import subprocess
    import os

    # Create feature branch for Codex
    branch_name = f"feature/codex-{task_name}"

    # 1. Create and checkout branch
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    # 2. Call Codex
    result = subprocess.run(
        ["codex", "exec", prompt],
        capture_output=True,
        text=True,
        timeout=600
    )

    # 3. Check what files changed
    files_changed = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True
    ).stdout

    # 4. Return to original branch (don't auto-merge)
    subprocess.run(["git", "checkout", base_branch], check=True)

    return {
        "success": result.returncode == 0,
        "branch": branch_name,
        "files_changed": files_changed,
        "output": result.stdout,
        "error": result.stderr
    }

# Example:
result = delegate_to_codex_isolated(
    task_name="semantic-chunker",
    prompt="Implement SemanticChunker from spec: .spec-workflow/specs/level2/semantic_chunker_component.md"
)

print(f"Branch: {result['branch']}")
print(f"Files changed:\n{result['files_changed']}")
# Review changes before merging
```

---

### Method 3: Spec-Driven Task Delegation (Best Practice)

**When to use:** Large tasks with detailed specifications

**Template for Task Delegation:**

```python
def delegate_spec_task_to_codex(spec_file: str, component_name: str) -> dict:
    """
    Delegate implementation task to Codex with full spec context.

    Args:
        spec_file: Path to component or function spec
        component_name: Name of component (for git branch)

    Returns:
        dict with task results
    """
    from pathlib import Path

    # Read spec to get test count and details
    spec_path = Path(spec_file)
    if not spec_path.exists():
        return {"success": False, "error": f"Spec not found: {spec_file}"}

    # Construct detailed prompt for Codex
    prompt = f"""
TASK: Implement {component_name} Component using Test-Driven Development

SPECIFICATION: {spec_file}

INSTRUCTIONS:

1. READ SPECIFICATION:
   - Read the complete spec file: {spec_file}
   - Understand all methods, parameters, returns, exceptions
   - Note edge cases and test scenarios

2. WRITE TESTS FIRST (TDD - RED Phase):
   - Create test file: tests/unit/test_{component_name.lower()}.py
   - Implement ALL test scenarios from spec (10+ tests minimum)
   - Run pytest → tests should FAIL (no code yet)

3. IMPLEMENT CODE (GREEN Phase):
   - Create implementation: src/zapomni_core/{component_name.lower()}.py
   - Write minimal code to make tests pass
   - Follow spec signatures exactly (type hints, docstrings)

4. VERIFY TESTS PASS:
   - Run: pytest tests/unit/test_{component_name.lower()}.py
   - All tests should be GREEN
   - Check coverage: >= 90%

5. REFACTOR (if needed):
   - Improve code quality (DRY, readability)
   - Tests must stay GREEN during refactoring

6. REPORT:
   - List files created
   - Test results (X/Y passing)
   - Coverage percentage
   - Any issues encountered

REQUIREMENTS:
- Follow specification EXACTLY (signatures, types, behavior)
- TDD approach (tests before code)
- Type hints on all functions
- Google-style docstrings
- Black formatting (100 char line length)
- No commits (leave for review)

CONTEXT:
- Project: Zapomni - Local-first MCP memory system
- Foundation already implemented and tested
- You are implementing a component that uses foundation code
"""

    # Create isolated branch
    branch = f"feature/codex-{component_name.lower()}"
    subprocess.run(["git", "checkout", "-b", branch], check=False)

    # Call Codex
    result = subprocess.run(
        ["codex", "exec", prompt],
        capture_output=True,
        text=True,
        timeout=900  # 15 minutes for implementation
    )

    # Get file changes
    files = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True
    ).stdout

    return {
        "success": result.returncode == 0,
        "branch": branch,
        "output": result.stdout,
        "error": result.stderr,
        "files_changed": files
    }

# Usage:
result = delegate_spec_task_to_codex(
    spec_file=".spec-workflow/specs/level2/semantic_chunker_component.md",
    component_name="SemanticChunker"
)

if result["success"]:
    print(f"✅ Codex completed in branch: {result['branch']}")
    print(f"Files changed:\n{result['files_changed']}")
    print("Ready for review!")
else:
    print(f"❌ Codex failed: {result['error']}")
```

---

## 📊 Codex CLI Commands Reference

### Basic Commands

```bash
# Non-interactive execution (for subprocess)
codex exec "task description"

# Interactive mode (manual use)
codex "start conversation"

# Check status
codex --version
codex login --status

# MCP server mode (experimental)
codex mcp-server

# Resume previous session
codex resume
codex resume --last

# Apply changes
codex apply
```

### Useful Options

```bash
# Specify model
codex exec "task" --model gpt-5.1

# Set approval mode
codex exec "task" --approval never     # Auto-approve all
codex exec "task" --approval always    # Ask for all
codex exec "task" --approval auto      # Smart approval

# Sandbox mode
codex exec "task" --sandbox read-only  # Safe
codex exec "task" --sandbox full       # Can write files

# Reasoning effort
codex exec "task" --reasoning-effort high
```

---

## 🔄 Coordination Patterns

### Pattern 1: Sequential Pipeline (Safest)

```python
# Step 1: Claude Code creates specs
print("Phase 1: Claude Code - Creating specifications...")
# (Claude Code work)

# Step 2: Codex implements
print("Phase 2: Delegating to Codex - Implementation...")
result = call_codex("Implement from spec: ...")

# Step 3: Claude Code reviews
print("Phase 3: Claude Code - Code review...")
# (Claude Code review)
```

### Pattern 2: Parallel with Git Branches

```python
# Claude Code works in feature/claude-add-memory
subprocess.run(["git", "checkout", "-b", "feature/claude-add-memory"])
# (Claude implements add_memory)

# Codex works in feature/codex-search-memory
result = delegate_to_codex_isolated(
    task_name="search-memory",
    prompt="Implement search_memory..."
)
# Codex works in feature/codex-search-memory

# Later: Merge both branches
```

### Pattern 3: Spec → Implement → Review Chain

```python
def spec_implement_review_chain(component_name: str):
    """Complete workflow: Claude specs → Codex implements → Claude reviews."""

    # Phase 1: Claude creates spec (if not exists)
    spec_file = f".spec-workflow/specs/level2/{component_name}_component.md"
    if not Path(spec_file).exists():
        print(f"❌ Spec not found: {spec_file}")
        print("Claude Code should create spec first")
        return

    # Phase 2: Codex implements
    print(f"📤 Delegating {component_name} to Codex CLI...")
    result = delegate_spec_task_to_codex(spec_file, component_name)

    if not result["success"]:
        print(f"❌ Codex failed: {result['error']}")
        return

    # Phase 3: Claude reviews (through Task agent)
    print(f"🔍 Reviewing Codex implementation in {result['branch']}...")
    review_prompt = f"""
    Code review Codex implementation:
    - Branch: {result['branch']}
    - Files: {result['files_changed']}
    - Spec: {spec_file}

    Check:
    1. Follows spec exactly
    2. Tests pass (90%+ coverage)
    3. Type hints, docstrings
    4. No security issues

    Approve or request changes.
    """
    # Launch review agent here

    return result
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Codex Not Found

**Problem:** `bash: codex: command not found`

**Solution:**
```bash
# Install Codex CLI (if not installed)
npm install -g @openai/codex-cli

# Or check PATH
echo $PATH | grep codex
export PATH="$HOME/.nvm/versions/node/vX.X.X/bin:$PATH"
```

---

### Issue 2: Authentication Required

**Problem:** `Error: Not logged in`

**Solution:**
```bash
# Login to Codex
codex login

# Verify
codex login --status
```

---

### Issue 3: File Conflicts

**Problem:** Both Claude and Codex modify same files

**Solution:**
```python
# ALWAYS use git branches for isolation
subprocess.run(["git", "checkout", "-b", "feature/codex-task"])
# Codex works here
# Review and merge later
```

---

### Issue 4: Timeout on Large Tasks

**Problem:** Task exceeds 5-minute default timeout

**Solution:**
```python
# Increase timeout for complex tasks
result = subprocess.run(
    ["codex", "exec", prompt],
    timeout=1800  # 30 minutes
)
```

---

## 📝 Task Prompt Template

**For Claude Code agents delegating to Codex:**

```
TASK: Implement {ComponentName} Component

SPECIFICATION: {spec_file_path}

APPROACH: Test-Driven Development (TDD)

STEPS:

1. READ SPEC:
   - File: {spec_file_path}
   - Understand: Methods, parameters, returns, exceptions
   - Note: Edge cases (6+), test scenarios (10+)

2. WRITE TESTS FIRST:
   - File: tests/unit/test_{component}.py
   - Implement ALL test scenarios from spec
   - Run pytest → should FAIL (RED phase)

3. IMPLEMENT CODE:
   - File: src/{module}/{component}.py
   - Follow spec signatures exactly
   - Type hints + docstrings (Google style)
   - Make tests pass (GREEN phase)

4. VERIFY:
   - Run: pytest tests/unit/test_{component}.py -v
   - Target: 100% tests passing, 90%+ coverage
   - Format: black, isort

5. REPORT:
   - Files created: [list]
   - Tests: X/Y passing ({%} coverage)
   - Issues: [any problems]

REQUIREMENTS:
- Specification compliance: 100%
- TDD methodology (tests first)
- Type hints on all public APIs
- Black formatting (100 char lines)
- NO git commits (leave for review)

CONTEXT:
- Project: Zapomni MCP memory system
- Foundation: Already implemented (models, exceptions, DB client)
- Your role: Implement component using foundation
- Next: Claude Code will review your work

DEPENDENCIES (already available):
{list relevant foundation components}

BEGIN IMPLEMENTATION.
```

---

## 🔍 Example: Delegating SemanticChunker

**Complete example of delegating a component to Codex:**

```python
import subprocess
from pathlib import Path

# Task details
component_name = "SemanticChunker"
spec_file = ".spec-workflow/specs/level2/semantic_chunker_component.md"
output_file = "src/zapomni_core/chunking/semantic_chunker.py"
test_file = "tests/unit/test_semantic_chunker.py"

# Verify spec exists
if not Path(spec_file).exists():
    print(f"❌ Spec not found: {spec_file}")
    exit(1)

# Create isolated branch
subprocess.run(["git", "checkout", "-b", f"feature/codex-{component_name.lower()}"])

# Construct prompt
prompt = f"""
TASK: Implement SemanticChunker Component using TDD

SPECIFICATION: {spec_file}

COMPONENT OVERVIEW:
- Purpose: Split text/code into semantic chunks using LangChain
- Key Methods:
  - __init__(chunk_size=512, overlap=50)
  - chunk_text(text: str) -> List[Chunk]
  - merge_small_chunks(chunks: List[Chunk]) -> List[Chunk]

DEPENDENCIES (from spec):
- langchain>=0.1.0 (RecursiveCharacterTextSplitter)
- tiktoken>=0.5.0 (token counting)
- zapomni_db.models.Chunk (already implemented)

TDD WORKFLOW:

1. READ SPEC: {spec_file}
   - Function specs also available:
     - .spec-workflow/specs/level3/semantic_chunker_init.md
     - .spec-workflow/specs/level3/semantic_chunker_chunk_text.md

2. WRITE TESTS: {test_file}
   - 24+ test scenarios from spec
   - Test categories: basic, edge cases, performance

3. IMPLEMENT: {output_file}
   - Follow spec signatures exactly
   - Use LangChain RecursiveCharacterTextSplitter
   - Separators: ["\\n\\n", "\\n", " ", ""]

4. VERIFY:
   - pytest {test_file} -v
   - Coverage >= 90%

5. REPORT RESULTS

IMPORTANT:
- Foundation code in src/zapomni_db/models.py (Chunk model)
- Use Pydantic models from zapomni_db.models
- NO commits (review first)

START IMPLEMENTATION.
"""

# Execute Codex
print(f"📤 Delegating to Codex CLI: {component_name}")
print(f"Branch: feature/codex-{component_name.lower()}")
print(f"Timeout: 10 minutes")
print("=" * 60)

result = subprocess.run(
    ["codex", "exec", prompt],
    capture_output=True,
    text=True,
    timeout=600
)

# Parse results
print("\n" + "=" * 60)
print("📥 CODEX RESULTS:")
print("=" * 60)
print(result.stdout)

if result.returncode == 0:
    print("\n✅ Codex completed successfully!")

    # Check what files created
    files = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True
    ).stdout

    print(f"\nFiles changed:\n{files}")
    print(f"\nBranch: feature/codex-{component_name.lower()}")
    print("Ready for Claude Code review!")
else:
    print(f"\n❌ Codex failed with code: {result.returncode}")
    print(f"Error: {result.stderr}")
```

---

## 📚 Best Practices

### 1. Always Use Git Branches

```python
# ✅ GOOD: Isolated branch
git checkout -b feature/codex-task
codex exec "implement..."
# Review before merge

# ❌ BAD: Direct to main
codex exec "implement..."  # Changes main directly
```

### 2. Provide Complete Context

```python
# ✅ GOOD: Full context
prompt = f"""
TASK: Implement X
SPEC: {spec_file}
DEPENDENCIES: {list deps}
OUTPUT: {expected_files}
TDD: tests first, then code
"""

# ❌ BAD: Vague
prompt = "implement component"  # No context
```

### 3. Set Appropriate Timeouts

```python
# ✅ GOOD: Task-based timeout
simple_task_timeout = 300    # 5 min
complex_task_timeout = 1800  # 30 min

# ❌ BAD: Too short
timeout = 60  # May kill Codex mid-task
```

### 4. Review Before Merging

```python
# ✅ GOOD: Always review
result = delegate_to_codex(...)
# Claude Code reviews Codex output
# Approve → merge

# ❌ BAD: Auto-merge
result = delegate_to_codex(...)
subprocess.run(["git", "merge", result["branch"]])  # No review!
```

---

## 🎯 Integration with Claude Code Agent Workflow

**In AGENT_WORKFLOW.md, add Codex delegation pattern:**

```markdown
### 🤖 Codex CLI Integration (Optional)

For bulk code generation tasks:

1. **Claude Code** creates detailed specification
2. **Codex CLI** implements code from spec (subprocess call)
3. **Claude Code** reviews implementation (Task agent)
4. **User** approves merge

**When to delegate to Codex:**
- Large "one-shot" code generation (500+ lines)
- Bulk implementation from detailed specs
- Rapid prototyping

**When to keep with Claude Code:**
- Architecture decisions
- Spec creation
- Code review
- Complex reasoning
- Multi-step planning
```

---

## 🔐 Security Considerations

### 1. Sandbox Mode

```bash
# For untrusted tasks, use read-only sandbox
codex exec "analyze code" --sandbox read-only

# For implementation, use full access (but review output)
codex exec "implement component" --sandbox full
```

### 2. Approval Mode

```bash
# Auto-approve (for non-destructive tasks)
codex exec "create tests" --approval never

# Manual approval (for file modifications)
codex exec "implement feature" --approval always
```

### 3. Validate Output

```python
# After Codex completes, validate:
# 1. No malicious code
# 2. Follows spec
# 3. Tests pass
# 4. No secrets in code
```

---

## 📊 Performance Characteristics

**Codex CLI Performance (observed):**
- Simple task (< 100 lines): 30-60 seconds
- Medium task (100-500 lines): 2-5 minutes
- Complex task (500+ lines): 5-15 minutes

**Token Usage:**
- Codex uses GPT-5.1 (expensive but powerful)
- Example: Simple "Hello" used 12,277 tokens
- Budget accordingly for complex tasks

---

## 🎓 Quick Start for Claude Code Agents

**Copy-paste this into agent prompts:**

```python
# Quick Codex delegation template
import subprocess

def quick_codex_task(task_desc: str) -> str:
    """Quick Codex delegation for simple tasks."""
    result = subprocess.run(
        ["codex", "exec", task_desc],
        capture_output=True,
        text=True,
        timeout=300
    )
    return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

# Usage in agent:
output = quick_codex_task("Implement helper function from docstring in file.py")
print(output)
```

---

## 📖 References

**Codex CLI:**
- GitHub: https://github.com/openai/codex
- Docs: https://developers.openai.com/codex/cli
- MCP Support: https://developers.openai.com/codex/mcp/

**Integration Examples:**
- Codex MCP Server: https://lobehub.com/mcp/mr-tomahawk-codex-cli-mcp-tool
- Multi-agent workflows: https://skywork.ai/skypage/en/Codex-as-Mcp-MCP-Server

**Claude Code CLI:**
- GitHub: https://github.com/anthropics/claude-code

---

## 🚀 Summary

**✅ CAN DO:**
- Call Codex via subprocess: `subprocess.run(["codex", "exec", "..."])`
- Delegate tasks programmatically
- Use git branches for isolation
- Review Codex output with Claude Code agents

**⚠️ CONSIDERATIONS:**
- Requires Codex CLI installed + authenticated
- Use git branches to avoid conflicts
- Always review before merging
- Set appropriate timeouts

**🎯 BEST USE CASE:**
- Claude Code: Architecture, planning, specs, review
- Codex CLI: Bulk implementation from specs
- Together: Faster development with quality control

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Project:** Zapomni
**Last Updated:** 2025-11-23

*This guide enables Claude Code CLI agents to programmatically delegate tasks to Codex CLI for efficient multi-agent development workflows.*
