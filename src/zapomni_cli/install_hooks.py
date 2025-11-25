"""
Git hooks installation for Zapomni.

Installs post-commit, post-merge, and post-checkout hooks
that trigger automatic re-indexing on file changes.
"""

import os
import shutil
import stat
from pathlib import Path
from typing import List

# Hook templates directory
HOOKS_DIR = Path(__file__).parent / "hooks"

# Hooks to install
HOOKS_TO_INSTALL = [
    "post-commit",
    "post-merge",
    "post-checkout",
]


def find_git_dir(repo_path: Path) -> Path:
    """
    Find .git directory.

    Args:
        repo_path: Repository root path

    Returns:
        Path to .git directory

    Raises:
        ValueError: If not a Git repository
    """
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"{repo_path} is not a Git repository")

    return git_dir


def install_hooks_command(repo_path: Path) -> bool:
    """
    Install Git hooks in repository.

    Args:
        repo_path: Path to Git repository

    Returns:
        True if successful, False otherwise
    """
    try:
        # Find .git directory
        git_dir = find_git_dir(repo_path)
        hooks_dir = git_dir / "hooks"

        # Ensure hooks directory exists
        hooks_dir.mkdir(exist_ok=True)

        # Install each hook
        installed = []
        for hook_name in HOOKS_TO_INSTALL:
            hook_template = HOOKS_DIR / hook_name
            hook_dest = hooks_dir / hook_name

            if not hook_template.exists():
                print(f"⚠️  Hook template not found: {hook_name}")
                continue

            # Backup existing hook if present
            if hook_dest.exists():
                backup = hook_dest.with_suffix(".backup")
                shutil.copy2(hook_dest, backup)
                print(f"📦 Backed up existing {hook_name} to {backup.name}")

            # Copy hook script
            shutil.copy2(hook_template, hook_dest)

            # Make executable
            hook_dest.chmod(hook_dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            installed.append(hook_name)
            print(f"✅ Installed {hook_name}")

        if installed:
            print(f"\n🎉 Successfully installed {len(installed)} Git hooks!")
            print(f"📂 Repository: {repo_path.absolute()}")
            print("\nGit operations (commit, merge, checkout) will now trigger")
            print("automatic re-indexing of changed files.")
            return True
        else:
            print("❌ No hooks were installed")
            return False

    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
