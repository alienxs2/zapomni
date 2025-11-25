"""
Tests for Git hooks installation CLI.
"""

import tempfile
from pathlib import Path

import pytest

from zapomni_cli.install_hooks import find_git_dir, install_hooks_command


class TestFindGitDir:
    """Test .git directory detection."""

    def test_find_git_dir_success(self):
        """Test finding .git in valid repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            git_dir = repo / ".git"
            git_dir.mkdir()

            result = find_git_dir(repo)
            assert result == git_dir

    def test_find_git_dir_not_repo(self):
        """Test error when not a Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)

            with pytest.raises(ValueError, match="not a Git repository"):
                find_git_dir(repo)


class TestInstallHooksCommand:
    """Test hooks installation."""

    def test_install_hooks_not_git_repo(self):
        """Test installation fails in non-Git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = install_hooks_command(Path(tmpdir))
            assert result is False

    def test_install_hooks_success(self):
        """Test successful hooks installation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            git_dir = repo / ".git"
            git_dir.mkdir()

            result = install_hooks_command(repo)
            assert result is True

            # Verify hooks were installed
            hooks_dir = git_dir / "hooks"
            assert hooks_dir.exists()

            for hook_name in ["post-commit", "post-merge", "post-checkout"]:
                hook_file = hooks_dir / hook_name
                assert hook_file.exists()
                # Verify executable
                assert hook_file.stat().st_mode & 0o111  # Has execute bit
