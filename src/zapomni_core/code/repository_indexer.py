"""
CodeRepositoryIndexer for indexing Git repositories and extracting code files.

Provides functionality to:
- Index entire Git repositories
- Extract code files with extension filtering
- Collect git metadata (author, commits, timestamps)
- Respect .gitignore patterns
- Calculate file statistics (size, line count)

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.utils import get_logger

logger = get_logger(__name__)

# Default code file extensions
DEFAULT_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
}

# Files and patterns to always ignore
DEFAULT_IGNORE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".egg-info",
    "dist",
    "build",
    ".git",
    ".gitignore",
    ".DS_Store",
    "Thumbs.db",
    ".vscode",
    ".idea",
}


class CodeRepositoryIndexer:
    """
    Indexes Git repositories and extracts code files for processing.

    Handles repository scanning, file filtering, git metadata extraction,
    and gitignore pattern matching.

    Attributes:
        config: Configuration dictionary
        code_extensions: Set of code file extensions to include
        ignore_patterns: Set of patterns to ignore during indexing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CodeRepositoryIndexer.

        Args:
            config: Configuration dictionary with optional keys:
                - code_extensions: Set of extensions to include (default: DEFAULT_CODE_EXTENSIONS)
                - ignore_patterns: Set of patterns to ignore (default: DEFAULT_IGNORE_PATTERNS)
                - max_file_size: Maximum file size in bytes (default: 10MB)
                - follow_symlinks: Whether to follow symlinks (default: False)

        Raises:
            ValidationError: If config is invalid
        """
        self.config = config or {}
        self._validate_config()

        self.code_extensions = self.config.get("code_extensions", DEFAULT_CODE_EXTENSIONS)
        self.ignore_patterns = self.config.get("ignore_patterns", DEFAULT_IGNORE_PATTERNS)
        self.max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.follow_symlinks = self.config.get("follow_symlinks", False)

        logger.info(
            "repository_indexer_initialized",
            extensions=len(self.code_extensions),
            ignore_patterns=len(self.ignore_patterns),
            max_file_size=self.max_file_size,
        )

    def _validate_config(self) -> None:
        """
        Validate configuration dictionary.

        Raises:
            ValidationError: If config is invalid
        """
        if not isinstance(self.config, dict):
            raise ValidationError(
                "Configuration must be a dictionary",
                error_code="VAL_001",
                details={"received_type": type(self.config).__name__},
            )

        if "max_file_size" in self.config:
            if (
                not isinstance(self.config["max_file_size"], int)
                or self.config["max_file_size"] <= 0
            ):
                raise ValidationError(
                    "max_file_size must be a positive integer",
                    error_code="VAL_003",
                    details={"received_value": self.config["max_file_size"]},
                )

    def index_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Index entire Git repository.

        Scans the repository, collects code files, and gathers statistics.

        Args:
            repo_path: Path to the Git repository

        Returns:
            Dictionary with structure:
            {
                "repository": {
                    "path": str,
                    "name": str,
                    "git_url": Optional[str],
                    "default_branch": str,
                },
                "files": List[Dict] - from get_code_files(),
                "statistics": {
                    "total_files": int,
                    "total_lines": int,
                    "total_size": int,
                    "by_extension": {ext: count}
                }
            }

        Raises:
            ValidationError: If repo_path is invalid
            ProcessingError: If repository cannot be read
        """
        repo_path = self._validate_repo_path(repo_path)

        try:
            # Get code files
            files = self.get_code_files(repo_path)

            # Calculate statistics
            stats = self._calculate_statistics(files)

            # Get git info
            git_info = self._get_git_info(repo_path)

            result = {
                "repository": {
                    "path": str(repo_path),
                    "name": repo_path.name,
                    "git_url": git_info.get("url"),
                    "default_branch": git_info.get("default_branch", "main"),
                },
                "files": files,
                "statistics": stats,
            }

            logger.info(
                "repository_indexed",
                path=str(repo_path),
                file_count=len(files),
                total_lines=stats["total_lines"],
            )

            return result

        except ProcessingError:
            raise
        except Exception as exc:
            logger.error(
                "repository_indexing_failed",
                path=str(repo_path),
                error=str(exc),
            )
            raise ProcessingError(
                message=f"Failed to index repository at {repo_path}",
                error_code="PROC_003",
                details={"path": str(repo_path)},
                original_exception=exc,
            )

    def get_code_files(
        self,
        repo_path: str,
        extensions: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all code files in repository matching extensions.

        Args:
            repo_path: Path to repository
            extensions: Set of extensions to include (default: configured extensions)

        Returns:
            List of file dictionaries with structure:
            {
                "path": str,
                "relative_path": str,
                "extension": str,
                "size_bytes": int,
                "lines": int,
                "encoding": str,
                "last_modified": float,
            }

        Raises:
            ValidationError: If repo_path is invalid
            ProcessingError: If files cannot be read
        """
        repo_path = self._validate_repo_path(repo_path)
        extensions = extensions or self.code_extensions

        try:
            files = []
            gitignore_patterns = self._load_gitignore(repo_path)

            for file_path in repo_path.rglob("*"):
                # Check if it's a file (handle symlinks based on config)
                if not self._is_file(file_path):
                    continue

                # Check if file should be included
                if not self._should_include_file(file_path, repo_path, extensions):
                    continue

                # Check gitignore patterns
                relative_path = file_path.relative_to(repo_path)
                if self._is_ignored(str(relative_path), gitignore_patterns):
                    continue

                # Get file metadata
                try:
                    file_info = self.get_file_metadata(str(file_path), repo_path)
                    files.append(file_info)
                except Exception as exc:
                    logger.warning(
                        "file_metadata_extraction_failed",
                        path=str(file_path),
                        error=str(exc),
                    )
                    continue

            logger.info("code_files_collected", count=len(files), repo=str(repo_path))
            return sorted(files, key=lambda f: f["relative_path"])

        except ProcessingError:
            raise
        except Exception as exc:
            logger.error(
                "code_files_collection_failed",
                path=str(repo_path),
                error=str(exc),
            )
            raise ProcessingError(
                message=f"Failed to collect code files from {repo_path}",
                error_code="PROC_003",
                details={"path": str(repo_path)},
                original_exception=exc,
            )

    def get_file_metadata(
        self,
        file_path: str,
        repo_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata for a code file.

        Includes git blame info, file statistics, and basic properties.

        Args:
            file_path: Absolute path to file
            repo_root: Optional repository root for relative path calculation

        Returns:
            Dictionary with structure:
            {
                "path": str,
                "relative_path": str,
                "extension": str,
                "size_bytes": int,
                "lines": int,
                "encoding": str,
                "last_modified": float,
                "git_info": {
                    "last_author": Optional[str],
                    "last_commit": Optional[str],
                    "last_commit_date": Optional[float],
                    "blame_info": Optional[Dict]
                }
            }

        Raises:
            ValidationError: If file_path is invalid
            ProcessingError: If file cannot be read
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.is_file():
            raise ValidationError(
                f"File does not exist: {file_path}",
                error_code="VAL_001",
                details={"path": file_path},
            )

        try:
            # Get basic file stats
            stat_info = file_path_obj.stat()
            size_bytes = stat_info.st_size

            # Get encoding and line count
            encoding = self._detect_encoding(file_path_obj)
            lines = self._count_lines(file_path_obj, encoding)

            # Get git info if in a repository
            git_info = self._get_git_file_info(file_path)

            # Find repository root to calculate relative path
            if repo_root is None:
                repo_root = self._find_git_root(file_path_obj)

            if repo_root and file_path_obj.is_relative_to(repo_root):
                relative_path = file_path_obj.relative_to(repo_root)
            else:
                relative_path = file_path_obj.name

            result = {
                "path": str(file_path_obj.absolute()),
                "relative_path": str(relative_path),
                "extension": file_path_obj.suffix.lower(),
                "size_bytes": size_bytes,
                "lines": lines,
                "encoding": encoding,
                "last_modified": stat_info.st_mtime,
                "git_info": git_info,
            }

            logger.debug(
                "file_metadata_extracted",
                path=str(file_path),
                lines=lines,
                size=size_bytes,
            )

            return result

        except ValidationError:
            raise
        except Exception as exc:
            logger.error(
                "file_metadata_extraction_error",
                path=file_path,
                error=str(exc),
            )
            raise ProcessingError(
                message=f"Failed to extract metadata for {file_path}",
                error_code="PROC_003",
                details={"path": file_path},
                original_exception=exc,
            )

    def filter_ignored_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out ignored files from a list.

        Applies .gitignore patterns and default ignore patterns.

        Args:
            files: List of file dictionaries (from get_code_files)

        Returns:
            Filtered list of files that should be included

        Raises:
            ValidationError: If files list is invalid
        """
        if not isinstance(files, list):
            raise ValidationError(
                "files must be a list",
                error_code="VAL_002",
                details={"received_type": type(files).__name__},
            )

        try:
            filtered = []

            for file_dict in files:
                if not isinstance(file_dict, dict):
                    logger.warning("skipping_non_dict_file_entry")
                    continue

                relative_path = file_dict.get("relative_path", "")

                # Apply default ignore patterns
                if self._matches_ignore_patterns(relative_path, self.ignore_patterns):
                    continue

                filtered.append(file_dict)

            logger.info(
                "files_filtered",
                total=len(files),
                filtered=len(filtered),
                removed=len(files) - len(filtered),
            )

            return filtered

        except Exception as exc:
            logger.error("file_filtering_failed", error=str(exc))
            raise ProcessingError(
                message="Failed to filter ignored files",
                error_code="PROC_001",
                original_exception=exc,
            )

    # Private helper methods

    def _is_file(self, file_path: Path) -> bool:
        """
        Check if path is a file, respecting symlink settings.

        Args:
            file_path: Path to check

        Returns:
            True if path is a file
        """
        try:
            if self.follow_symlinks:
                return file_path.is_file()
            else:
                # Don't follow symlinks
                return file_path.is_file() and not file_path.is_symlink()
        except (OSError, PermissionError):
            return False

    def _validate_repo_path(self, repo_path: str) -> Path:
        """
        Validate and convert repository path to Path object.

        Args:
            repo_path: Path to repository

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid or not a directory
        """
        if not isinstance(repo_path, (str, Path)):
            raise ValidationError(
                "repo_path must be a string or Path",
                error_code="VAL_002",
                details={"received_type": type(repo_path).__name__},
            )

        path_obj = Path(repo_path).expanduser().resolve()

        if not path_obj.exists():
            raise ValidationError(
                f"Repository path does not exist: {repo_path}",
                error_code="VAL_001",
                details={"path": repo_path},
            )

        if not path_obj.is_dir():
            raise ValidationError(
                f"Repository path is not a directory: {repo_path}",
                error_code="VAL_001",
                details={"path": repo_path},
            )

        return path_obj

    def _should_include_file(
        self,
        file_path: Path,
        repo_root: Path,
        extensions: Set[str],
    ) -> bool:
        """Check if file should be included based on extension and size."""
        if file_path.suffix.lower() not in extensions:
            return False

        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                logger.debug(
                    "file_too_large",
                    path=str(file_path),
                    size=file_path.stat().st_size,
                )
                return False
        except OSError:
            return False

        return True

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            # Try UTF-8 first
            file_path.read_text(encoding="utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        # Try other common encodings
        for encoding in ["latin-1", "iso-8859-1", "cp1252"]:
            try:
                file_path.read_text(encoding=encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        return "unknown"

    def _count_lines(self, file_path: Path, encoding: str) -> int:
        """Count lines in a file."""
        try:
            if encoding == "unknown":
                return 0

            with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _load_gitignore(self, repo_path: Path) -> List[str]:
        """
        Load .gitignore patterns from repository.

        Args:
            repo_path: Path to repository

        Returns:
            List of gitignore patterns
        """
        gitignore_path = repo_path / ".gitignore"

        if not gitignore_path.exists():
            return []

        try:
            patterns = []
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        patterns.append(line)

            logger.debug("gitignore_loaded", path=str(gitignore_path), patterns=len(patterns))
            return patterns

        except Exception as exc:
            logger.warning(
                "gitignore_load_failed",
                path=str(gitignore_path),
                error=str(exc),
            )
            return []

    def _is_ignored(self, file_path: str, gitignore_patterns: List[str]) -> bool:
        """Check if file matches any gitignore pattern."""
        for pattern in gitignore_patterns:
            if self._pattern_matches(file_path, pattern):
                return True
        return False

    def _matches_ignore_patterns(
        self,
        file_path: str,
        ignore_patterns: Set[str],
    ) -> bool:
        """Check if file matches any default ignore patterns."""
        for pattern in ignore_patterns:
            if self._pattern_matches(file_path, pattern):
                return True
        return False

    def _pattern_matches(self, file_path: str, pattern: str) -> bool:
        """
        Match a file path against a gitignore-style pattern.

        Supports simple wildcards and directory patterns.
        """
        # Handle negation
        if pattern.startswith("!"):
            return False

        # Convert gitignore pattern to regex
        # This is a simplified implementation
        if "*" not in pattern and "/" not in pattern:
            # Simple filename matching
            return file_path.endswith(pattern) or file_path.endswith(f"/{pattern}")

        # Handle directory patterns (e.g., __pycache__/)
        if pattern.endswith("/"):
            dir_pattern = pattern.rstrip("/")
            # Match directory anywhere in path or at the beginning
            return (
                f"/{dir_pattern}/" in f"/{file_path}/"
                or file_path.startswith(dir_pattern + "/")
                or file_path.endswith("/" + dir_pattern)
            )

        # Handle wildcard patterns
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        regex_pattern = f"^.*{regex_pattern}$|^.*/{regex_pattern}$"

        try:
            return bool(re.match(regex_pattern, file_path))
        except re.error:
            return False

    def _find_git_root(self, file_path: Path) -> Optional[Path]:
        """Find the git repository root for a file."""
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent

        return None

    def _get_git_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get git metadata for a file.

        Attempts to extract author and commit info.
        Falls back gracefully if git is not available.
        """
        try:
            import subprocess

            file_path_obj = Path(file_path)
            repo_root = self._find_git_root(file_path_obj)

            if not repo_root:
                return {
                    "last_author": None,
                    "last_commit": None,
                    "last_commit_date": None,
                    "blame_info": None,
                }

            # Try to get last commit info
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%an|%H|%aI", "--", file_path],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split("|")
                    if len(parts) >= 2:
                        return {
                            "last_author": parts[0],
                            "last_commit": parts[1],
                            "last_commit_date": parts[2] if len(parts) > 2 else None,
                            "blame_info": None,
                        }
            except Exception:
                pass

        except Exception:
            pass

        return {
            "last_author": None,
            "last_commit": None,
            "last_commit_date": None,
            "blame_info": None,
        }

    def _get_git_info(self, repo_path: Path) -> Dict[str, Any]:
        """Get git repository information."""
        try:
            import subprocess

            git_url = None
            default_branch = "main"

            # Try to get remote URL
            try:
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    git_url = result.stdout.strip() or None
            except Exception:
                pass

            # Try to get default branch
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    default_branch = result.stdout.strip()
            except Exception:
                pass

            return {
                "url": git_url,
                "default_branch": default_branch,
            }

        except Exception:
            return {
                "url": None,
                "default_branch": "main",
            }

    def _calculate_statistics(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from file list."""
        stats = {
            "total_files": len(files),
            "total_lines": 0,
            "total_size": 0,
            "by_extension": {},
        }

        for file_info in files:
            stats["total_lines"] += file_info.get("lines", 0)
            stats["total_size"] += file_info.get("size_bytes", 0)

            ext = file_info.get("extension", "unknown")
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

        return stats
