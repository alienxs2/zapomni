"""
Unit tests for CodeRepositoryIndexer component.

Tests repository scanning, file filtering, metadata extraction,
and gitignore pattern matching.

Follows TDD approach with comprehensive mocking of dependencies.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from typing import Dict, Any
import pytest

from zapomni_core.code.repository_indexer import (
    CodeRepositoryIndexer,
    DEFAULT_CODE_EXTENSIONS,
    DEFAULT_IGNORE_PATTERNS,
)
from zapomni_core.exceptions import ValidationError, ProcessingError


class TestCodeRepositoryIndexerInit:
    """Tests for CodeRepositoryIndexer.__init__ configuration and validation."""

    def test_init_defaults_success(self):
        """Default initialization should create instance with default config."""
        indexer = CodeRepositoryIndexer()

        assert indexer is not None
        assert indexer.config == {}
        assert indexer.code_extensions == DEFAULT_CODE_EXTENSIONS
        assert indexer.ignore_patterns == DEFAULT_IGNORE_PATTERNS
        assert indexer.max_file_size == 10 * 1024 * 1024
        assert indexer.follow_symlinks is False

    def test_init_custom_config_success(self):
        """Custom config should override defaults."""
        custom_extensions = {".py", ".js"}
        custom_ignore = {"*.pyc"}
        custom_size = 5 * 1024 * 1024

        config = {
            "code_extensions": custom_extensions,
            "ignore_patterns": custom_ignore,
            "max_file_size": custom_size,
            "follow_symlinks": True,
        }

        indexer = CodeRepositoryIndexer(config)

        assert indexer.code_extensions == custom_extensions
        assert indexer.ignore_patterns == custom_ignore
        assert indexer.max_file_size == custom_size
        assert indexer.follow_symlinks is True

    def test_init_invalid_config_type(self):
        """Invalid config type should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodeRepositoryIndexer(config="invalid")

        assert exc_info.value.error_code == "VAL_001"

    def test_init_invalid_max_file_size(self):
        """Invalid max_file_size should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodeRepositoryIndexer(config={"max_file_size": -1})

        assert exc_info.value.error_code == "VAL_003"

    def test_init_invalid_max_file_size_zero(self):
        """max_file_size of 0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodeRepositoryIndexer(config={"max_file_size": 0})

        assert exc_info.value.error_code == "VAL_003"

    def test_init_invalid_max_file_size_type(self):
        """Non-integer max_file_size should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodeRepositoryIndexer(config={"max_file_size": "not_an_int"})

        assert exc_info.value.error_code == "VAL_003"


class TestCodeRepositoryIndexerValidateRepoPath:
    """Tests for repository path validation."""

    def test_validate_repo_path_string_success(self):
        """Valid string path should be converted to Path object."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = indexer._validate_repo_path(tmpdir)
            assert isinstance(result, Path)
            assert result.exists()

    def test_validate_repo_path_path_object_success(self):
        """Valid Path object should be returned as is."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path_obj = Path(tmpdir)
            result = indexer._validate_repo_path(path_obj)
            assert isinstance(result, Path)
            assert result.exists()

    def test_validate_repo_path_nonexistent(self):
        """Nonexistent path should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError) as exc_info:
            indexer._validate_repo_path("/nonexistent/path")

        assert exc_info.value.error_code == "VAL_001"

    def test_validate_repo_path_not_directory(self):
        """File path (not directory) should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValidationError) as exc_info:
                indexer._validate_repo_path(tmpfile.name)

            assert exc_info.value.error_code == "VAL_001"

    def test_validate_repo_path_invalid_type(self):
        """Non-string/Path type should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError) as exc_info:
            indexer._validate_repo_path(12345)

        assert exc_info.value.error_code == "VAL_002"

    def test_validate_repo_path_expands_user(self):
        """Path with ~ should be expanded."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path and verify it expands properly
            result = indexer._validate_repo_path(tmpdir)
            assert "~" not in str(result)


class TestCodeRepositoryIndexerGetCodeFiles:
    """Tests for code file collection."""

    def test_get_code_files_empty_directory(self):
        """Empty directory should return empty list."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = indexer.get_code_files(tmpdir)
            assert files == []

    def test_get_code_files_single_python_file(self):
        """Single Python file should be found."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")

            files = indexer.get_code_files(tmpdir)
            assert len(files) == 1
            assert files[0]["extension"] == ".py"
            assert files[0]["relative_path"] == "test.py"

    def test_get_code_files_multiple_extensions(self):
        """Multiple file types should be found."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            py_file = Path(tmpdir) / "script.py"
            py_file.write_text("print('hello')")

            js_file = Path(tmpdir) / "script.js"
            js_file.write_text("console.log('hello');")

            txt_file = Path(tmpdir) / "readme.txt"
            txt_file.write_text("readme")

            files = indexer.get_code_files(tmpdir)
            # Should find py and js, but not txt
            assert len(files) == 2
            extensions = {f["extension"] for f in files}
            assert extensions == {".py", ".js"}

    def test_get_code_files_custom_extensions(self):
        """Custom extensions should be respected."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            py_file = Path(tmpdir) / "script.py"
            py_file.write_text("print('hello')")

            java_file = Path(tmpdir) / "Main.java"
            java_file.write_text("public class Main {}")

            files = indexer.get_code_files(tmpdir, extensions={".java"})
            assert len(files) == 1
            assert files[0]["extension"] == ".java"

    def test_get_code_files_respects_max_size(self):
        """Files exceeding max_file_size should be excluded."""
        config = {"max_file_size": 100}  # 100 bytes
        indexer = CodeRepositoryIndexer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small file
            small_file = Path(tmpdir) / "small.py"
            small_file.write_text("x = 1")

            # Create a large file
            large_file = Path(tmpdir) / "large.py"
            large_file.write_text("x = " + "1" * 200)

            files = indexer.get_code_files(tmpdir)
            assert len(files) == 1
            assert files[0]["relative_path"] == "small.py"

    def test_get_code_files_nested_structure(self):
        """Nested directory structure should be scanned."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("def main(): pass")

            sub_dir = src_dir / "utils"
            sub_dir.mkdir()
            (sub_dir / "helpers.py").write_text("def helper(): pass")

            files = indexer.get_code_files(tmpdir)
            assert len(files) == 2
            paths = {f["relative_path"] for f in files}
            assert "src/main.py" in paths or "src\\main.py" in paths
            assert "src/utils/helpers.py" in paths or "src\\utils\\helpers.py" in paths

    def test_get_code_files_invalid_repo_path(self):
        """Invalid repo path should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError):
            indexer.get_code_files("/nonexistent/path")

    def test_get_code_files_with_gitignore(self):
        """Files matching .gitignore should be excluded."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .gitignore
            gitignore = Path(tmpdir) / ".gitignore"
            gitignore.write_text("*.pyc\n__pycache__\n")

            # Create files
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "cache.pyc").write_text("compiled")

            files = indexer.get_code_files(tmpdir)
            # Should find main.py but exclude .pyc files
            assert any(f["relative_path"] == "main.py" for f in files)
            assert not any(f["relative_path"] == "cache.pyc" for f in files)


class TestCodeRepositoryIndexerGetFileMetadata:
    """Tests for file metadata extraction."""

    def test_get_file_metadata_python_file(self):
        """Python file metadata should be extracted correctly."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')\nprint('world')")
            f.flush()
            file_path = f.name

        try:
            metadata = indexer.get_file_metadata(file_path)

            assert metadata["extension"] == ".py"
            assert metadata["size_bytes"] > 0
            assert metadata["lines"] == 2
            assert metadata["encoding"] == "utf-8"
            assert metadata["path"] == Path(file_path).absolute().as_posix()
            assert "git_info" in metadata
        finally:
            os.unlink(file_path)

    def test_get_file_metadata_nonexistent_file(self):
        """Nonexistent file should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError) as exc_info:
            indexer.get_file_metadata("/nonexistent/file.py")

        assert exc_info.value.error_code == "VAL_001"

    def test_get_file_metadata_line_counting(self):
        """Line count should be accurate."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            f.flush()
            file_path = f.name

        try:
            metadata = indexer.get_file_metadata(file_path)
            assert metadata["lines"] == 3
        finally:
            os.unlink(file_path)

    def test_get_file_metadata_file_size(self):
        """File size should be accurate."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = "x" * 1000
            f.write(content)
            f.flush()
            file_path = f.name

        try:
            metadata = indexer.get_file_metadata(file_path)
            assert metadata["size_bytes"] == 1000
        finally:
            os.unlink(file_path)

    def test_get_file_metadata_empty_file(self):
        """Empty file should have 0 lines."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.flush()
            file_path = f.name

        try:
            metadata = indexer.get_file_metadata(file_path)
            assert metadata["lines"] == 0
            assert metadata["size_bytes"] == 0
        finally:
            os.unlink(file_path)


class TestCodeRepositoryIndexerFilterIgnoredFiles:
    """Tests for filtering ignored files."""

    def test_filter_ignored_files_empty_list(self):
        """Empty list should return empty list."""
        indexer = CodeRepositoryIndexer()
        result = indexer.filter_ignored_files([])
        assert result == []

    def test_filter_ignored_files_no_matches(self):
        """Files not matching ignore patterns should be kept."""
        indexer = CodeRepositoryIndexer()

        files = [
            {"relative_path": "main.py"},
            {"relative_path": "utils.py"},
        ]

        result = indexer.filter_ignored_files(files)
        assert len(result) == 2

    def test_filter_ignored_files_with_matches(self):
        """Files matching ignore patterns should be removed."""
        indexer = CodeRepositoryIndexer()

        files = [
            {"relative_path": "main.py"},
            {"relative_path": "__pycache__/cache.pyc"},
            {"relative_path": "build/output.pyc"},
        ]

        result = indexer.filter_ignored_files(files)
        # Should keep only main.py
        assert len(result) == 1
        assert result[0]["relative_path"] == "main.py"

    def test_filter_ignored_files_custom_patterns(self):
        """Custom ignore patterns should be applied."""
        config = {"ignore_patterns": {"*.tmp", "test_*"}}
        indexer = CodeRepositoryIndexer(config)

        files = [
            {"relative_path": "main.py"},
            {"relative_path": "cache.tmp"},
            {"relative_path": "test_file.py"},
        ]

        result = indexer.filter_ignored_files(files)
        assert len(result) == 1
        assert result[0]["relative_path"] == "main.py"

    def test_filter_ignored_files_invalid_input(self):
        """Invalid input should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError) as exc_info:
            indexer.filter_ignored_files("not a list")

        assert exc_info.value.error_code == "VAL_002"

    def test_filter_ignored_files_non_dict_entry(self):
        """Non-dict entries should be skipped gracefully."""
        indexer = CodeRepositoryIndexer()

        files = [
            {"relative_path": "main.py"},
            "invalid_entry",
            {"relative_path": "utils.py"},
        ]

        result = indexer.filter_ignored_files(files)
        # Should skip invalid entry
        assert len(result) == 2


class TestCodeRepositoryIndexerPatternMatching:
    """Tests for .gitignore pattern matching."""

    def test_pattern_matching_simple_extension(self):
        """Simple extension pattern should match."""
        indexer = CodeRepositoryIndexer()

        assert indexer._pattern_matches("file.pyc", "*.pyc")
        assert indexer._pattern_matches("build/file.pyc", "*.pyc")
        assert not indexer._pattern_matches("file.py", "*.pyc")

    def test_pattern_matching_directory(self):
        """Directory pattern should match directory contents."""
        indexer = CodeRepositoryIndexer()

        assert indexer._pattern_matches("__pycache__/file", "__pycache__/")
        assert indexer._pattern_matches("dir/__pycache__/file", "__pycache__/")
        assert not indexer._pattern_matches("pycache/file", "__pycache__/")

    def test_pattern_matching_exact_filename(self):
        """Exact filename pattern should match."""
        indexer = CodeRepositoryIndexer()

        assert indexer._pattern_matches("file.txt", "file.txt")
        assert indexer._pattern_matches("dir/file.txt", "file.txt")
        assert not indexer._pattern_matches("file.py", "file.txt")

    def test_pattern_matching_negation(self):
        """Negation patterns should not match."""
        indexer = CodeRepositoryIndexer()

        assert not indexer._pattern_matches("file.py", "!*.py")


class TestCodeRepositoryIndexerIndexRepository:
    """Tests for full repository indexing."""

    def test_index_repository_empty(self):
        """Empty repository should return valid structure."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = indexer.index_repository(tmpdir)

            assert "repository" in result
            assert "files" in result
            assert "statistics" in result
            assert result["files"] == []
            assert result["statistics"]["total_files"] == 0

    def test_index_repository_with_files(self):
        """Repository with files should return correct structure."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "utils.py").write_text("def helper():\n    pass")

            result = indexer.index_repository(tmpdir)

            assert result["repository"]["path"] == tmpdir
            assert result["repository"]["name"] == Path(tmpdir).name
            assert len(result["files"]) == 2
            assert result["statistics"]["total_files"] == 2
            assert result["statistics"]["total_lines"] > 0

    def test_index_repository_statistics(self):
        """Statistics should be calculated correctly."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with known stats
            (Path(tmpdir) / "file1.py").write_text("line1\nline2")
            (Path(tmpdir) / "file2.py").write_text("line1\nline2\nline3")

            result = indexer.index_repository(tmpdir)
            stats = result["statistics"]

            assert stats["total_files"] == 2
            assert stats["total_lines"] == 5  # 2 + 3
            assert ".py" in stats["by_extension"]
            assert stats["by_extension"][".py"] == 2

    def test_index_repository_invalid_path(self):
        """Invalid repository path should raise ValidationError."""
        indexer = CodeRepositoryIndexer()

        with pytest.raises(ValidationError):
            indexer.index_repository("/nonexistent/path")

    def test_index_repository_git_info(self):
        """Repository info should include git details."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = indexer.index_repository(tmpdir)

            assert "repository" in result
            assert "path" in result["repository"]
            assert "name" in result["repository"]
            assert "git_url" in result["repository"]
            assert "default_branch" in result["repository"]


class TestCodeRepositoryIndexerEncodingDetection:
    """Tests for file encoding detection."""

    def test_detect_encoding_utf8(self):
        """UTF-8 files should be detected as UTF-8."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# -*- coding: utf-8 -*-\nprint('hello')")
            f.flush()
            file_path = f.name

        try:
            encoding = indexer._detect_encoding(Path(file_path))
            assert encoding == "utf-8"
        finally:
            os.unlink(file_path)

    def test_detect_encoding_fallback(self):
        """Unknown encoding should return 'unknown'."""
        indexer = CodeRepositoryIndexer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            f.flush()
            file_path = f.name

        try:
            encoding = indexer._detect_encoding(Path(file_path))
            assert encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252", "unknown"]
        finally:
            os.unlink(file_path)


class TestCodeRepositoryIndexerIntegration:
    """Integration tests for repository indexing."""

    def test_full_workflow_basic_repo(self):
        """Complete indexing workflow on a basic repository."""
        indexer = CodeRepositoryIndexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create realistic structure
            src_dir = repo_path / "src"
            src_dir.mkdir()
            (src_dir / "__init__.py").write_text("")
            (src_dir / "main.py").write_text("def main():\n    pass")

            tests_dir = repo_path / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_main.py").write_text("def test_main():\n    pass")

            # Create .gitignore
            (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/")

            # Index repository
            result = indexer.index_repository(str(repo_path))

            # Verify structure
            assert result["repository"]["path"] == str(repo_path)
            assert len(result["files"]) == 3
            assert result["statistics"]["total_files"] == 3
            assert result["statistics"]["by_extension"][".py"] == 3

            # Verify files are sorted
            paths = [f["relative_path"] for f in result["files"]]
            assert paths == sorted(paths)

    def test_full_workflow_with_filtering(self):
        """Repository indexing with custom filtering."""
        config = {
            "code_extensions": {".py"},
            "max_file_size": 500,
        }
        indexer = CodeRepositoryIndexer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create files
            (repo_path / "small.py").write_text("x = 1")
            (repo_path / "large.py").write_text("x = " + "1" * 600)  # Too large
            (repo_path / "script.js").write_text("console.log('hi');")  # Wrong extension

            # Index repository
            result = indexer.index_repository(str(repo_path))

            # Should only find small.py
            assert len(result["files"]) == 1
            assert result["files"][0]["relative_path"] == "small.py"
