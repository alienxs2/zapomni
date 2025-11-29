"""
Zapomni CLI entry point.

Usage:
    zapomni install-hooks [--repo-path PATH]
    zapomni --help
    zapomni --version
"""

import argparse
import sys
from pathlib import Path

from zapomni_cli.install_hooks import install_hooks_command


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="zapomni", description="Zapomni MCP Server CLI")

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # install-hooks command
    install_parser = subparsers.add_parser(
        "install-hooks", help="Install Git hooks for automatic re-indexing"
    )
    install_parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path.cwd(),
        help="Path to Git repository (default: current directory)",
    )

    args = parser.parse_args()

    if args.command == "install-hooks":
        success = install_hooks_command(args.repo_path)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
