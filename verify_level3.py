#!/usr/bin/env python3
"""
Comprehensive verification of all 50 Level 3 function specs.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set

@dataclass
class SpecAnalysis:
    filename: str
    has_signature: bool
    has_parameters: bool
    has_returns: bool
    has_exceptions: bool
    edge_case_count: int
    test_count: int
    has_pseudocode: bool
    line_count: int
    has_parent_reference: bool
    issues: List[str]
    warnings: List[str]

    @property
    def is_complete(self) -> bool:
        return (self.has_signature and self.has_parameters and
                self.has_returns and self.has_exceptions and
                self.edge_case_count >= 6 and self.test_count >= 10)

def analyze_spec(filepath: Path) -> SpecAnalysis:
    """Analyze a single spec file."""
    content = filepath.read_text()
    lines = content.split('\n')

    issues = []
    warnings = []

    # Check required sections
    has_signature = bool(re.search(r'##\s+Signature', content, re.IGNORECASE))
    has_parameters = bool(re.search(r'##\s+Parameters', content, re.IGNORECASE))
    has_returns = bool(re.search(r'##\s+Returns?', content, re.IGNORECASE))
    has_exceptions = bool(re.search(r'##\s+Exceptions?', content, re.IGNORECASE))
    has_parent_reference = bool(re.search(r'Parent Component:', content, re.IGNORECASE))

    # Count edge cases
    edge_case_section = re.search(r'##\s+Edge Cases(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
    if edge_case_section:
        edge_case_content = edge_case_section.group(1)
        edge_case_count = len(re.findall(r'^\s*[-\*]\s+', edge_case_content, re.MULTILINE))
    else:
        edge_case_count = 0

    # Count test scenarios
    test_section = re.search(r'##\s+Test (?:Scenarios|Cases)(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
    if test_section:
        test_content = test_section.group(1)
        test_count = len(re.findall(r'^\s*[-\*]\s+', test_content, re.MULTILINE))
    else:
        test_count = 0

    # Check for pseudocode/algorithm
    has_pseudocode = bool(re.search(r'(Algorithm|Pseudocode|Implementation|Process Flow)', content, re.IGNORECASE))

    # Validation
    if not has_signature:
        issues.append("Missing Signature section")
    if not has_parameters:
        issues.append("Missing Parameters section")
    if not has_returns:
        issues.append("Missing Returns section")
    if not has_exceptions:
        issues.append("Missing Exceptions section")
    if not has_parent_reference:
        warnings.append("Missing Parent Component reference")
    if edge_case_count < 6:
        issues.append(f"Only {edge_case_count} edge cases (need 6+)")
    if test_count < 10:
        issues.append(f"Only {test_count} tests (need 10+)")
    if not has_pseudocode:
        warnings.append("No algorithm/pseudocode section found")

    return SpecAnalysis(
        filename=filepath.name,
        has_signature=has_signature,
        has_parameters=has_parameters,
        has_returns=has_returns,
        has_exceptions=has_exceptions,
        edge_case_count=edge_case_count,
        test_count=test_count,
        has_pseudocode=has_pseudocode,
        line_count=len(lines),
        has_parent_reference=has_parent_reference,
        issues=issues,
        warnings=warnings
    )

def extract_data_models(filepath: Path) -> Set[str]:
    """Extract data model references from a spec."""
    content = filepath.read_text()
    models = set()

    # Common patterns for data models
    patterns = [
        r'\b(Chunk|Memory|SearchResult|Entity|Metadata|Config|Task)\b',
        r'class\s+(\w+)',
        r'interface\s+(\w+)',
        r'type\s+(\w+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        models.update(matches)

    return models

def main():
    spec_dir = Path('/home/dev/zapomni/.spec-workflow/specs/level3')
    spec_files = sorted([f for f in spec_dir.glob('*.md') if 'SUMMARY' not in f.name])

    print("=" * 80)
    print("LEVEL 3 FUNCTION SPECS - COMPREHENSIVE VERIFICATION")
    print("=" * 80)
    print(f"\nAnalyzing {len(spec_files)} function specs...\n")

    analyses = []
    all_models = set()

    for filepath in spec_files:
        analysis = analyze_spec(filepath)
        analyses.append(analysis)
        all_models.update(extract_data_models(filepath))

    # Calculate metrics
    complete_count = sum(1 for a in analyses if a.is_complete)
    total_tests = sum(a.test_count for a in analyses)
    total_edge_cases = sum(a.edge_case_count for a in analyses)
    total_lines = sum(a.line_count for a in analyses)

    avg_tests = total_tests / len(analyses)
    avg_edge_cases = total_edge_cases / len(analyses)
    avg_lines = total_lines / len(analyses)

    completeness_pct = (complete_count / len(analyses)) * 100

    # Collect all issues and warnings
    all_issues = []
    all_warnings = []

    for analysis in analyses:
        if analysis.issues:
            all_issues.append(f"{analysis.filename}: {', '.join(analysis.issues)}")
        if analysis.warnings:
            all_warnings.append(f"{analysis.filename}: {', '.join(analysis.warnings)}")

    # Report
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Functions analyzed: {len(analyses)}")
    print(f"Complete specs: {complete_count}/{len(analyses)} ({completeness_pct:.1f}%)")
    print(f"Total test scenarios: {total_tests}")
    print(f"Total edge cases: {total_edge_cases}")
    print(f"Unique data models referenced: {len(all_models)}")

    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    print(f"Avg tests per function: {avg_tests:.1f}")
    print(f"Avg edge cases per function: {avg_edge_cases:.1f}")
    print(f"Avg spec length: {avg_lines:.0f} lines")

    print("\n" + "=" * 80)
    print(f"CRITICAL ISSUES ({len(all_issues)})")
    print("=" * 80)
    if all_issues:
        for issue in all_issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more")
    else:
        print("  NONE")

    print("\n" + "=" * 80)
    print(f"WARNINGS ({len(all_warnings)})")
    print("=" * 80)
    if all_warnings:
        for warning in all_warnings[:20]:  # Show first 20
            print(f"  - {warning}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")
    else:
        print("  NONE")

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN BY SPEC")
    print("=" * 80)

    for analysis in analyses:
        status = "✓ COMPLETE" if analysis.is_complete else "✗ INCOMPLETE"
        print(f"\n{analysis.filename}: {status}")
        print(f"  Tests: {analysis.test_count}, Edge Cases: {analysis.edge_case_count}, Lines: {analysis.line_count}")
        if analysis.issues:
            print(f"  Issues: {'; '.join(analysis.issues)}")
        if analysis.warnings:
            print(f"  Warnings: {'; '.join(analysis.warnings)}")

    print("\n" + "=" * 80)
    print("DATA MODELS REFERENCED")
    print("=" * 80)
    for model in sorted(all_models):
        print(f"  - {model}")

    # Decision
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    if completeness_pct == 100 and len(all_issues) == 0:
        decision = "APPROVE"
        rationale = "All specs complete with no critical issues"
    elif completeness_pct >= 90 and len(all_issues) <= 5:
        decision = "APPROVE_WITH_WARNINGS"
        rationale = f"{completeness_pct:.1f}% complete with {len(all_issues)} minor issues"
    else:
        decision = "REFINE"
        rationale = f"Only {completeness_pct:.1f}% complete with {len(all_issues)} issues requiring attention"

    print(f"Decision: {decision}")
    print(f"Rationale: {rationale}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if decision == "APPROVE":
        print("✓ Proceed to implementation phase")
        print("✓ All function specs are ready for development")
    elif decision == "APPROVE_WITH_WARNINGS":
        print("✓ Proceed to implementation with minor fixes")
        print("! Address warnings during implementation")
    else:
        print("! Fix critical issues before proceeding")
        print("! Focus on specs with missing required sections")

    print("\n")

if __name__ == '__main__':
    main()
