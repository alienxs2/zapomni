#!/usr/bin/env python3
"""
Comprehensive verification of all 50 Level 3 function specs - Updated.
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
    parent_component: str
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

    # Check for function signature section (can be "Function Signature" or in docstring)
    has_signature = bool(re.search(r'(##\s+Function Signature|def\s+\w+\()', content, re.IGNORECASE))

    # Parameters can be in Args section of docstring
    has_parameters = bool(re.search(r'(##\s+Parameters|Args:|Arguments:)', content, re.IGNORECASE))

    # Returns can be in docstring or separate section
    has_returns = bool(re.search(r'(##\s+Returns?|Returns:|Return:)', content, re.IGNORECASE))

    # Exceptions/Raises
    has_exceptions = bool(re.search(r'(##\s+Exceptions?|Raises:|Errors:)', content, re.IGNORECASE))

    # Parent component reference
    parent_match = re.search(r'\*\*Component:\*\*\s+(\w+)', content)
    has_parent_reference = bool(parent_match)
    parent_component = parent_match.group(1) if parent_match else "Unknown"

    # Count edge cases
    edge_case_section = re.search(r'##\s+Edge Cases(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
    if edge_case_section:
        edge_case_content = edge_case_section.group(1)
        # Count numbered items, bullet points, and sub-points
        edge_case_count = len(re.findall(r'^\s*[-\*\d]+[\.\)]\s+', edge_case_content, re.MULTILINE))
    else:
        edge_case_count = 0

    # Count test scenarios
    test_section = re.search(r'##\s+Test (?:Scenarios|Cases|Coverage)(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
    if test_section:
        test_content = test_section.group(1)
        # Count numbered items and bullet points
        test_count = len(re.findall(r'^\s*[-\*\d]+[\.\)]\s+', test_content, re.MULTILINE))
    else:
        test_count = 0

    # Check for pseudocode/algorithm
    has_pseudocode = bool(re.search(r'(Algorithm|Pseudocode|Implementation|Process Flow|Implementation Details|How It Works)', content, re.IGNORECASE))

    # Validation
    if not has_signature:
        issues.append("Missing Function Signature")
    if not has_parameters:
        issues.append("Missing Parameters/Args")
    if not has_returns:
        issues.append("Missing Returns")
    if not has_exceptions:
        issues.append("Missing Exceptions/Raises")
    if not has_parent_reference:
        warnings.append("Missing Component reference")
    if edge_case_count < 6:
        issues.append(f"Only {edge_case_count} edge cases (need 6+)")
    if test_count < 10:
        issues.append(f"Only {test_count} test scenarios (need 10+)")
    if not has_pseudocode:
        warnings.append("No algorithm/implementation section")

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
        parent_component=parent_component,
        issues=issues,
        warnings=warnings
    )

def extract_data_models(filepath: Path) -> Set[str]:
    """Extract data model references from a spec."""
    content = filepath.read_text()
    models = set()

    # Look for actual data model names (capitalized, likely types)
    patterns = [
        r'\b(Memory|Chunk|SearchResult|Entity|Metadata|Config|Task|Tool|ValidationError|DatabaseError|TimeoutError|ChunkingError|EmbeddingError)\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        models.update(matches)

    return models

def group_by_component(analyses: List[SpecAnalysis]) -> Dict[str, List[SpecAnalysis]]:
    """Group analyses by parent component."""
    groups = {}
    for analysis in analyses:
        if analysis.parent_component not in groups:
            groups[analysis.parent_component] = []
        groups[analysis.parent_component].append(analysis)
    return groups

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

    avg_tests = total_tests / len(analyses) if analyses else 0
    avg_edge_cases = total_edge_cases / len(analyses) if analyses else 0
    avg_lines = total_lines / len(analyses) if analyses else 0

    completeness_pct = (complete_count / len(analyses)) * 100 if analyses else 0

    # Collect all issues and warnings
    all_issues = []
    all_warnings = []

    for analysis in analyses:
        if analysis.issues:
            all_issues.append(f"{analysis.filename}: {', '.join(analysis.issues)}")
        if analysis.warnings:
            all_warnings.append(f"{analysis.filename}: {', '.join(analysis.warnings)}")

    # Group by component
    by_component = group_by_component(analyses)

    # Report
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Functions analyzed: {len(analyses)}")
    print(f"Complete specs: {complete_count}/{len(analyses)} ({completeness_pct:.1f}%)")
    print(f"Total test scenarios: {total_tests}")
    print(f"Total edge cases: {total_edge_cases}")
    print(f"Unique data models referenced: {len(all_models)}")
    print(f"Components covered: {len(by_component)}")

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
        for issue in all_issues[:30]:  # Show first 30
            print(f"  - {issue}")
        if len(all_issues) > 30:
            print(f"  ... and {len(all_issues) - 30} more")
    else:
        print("  NONE")

    print("\n" + "=" * 80)
    print(f"WARNINGS ({len(all_warnings)})")
    print("=" * 80)
    if all_warnings:
        for warning in all_warnings[:30]:  # Show first 30
            print(f"  - {warning}")
        if len(all_warnings) > 30:
            print(f"  ... and {len(all_warnings) - 30} more")
    else:
        print("  NONE")

    print("\n" + "=" * 80)
    print("BREAKDOWN BY COMPONENT")
    print("=" * 80)
    for component in sorted(by_component.keys()):
        specs = by_component[component]
        complete = sum(1 for s in specs if s.is_complete)
        print(f"\n{component}: {complete}/{len(specs)} complete")
        for spec in specs:
            status = "✓" if spec.is_complete else "✗"
            print(f"  {status} {spec.filename} (T:{spec.test_count}, E:{spec.edge_case_count})")

    print("\n" + "=" * 80)
    print("DATA MODELS REFERENCED")
    print("=" * 80)
    for model in sorted(all_models):
        print(f"  - {model}")

    print("\n" + "=" * 80)
    print("SPECS WITH ISSUES")
    print("=" * 80)
    incomplete_specs = [a for a in analyses if not a.is_complete]
    for spec in incomplete_specs[:20]:
        print(f"\n{spec.filename}:")
        for issue in spec.issues:
            print(f"  ✗ {issue}")

    # Decision
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    if completeness_pct == 100 and len(all_issues) == 0:
        decision = "APPROVE"
        rationale = "All specs complete with no critical issues"
    elif completeness_pct >= 90 and len(all_issues) <= 10:
        decision = "APPROVE_WITH_WARNINGS"
        rationale = f"{completeness_pct:.1f}% complete with {len(all_issues)} minor issues"
    elif completeness_pct >= 70:
        decision = "APPROVE_WITH_WARNINGS"
        rationale = f"{completeness_pct:.1f}% complete, most specs ready for implementation"
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
        print("✓ Proceed to implementation")
        print("! Address incomplete specs during development")
        print(f"! {len(incomplete_specs)} specs need edge cases/tests")
    else:
        print("! Fix critical issues before proceeding")
        print("! Focus on specs with missing required sections")

    print("\n")

if __name__ == '__main__':
    main()
