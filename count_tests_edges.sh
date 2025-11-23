#!/bin/bash
for file in /home/dev/zapomni/.spec-workflow/specs/level3/*.md; do
  if [[ ! "$file" =~ SUMMARY ]]; then
    name=$(basename "$file")
    edge_cases=$(grep -c "^### Edge Case\|^#### [0-9]" "$file" 2>/dev/null || echo 0)
    tests=$(grep -c "^### .*test_\|^#### [0-9]\. test_" "$file" 2>/dev/null || echo 0)
    echo "$name: E=$edge_cases T=$tests"
  fi
done | sort
