#!/bin/bash

set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

expected_failures=(
  tutorials/qasm_thru_service.py
)

is_expected_to_fail() {
  for expected in "${expected_failures[@]}"; do
    if [[ "$1" == "$expected" ]]; then
      return 0
    fi
  done
  return 1
}

failed=()

for file in tutorials/*.py; do
  echo "🔹 Running $file"

  if is_expected_to_fail "$file"; then
    echo "⚠️ Expecting failure for $file"
    if poetry run python "$file"; then
      echo "❌ $file was expected to fail but passed"
      failed+=("$file (unexpected success)")
    else
      echo "✅ $file failed as expected"
    fi
  else
    if ! poetry run python "$file"; then
      echo "❌ $file failed unexpectedly"
      failed+=("$file (unexpected failure)")
    else
      echo "✅ $file passed"
    fi
  fi
done

echo ""
if [ ${#failed[@]} -ne 0 ]; then
  echo "❌ Some scripts failed:"
  for f in "${failed[@]}"; do
    echo "   - $f"
  done
  exit 1
else
  echo "✅ All tutorials scripts behaved as expected."
fi
