#!/bin/bash

# TRELLIS Orchestrator Runner
# This script runs the TRELLIS orchestrator with proper environment setup

set -e

echo "Starting TRELLIS Orchestrator..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if orchestrator file exists
if [ ! -f "orchestrator_trellis.py" ]; then
    echo "Error: orchestrator_trellis.py not found in current directory"
    exit 1
fi

# Default values
NUM_TASKS=5
declare -a operations_array

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --harvest)
            operations_array+=("harvest")
            shift
            ;;
        --validate)
            operations_array+=("validate")
            shift
            ;;
        --no-validate)
            # This is handled by not including 'validate'
            shift
            ;;
        --submit)
            operations_array+=("submit")
            shift
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --harvest          Enable harvesting tasks."
            echo "  --validate         Enable validating results."
            echo "  --no-validate      Disable validating results (by omitting --validate)."
            echo "  --submit           Enable submitting results."
            echo "  --num-tasks <N>    Number of tasks to process (default: 5)."
            echo "  --help             Show this help message."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Join the array into a comma-separated string
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}
OPERATIONS=$(join_by , "${operations_array[@]}")

# If no operations specified, use defaults
if [ -z "$OPERATIONS" ]; then
    echo "No operations specified. Defaulting to harvest,validate,submit."
    OPERATIONS="harvest,validate,submit"
fi


# Run the orchestrator
echo "Running TRELLIS orchestrator with operations: $OPERATIONS, num_tasks: $NUM_TASKS"
python3 orchestrator_trellis.py --operations "$OPERATIONS" --num_tasks "$NUM_TASKS"

echo "TRELLIS orchestrator finished." 