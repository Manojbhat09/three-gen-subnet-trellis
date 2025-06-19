#!/bin/bash

# Unified TRELLIS Mining Runner
# Purpose: Single entrypoint to run either one-shot or continuous TRELLIS mining

set -e

# --- Environment Setup ---
# Check if we're in the correct conda environment
if [[ "$CONDA_DEFAULT_ENV" != "trellis_new" ]]; then
    echo "⚠️  Wrong environment detected. Current: $CONDA_DEFAULT_ENV"
    echo "   Activating trellis_new environment..."
    
    # Initialize conda for bash if needed
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found in PATH"
        exit 1
    fi
    
    # Source conda
    eval "$(conda shell.bash hook)"
    
    # Try to activate the environment
    if conda activate trellis_new 2>/dev/null; then
        echo "✅ Activated environment: trellis_new"
    else
        print_error "Failed to activate trellis_new environment"
        print_error "Please ensure the environment exists: conda env list | grep trellis_new"
        exit 1
    fi
fi

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS_SERVER_PORT=8096
VALIDATION_SERVER_PORT=10006
OUTPUT_DIR="./trellis_mining_outputs" # Unified output directory
DB_FILE="trellis_mining_tasks.db"     # Unified database file

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Helper Functions ---
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Service Checks ---
check_service() {
    local url=$1
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

check_trellis_server() {
    print_status "Checking TRELLIS server (http://localhost:$TRELLIS_SERVER_PORT)..."
    # First check if we can connect at all
    if ! curl -s -o /dev/null "http://localhost:${TRELLIS_SERVER_PORT}/status/" 2>/dev/null; then
        print_error "TRELLIS server is not running (connection refused)."
        return 1
    fi
    
    # Then check if the server is ready
    local status=$(curl -s "http://localhost:${TRELLIS_SERVER_PORT}/status/" 2>/dev/null)
    if echo "$status" | grep -q '"ready":true'; then
        print_success "TRELLIS server is running and ready."
        return 0
    else
        print_warning "TRELLIS server is running but models are not loaded."
        return 1
    fi
}

start_trellis_server() {
    print_status "Attempting to start TRELLIS server..."
    if [ ! -f "trellis_submit_server.py" ]; then
        print_error "trellis_submit_server.py not found. Cannot start server."
        exit 1
    fi
    
    python trellis_submit_server.py --port $TRELLIS_SERVER_PORT > trellis_server.log 2>&1 &
    TRELLIS_PID=$!
    
    print_status "Waiting for TRELLIS server to become ready (PID: $TRELLIS_PID)..."
    for i in {1..60}; do
        if check_trellis_server; then
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    print_error "TRELLIS server failed to start in time. Check trellis_server.log for errors."
    exit 1
}

# --- Usage ---
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified TRELLIS Mining Runner. Manages both one-shot and continuous mining modes.

Options:
  --continuous            Run in continuous, always-on mining mode. (Recommended)
  
  --harvest               Enable task harvesting from validators.
  --no-harvest            Disable task harvesting.
  
  --submit                Enable result submission to validators.
  --no-submit             Disable result submission.
  
  --validate              Enable local validation of generations.
  --no-validate           Disable local validation.
  
  --max-tasks N           (One-shot mode only) Max tasks to process. Default: 5.
  --start-server          Auto-start TRELLIS server if not running.
  --help                  Show this help message.

Database:
  Both modes use the same SQLite database ('${DB_FILE}') for task deduplication.

Examples:
  # Run a one-shot job to harvest and submit 3 tasks:
  $0 --harvest --submit --max-tasks 3

  # Run in continuous mining mode (recommended for production):
  $0 --continuous --start-server

  # Run a simple test without harvesting or submitting:
  $0 --no-harvest --no-submit

EOF
}

# --- Main Logic ---
main() {
    # Default arguments
    local mode="oneshot"
    local harvest=false
    local submit=false
    local validate=true
    local max_tasks=5
    local start_server=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --continuous) mode="continuous"; shift ;;
            --harvest) harvest=true; shift ;;
            --no-harvest) harvest=false; shift ;;
            --submit) submit=true; shift ;;
            --no-submit) submit=false; shift ;;
            --validate) validate=true; shift ;;
            --no-validate) validate=false; shift ;;
            --max-tasks) max_tasks="$2"; shift 2 ;;
            --start-server) start_server=true; shift ;;
            --help) show_usage; exit 0 ;;
            *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done

    # Print header
    print_status "--- UNIFIED TRELLIS MINING RUNNER ---"
    print_status "Mode: $mode"
    print_status "Database: $DB_FILE"
    print_status "-------------------------------------"
    
    # Setup cleanup trap
    trap 'kill $TRELLIS_PID 2>/dev/null' EXIT

    # Handle server startup
    if ! check_trellis_server; then
        if [ "$start_server" = true ]; then
            start_trellis_server
        else
            print_error "TRELLIS server not ready. Start it manually or use the --start-server flag."
            exit 1
        fi
    fi

    # Build script arguments for the correct orchestrator
    if [ "$mode" == "continuous" ]; then
        print_status "Starting CONTINUOUS orchestrator..."
        local script_args=()
        [ "$harvest" = false ] && script_args+=(--no-harvest)
        [ "$submit" = false ] && script_args+=(--no-submit)
        [ "$validate" = false ] && script_args+=(--no-validate)
        
        python3 continuous_trellis_orchestrator_optim.py "${script_args[@]}"
    else
        print_status "Starting ONE-SHOT orchestrator..."
        
        declare -a operations_array
        [ "$harvest" = true ] && operations_array+=("harvest")
        [ "$submit" = true ] && operations_array+=("submit")
        [ "$validate" = true ] && operations_array+=("validate")

        function join_by {
          local d=${1-} f=${2-}
          if shift 2; then
            printf %s "$f" "${@/#/$d}"
          fi
        }
        OPERATIONS=$(join_by , "${operations_array[@]}")
        
        if [ -z "$OPERATIONS" ]; then
            print_warning "No operations specified for one-shot mode. Nothing to do."
        else
            python3 orchestrator_trellis.py --operations "$OPERATIONS" --num-tasks "$max_tasks"
        fi
    fi

    print_success "--- Mining Run Finished ---"
}

main "$@" 