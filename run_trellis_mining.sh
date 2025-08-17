#!/bin/bash

# Unified TRELLIS Mining Runner
# Purpose: Single entrypoint to run either one-shot, continuous, or simulation TRELLIS mining

set -e

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
    # if check_service "http://localhost:$TRELLIS_SERVER_PORT/health/"; then
    local status=$(curl -s "http://localhost:${TRELLIS_SERVER_PORT}/status/")
    if echo "$status" | grep -q '"ready":true'; then
        print_success "TRELLIS server is running and ready."
        return 0
    else
        print_warning "TRELLIS server is running but models are not loaded."
        return 1
    fi
    # else
    #     print_error "TRELLIS server is not running."
    #     return 1
    # fi
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

Unified TRELLIS Mining Runner. Manages one-shot, continuous, and simulation mining modes.

Modes:
  --continuous            Run in continuous, always-on mining mode. (Recommended for production)
  --simulate              Run in simulation mode using prompts from a file.
  
Options:
  --harvest               Enable task harvesting from validators.
  --no-harvest            Disable task harvesting.
  
  --submit                Enable result submission to validators.
  --no-submit             Disable result submission.
  
  --validate              Enable local validation of generations.
  --no-validate           Disable local validation.
  
  --max-tasks N           (One-shot mode only) Max tasks to process. Default: 5.
  --start-server          Auto-start TRELLIS server if not running.
  
  --promptfile FILE       (Simulation mode only) Path to Python file with EPISODIC_TEST_PROMPTS list.
  --no-optimize           Disable prompt optimization.
  --aggressive-optimize   Enable aggressive optimization mode.
  --quiet-optimize        Reduce optimization logging detail.
  --no-reproducibility    Disable reproducibility optimization.
  --reproducibility-similarity FLOAT  Minimum similarity threshold for reproducibility (default: 0.3).
  --variable-seeds        Use prompt-hash based seeds (default: fixed seed 42).
  --seed INT              Fixed seed to use when not using variable seeds (default: 42).
  
  --help                  Show this help message.

Database:
  All modes use SQLite databases for task deduplication and tracking.

Examples:
  # Run a one-shot job to harvest and submit 3 tasks:
  $0 --harvest --submit --max-tasks 3

  # Run in continuous mining mode (recommended for production):
  $0 --continuous --start-server

  # Run simulation with prompts from file:
  $0 --simulate --promptfile episodic_test_prompts.py --start-server

  # Run simulation with custom optimization settings:
  $0 --simulate --promptfile episodic_test_prompts.py --aggressive-optimize --variable-seeds

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
    local promptfile=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --continuous) mode="continuous"; shift ;;
            --simulate) mode="simulation"; shift ;;
            --harvest) harvest=true; shift ;;
            --no-harvest) harvest=false; shift ;;
            --submit) submit=true; shift ;;
            --no-submit) submit=false; shift ;;
            --validate) validate=true; shift ;;
            --no-validate) validate=false; shift ;;
            --max-tasks) max_tasks="$2"; shift 2 ;;
            --start-server) start_server=true; shift ;;
            --promptfile) promptfile="$2"; shift 2 ;;
            --help) show_usage; exit 0 ;;
            *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done

    # Print header
    print_status "--- UNIFIED TRELLIS MINING RUNNER ---"
    print_status "Mode: $mode"
    print_status "Database: $DB_FILE"
    print_status "-------------------------------------"
    
    # Validate simulation mode requirements
    if [ "$mode" == "simulation" ]; then
        if [ -z "$promptfile" ]; then
            print_error "Simulation mode requires --promptfile argument."
            show_usage
            exit 1
        fi
        
        if [ ! -f "$promptfile" ]; then
            print_error "Prompt file not found: $promptfile"
            exit 1
        fi
        
        print_status "Simulation mode: Using prompts from $promptfile"
    fi
    
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
        
        # python3 continuous_trellis_orchestrator.py "${script_args[@]}"
        # python3 continuous_trellis_orchestrator_lora_mod.py --max-concurrent-tasks 2 --max-concurrent-pulls 2 --no-lora-routing --blacklist "${script_args[@]}"
        python3 continuous_trellis_orchestrator_lora.py  "${script_args[@]}"
        
    elif [ "$mode" == "simulation" ]; then
        print_status "Starting SIMULATION orchestrator..."
        local script_args=("--promptfile" "$promptfile")
        [ "$validate" = false ] && script_args+=(--no-validate)
        
        # Pass through any optimization arguments that were provided
        # Note: These would need to be passed as additional arguments to the script
        # For now, we'll use the basic arguments and users can run the simulator directly
        # for advanced optimization options
        
        python3 continuous_trellis_orchestrator_simulator.py "${script_args[@]}"
        
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