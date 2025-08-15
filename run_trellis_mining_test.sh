#!/bin/bash

# Unified TRELLIS Mining Runner (Test)
# - Uses testnet orchestrator and test-specific outputs/logs

set -e

# Fix CUDA deterministic behavior for production validation
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS_SERVER_PORT=8096
OUTPUT_DIR="./trellis_mining_outputs_test"
DB_FILE="continuous_trellis_tasks_test.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_trellis_server() {
  local status=$(curl -s "http://localhost:${TRELLIS_SERVER_PORT}/status/")
  if echo "$status" | grep -q '"ready":true'; then
    return 0
  else
    return 1
  fi
}

start_trellis_server() {
  print_status "Attempting to start TRELLIS server..."
  if [ ! -f "trellis_submit_server.py" ]; then
    print_error "trellis_submit_server.py not found. Cannot start server."
    exit 1
  fi
  python trellis_submit_server.py --port $TRELLIS_SERVER_PORT > trellis_server_test.log 2>&1 &
  TRELLIS_PID=$!
  print_status "Waiting for TRELLIS server to become ready (PID: $TRELLIS_PID)..."
  for i in {1..60}; do
    if check_trellis_server; then
      return 0
    fi
    sleep 2
    echo -n "."
  done
  print_error "TRELLIS server failed to start in time. Check trellis_server_test.log for errors."
  exit 1
}

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Testnet TRELLIS Mining Runner.

Options:
  --continuous            Run continuous test orchestrator (default).
  --no-harvest            Disable task harvesting.
  --no-submit             Disable result submission.
  --no-validate           Disable local validation.
  --dual-validation       After submit, run local production-accurate validation and print comparison table.
  --start-server          Auto-start TRELLIS server if not running.
  --no-optimize           Disable prompt optimization.
  --no-reproducibility    Disable reproducibility optimization using episodic memory.
  --reproducibility-similarity <f>
                          Set minimum similarity threshold for reproducibility optimization (default 0.3).
  --quiet-optimize        Reduce optimization logging verbosity.
  --help                  Show this help message.
EOF
}

main() {
  local mode="continuous"
  local harvest=false
  local submit=false
  local validate=false
  local start_server=false
  local dual_validation=false
  local optimize=true
  local repro=true
  local quiet_optimize=false
  local repro_similarity=""

  while [[ $# -gt 0 ]]; do
    case $1 in
      --continuous) mode="continuous"; shift ;;
      --harvest) harvest=true; shift ;;
      --submit) submit=true; shift ;;
      --validate) validate=true; shift ;;
      --no-harvest) harvest=false; shift ;;
      --no-submit) submit=false; shift ;;
      --no-validate) validate=false; shift ;;
      --dual-validation) dual_validation=true; shift ;;
      --start-server) start_server=true; shift ;;
      --no-optimize) optimize=false; shift ;;
      --no-reproducibility) repro=false; shift ;;
      --reproducibility-similarity) repro_similarity="$2"; shift 2 ;;
      --quiet-optimize) quiet_optimize=true; shift ;;
      --help) show_usage; exit 0 ;;
      *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
  done

  print_status "--- TESTNET TRELLIS MINING RUNNER ---"
  print_status "Mode: $mode"
  print_status "DB: $DB_FILE"

  trap 'kill $TRELLIS_PID 2>/dev/null || true' EXIT

  if ! check_trellis_server; then
    if [ "$start_server" = true ]; then
      start_trellis_server
    else
      print_warning "TRELLIS server not ready; attempting anyway. Use --start-server to auto-start."
    fi
  fi

  if [ "$mode" = "continuous" ]; then
    print_status "Starting CONTINUOUS test orchestrator (LoRA)..."
    declare -a script_args
    [ "$harvest" = false ] && script_args+=(--no-harvest)
    [ "$submit" = false ] && script_args+=(--no-submit)
    [ "$validate" = false ] && script_args+=(--no-validate)
    [ "$dual_validation" = true ] && script_args+=(--dual-validation)
    [ "$optimize" = false ] && script_args+=(--no-optimize)
    [ "$repro" = false ] && script_args+=(--no-reproducibility)
    [ "$quiet_optimize" = true ] && script_args+=(--quiet-optimize)
    [ -n "$repro_similarity" ] && script_args+=(--reproducibility-similarity "$repro_similarity")

    python3 continuous_trellis_orchestrator_lora_test.py --validators "${VALIDATORS:-79}" "${script_args[@]}"
  fi

  print_success "--- Test Mining Finished ---"
}

main "$@" 