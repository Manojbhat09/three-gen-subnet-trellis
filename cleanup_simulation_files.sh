#!/bin/bash

# Safe cleanup script for TRELLIS simulation files
# This script helps clean up simulation-related files with user confirmation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to get file size in human readable format
get_file_size() {
    if [ -f "$1" ]; then
        du -h "$1" | cut -f1
    else
        echo "N/A"
    fi
}

# Function to get directory size in human readable format
get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" | cut -f1
    else
        echo "N/A"
    fi
}

# Function to show file/directory info
show_info() {
    local path="$1"
    local type="$2"
    
    if [ -e "$path" ]; then
        if [ "$type" = "file" ]; then
            local size=$(get_file_size "$path")
            local lines=$(wc -l < "$path" 2>/dev/null || echo "0")
            echo "    📄 $path (Size: $size, Lines: $lines)"
        else
            local size=$(get_dir_size "$path")
            local files=$(find "$path" -type f 2>/dev/null | wc -l)
            echo "    📁 $path (Size: $size, Files: $files)"
        fi
    else
        echo "    ❌ $path (Not found)"
    fi
}

# Function to safely remove file/directory
safe_remove() {
    local path="$1"
    local type="$2"
    
    # Safety check: Never remove Python files
    if [[ "$path" == *.py ]]; then
        print_error "SAFETY: Refusing to remove Python file: $path"
        return 1
    fi
    
    if [ -e "$path" ]; then
        if [ "$type" = "file" ]; then
            rm -f "$path"
            print_success "Removed file: $path"
        else
            rm -rf "$path"
            print_success "Removed directory: $path"
        fi
    else
        print_warning "Path does not exist: $path"
    fi
}

# Function to safely rename file/directory with timestamp suffix
safe_rename() {
    local path="$1"
    local type="$2"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    if [ -e "$path" ]; then
        local new_path="${path}.backup_${timestamp}"
        mv "$path" "$new_path"
        print_success "Renamed $type: $path -> $new_path"
    else
        print_warning "Path does not exist: $path"
    fi
}

# Main cleanup function
main() {
    echo "🧹 TRELLIS Simulation Files Cleanup Script"
    echo "=========================================="
    echo
    
    # Define simulation-related files and directories (generated/cached files only)
    declare -A simulation_files=(
        ["Simulation Log File"]="continuous_trellis_simulator.log"
        ["Simulation Database File"]="continuous_trellis_simulator_tasks.db"
        ["Simulation Output Directory"]="continuous_trellis_simulation_outputs"
    )
    
    # Also check for normal mode files for comparison
    declare -A normal_files=(
        ["Normal Log File"]="continuous_trellis.log"
        ["Normal Database File"]="continuous_trellis_tasks.db"
        ["Normal Output Directory"]="continuous_trellis_outputs"
    )
    
    print_info "Scanning for simulation files..."
    echo
    
    # Check simulation files
    local found_simulation=false
    echo "🎯 Simulation Mode Files:"
    for desc in "${!simulation_files[@]}"; do
        local path="${simulation_files[$desc]}"
        if [ -e "$path" ]; then
            found_simulation=true
            if [ -d "$path" ]; then
                show_info "$path" "directory"
            else
                show_info "$path" "file"
            fi
        fi
    done
    
    if [ "$found_simulation" = false ]; then
        print_warning "No simulation files found!"
    fi
    
    echo
    echo "📊 Normal Mode Files (for reference):"
    for desc in "${!normal_files[@]}"; do
        local path="${normal_files[$desc]}"
        if [ -e "$path" ]; then
            if [ -d "$path" ]; then
                show_info "$path" "directory"
            else
                show_info "$path" "file"
            fi
        fi
    done
    
    echo
    echo "🔍 Additional generated/cached files:"
    
    # Look for generated PLY files in simulation output directory
    if [ -d "continuous_trellis_simulation_outputs" ]; then
        local ply_files=$(find continuous_trellis_simulation_outputs -name "*.ply.spz" -type f 2>/dev/null | head -10)
        if [ -n "$ply_files" ]; then
            echo "    Generated PLY files:"
            echo "$ply_files" | while read -r file; do
                show_info "$file" "file"
            done
        else
            echo "    No generated PLY files found"
        fi
        
        # Look for temp validation files
        local temp_files=$(find continuous_trellis_simulation_outputs -name "temp_validation_*" -type f 2>/dev/null | head -10)
        if [ -n "$temp_files" ]; then
            echo "    Temporary validation files:"
            echo "$temp_files" | while read -r file; do
                show_info "$file" "file"
            done
        else
            echo "    No temporary validation files found"
        fi
        
        # Look for stats files
        local stats_files=$(find continuous_trellis_simulation_outputs -name "continuous_stats_*" -type f 2>/dev/null | head -10)
        if [ -n "$stats_files" ]; then
            echo "    Statistics files:"
            echo "$stats_files" | while read -r file; do
                show_info "$file" "file"
            done
        else
            echo "    No statistics files found"
        fi
    else
        echo "    No simulation output directory found"
    fi
    
    echo
    echo "=========================================="
    
    if [ "$found_simulation" = false ]; then
        print_info "No simulation files to clean up. Exiting."
        exit 0
    fi
    
    # Ask user what to do
    echo
    print_warning "What would you like to do with the simulation files?"
    echo "1) Remove all simulation files (rm -rf)"
    echo "2) Rename all simulation files with timestamp suffix"
    echo "3) Remove specific files (interactive selection)"
    echo "4) Rename specific files (interactive selection)"
    echo "5) Exit without changes"
    echo
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            print_warning "This will PERMANENTLY DELETE all simulation files!"
            read -p "Are you sure? Type 'yes' to confirm: " confirm
            if [ "$confirm" = "yes" ]; then
                echo
                print_info "Removing all simulation files..."
                for desc in "${!simulation_files[@]}"; do
                    local path="${simulation_files[$desc]}"
                    if [ -e "$path" ]; then
                        if [ -d "$path" ]; then
                            safe_remove "$path" "directory"
                        else
                            safe_remove "$path" "file"
                        fi
                    fi
                done
                print_success "All simulation files removed!"
            else
                print_info "Operation cancelled."
            fi
            ;;
        2)
            print_info "Renaming all simulation files with timestamp suffix..."
            for desc in "${!simulation_files[@]}"; do
                local path="${simulation_files[$desc]}"
                if [ -e "$path" ]; then
                    if [ -d "$path" ]; then
                        safe_rename "$path" "directory"
                    else
                        safe_rename "$path" "file"
                    fi
                fi
            done
            print_success "All simulation files renamed!"
            ;;
        3)
            print_info "Interactive removal mode..."
            for desc in "${!simulation_files[@]}"; do
                local path="${simulation_files[$desc]}"
                if [ -e "$path" ]; then
                    echo
                    if [ -d "$path" ]; then
                        show_info "$path" "directory"
                    else
                        show_info "$path" "file"
                    fi
                    read -p "Remove $desc ($path)? [y/N]: " remove_choice
                    if [[ "$remove_choice" =~ ^[Yy]$ ]]; then
                        if [ -d "$path" ]; then
                            safe_remove "$path" "directory"
                        else
                            safe_remove "$path" "file"
                        fi
                    fi
                fi
            done
            ;;
        4)
            print_info "Interactive rename mode..."
            for desc in "${!simulation_files[@]}"; do
                local path="${simulation_files[$desc]}"
                if [ -e "$path" ]; then
                    echo
                    if [ -d "$path" ]; then
                        show_info "$path" "directory"
                    else
                        show_info "$path" "file"
                    fi
                    read -p "Rename $desc ($path)? [y/N]: " rename_choice
                    if [[ "$rename_choice" =~ ^[Yy]$ ]]; then
                        if [ -d "$path" ]; then
                            safe_rename "$path" "directory"
                        else
                            safe_rename "$path" "file"
                        fi
                    fi
                fi
            done
            ;;
        5)
            print_info "Exiting without changes."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
    
    echo
    print_success "Cleanup completed!"
    
    # Show final status
    echo
    print_info "Final status:"
    for desc in "${!simulation_files[@]}"; do
        local path="${simulation_files[$desc]}"
        if [ -e "$path" ]; then
            print_warning "Still exists: $path"
        else
            print_success "Cleaned: $path"
        fi
    done
}

# Run main function
main "$@"
