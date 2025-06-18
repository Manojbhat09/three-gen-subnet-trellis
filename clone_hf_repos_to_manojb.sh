#!/bin/bash

# Script to clone all Hugging Face repositories used in the project to Manojb organization
# Make sure you have huggingface-cli installed and are logged in: huggingface-cli login

set -e

# Organization name
ORG_NAME="Manojb"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check disk space
check_disk_space() {
    local required_gb=20  # Minimum 20GB free space
    local available_gb=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    
    print_status "Available disk space: ${available_gb}GB"
    
    if [ $available_gb -lt $required_gb ]; then
        print_error "Insufficient disk space! Need at least ${required_gb}GB free, have ${available_gb}GB"
        print_error "Please free up disk space before continuing."
        return 1
    fi
    return 0
}

# Function to cleanup all temporary directories
cleanup_all_temp() {
    print_status "Cleaning up any existing temporary directories..."
    rm -rf temp_* 2>/dev/null || true
    print_success "Cleanup completed"
}

# Function to show current disk usage
show_disk_usage() {
    local available_gb=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Current available disk space: ${available_gb}GB"
}

# Function to check if repository already exists
repo_exists() {
    local repo_name="$1"
    if huggingface-cli list "$repo_name" &>/dev/null; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to clone a repository
clone_repo() {
    local source_repo="$1"
    local repo_name=$(basename "$source_repo")
    local org_repo="${ORG_NAME}/${repo_name}"
    
    print_status "Processing: $source_repo -> $org_repo"
    show_disk_usage
    
    # Check if repository already exists and is complete
    if repo_exists "$org_repo"; then
        print_warning "Repository $org_repo already exists - skipping download and upload"
        print_success "✅ Skipped (already exists): $source_repo -> $org_repo"
        show_disk_usage
        echo "---"
        return 0
    fi
    
    # Create temporary directory with unique timestamp
    local temp_dir="temp_${repo_name}_$(date +%s)"
    
    # Ensure clean start - remove any existing temp directory
    rm -rf "$temp_dir" 2>/dev/null || true
    mkdir -p "$temp_dir"
    
    # Trap to ensure cleanup on any exit (including ctrl+c)
    trap "print_warning 'Cleaning up temporary directory...'; rm -rf '$temp_dir' 2>/dev/null || true" EXIT
    
    # Download the original repository
    print_status "Downloading $source_repo..."
    if huggingface-cli download "$source_repo" --local-dir "$temp_dir" --local-dir-use-symlinks False; then
        print_success "Downloaded $source_repo"
        
        # Show size of downloaded repository
        local repo_size=$(du -sh "$temp_dir" 2>/dev/null | cut -f1 || echo "Unknown")
        print_status "Repository size: $repo_size"
    else
        print_error "Failed to download $source_repo"
        rm -rf "$temp_dir" 2>/dev/null || true
        trap - EXIT  # Remove trap
        return 1
    fi
    
    # Create new repository in organization
    print_status "Creating repository $org_repo..."
    if huggingface-cli repo create "$org_repo" --type model --exist-ok; then
        print_success "Created repository $org_repo"
    else
        print_error "Failed to create repository $org_repo"
        rm -rf "$temp_dir" 2>/dev/null || true
        trap - EXIT  # Remove trap
        return 1
    fi
    
    # Upload to new repository (handle "no changes" case)
    print_status "Uploading to $org_repo..."
    local upload_result=0
    huggingface-cli upload "$org_repo" "$temp_dir" --commit-message "Cloned from $source_repo" || upload_result=$?
    
    if [ $upload_result -eq 0 ]; then
        print_success "Uploaded to $org_repo"
    else
        # Check if it's just a "no changes" scenario
        if [ $upload_result -eq 1 ]; then
            print_warning "No changes detected (repository may already be up to date)"
            print_success "Repository $org_repo is current"
        else
            print_error "Failed to upload to $org_repo (exit code: $upload_result)"
            rm -rf "$temp_dir" 2>/dev/null || true
            trap - EXIT  # Remove trap
            return 1
        fi
    fi
    
    # Immediate cleanup after successful upload
    print_status "Cleaning up temporary files..."
    rm -rf "$temp_dir" 2>/dev/null || true
    trap - EXIT  # Remove trap
    print_success "✅ Completed and cleaned up: $source_repo -> $org_repo"
    
    # Show disk space after cleanup
    show_disk_usage
    echo "---"
    return 0
}

# Array of all Hugging Face repositories to clone
declare -a repos=(
    # 3D Generation Models
    "microsoft/TRELLIS-text-xlarge"
    "microsoft/TRELLIS-text-large"
    "microsoft/TRELLIS-text-base"
    "microsoft/TRELLIS-image-large"
    "jetx/Hunyuan3D-2"
    "tencent/Hunyuan3D-2"
    "Stable-X/trellis-normal-v0-1"
    "Stable-X/stable-normal-v0-1"
    "cavargas10/TRELLIS"
    "hongfz16/3DTopia"
    "frozenburning/3DTopia-XL"
    "stabilityai/TripoSR"
    "stepfun-ai/Step1X-3D"
    "VAST-AI-Research/DetailGen3D"
    
    # Vision & Language Models
    "openai/clip-vit-large-patch14"
    "openai/clip-vit-base-patch32"
    "google/t5-v1_1-large"
    "google/t5-v1_1-xl"
    "Intel/dpt-large"
    "facebook/dinov2-base"
    "facebook/dinov2-with-registers-base"
    "facebook/dinov2-small-imagenet1k-1-layer"
    
    # Stable Diffusion & Image Generation
    "stabilityai/stable-diffusion-2-1-base"
    "stabilityai/stable-diffusion-2-base"
    "runwayml/stable-diffusion-v1-5"
    "gokaygokay/flux-game"
    "madebyollin/taesdxl"
    
    # ControlNet Models
    "lllyasviel/control_v11p_sd15_normalbae"
    "lllyasviel/control_v11f1p_sd15_depth"
    "lllyasviel/control_v11e_sd15_ip2p"
    "lllyasviel/control_v11p_sd15_inpaint"
    "lllyasviel/control_v11e_sd15_depth_aware_inpaint"
    "lllyasviel/control_v11p_sd15_openpose"
    "lllyasviel/control_v11f1e_sd15_tile"
    "lllyasviel/sd-controlnet-canny"
    
    # Specialized Models
    "ashawkey/stable-zero123-diffusers"
    "ashawkey/zero123-xl-diffusers"
    "Peng-Wang/ImageDream"
    "MVDream/MVDream"
    "Kijai/Hunyuan3D-2_safetensors"
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    
    # Validation & Utility Models
    "404-Gen/validation"
)

# Cleanup function for script exit
cleanup_on_exit() {
    print_warning "Script interrupted or finished. Cleaning up any remaining temporary directories..."
    cleanup_all_temp
}

# Set trap for script cleanup
trap cleanup_on_exit EXIT INT TERM

# Main execution
print_status "Starting repository cloning process..."
print_status "Target organization: $ORG_NAME"
print_status "Total repositories to clone: ${#repos[@]}"
echo "===================="

# Initial cleanup
cleanup_all_temp

# Check disk space
if ! check_disk_space; then
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    print_error "huggingface-cli is not installed. Please install it first:"
    echo "pip install huggingface_hub[cli]"
    exit 1
fi

# Check if user is logged in
if ! huggingface-cli whoami &> /dev/null; then
    print_error "You are not logged in to Hugging Face. Please login first:"
    echo "huggingface-cli login"
    exit 1
fi

print_warning "⚠️  IMPORTANT: Each repository will be downloaded, uploaded, and immediately deleted."
print_warning "⚠️  This process minimizes disk usage but requires stable internet connection."
print_warning "⚠️  If interrupted, you can safely re-run the script."
print_warning "⚠️  Existing repositories will be skipped automatically."
echo ""

# Counter for success/failure tracking
success_count=0
failure_count=0
skipped_count=0
declare -a failed_repos=()

# Clone each repository
for repo in "${repos[@]}"; do
    repo_index=$((success_count + failure_count + skipped_count + 1))
    print_status "🚀 Starting repository $repo_index/${#repos[@]}: $repo"
    
    # Disable exit on error for individual repository processing
    set +e
    clone_repo "$repo"
    result=$?
    set -e
    
    if [ $result -eq 0 ]; then
        if repo_exists "${ORG_NAME}/$(basename "$repo")"; then
            if [[ $(grep -c "skipping download" <<< "$(clone_repo "$repo" 2>&1)" || echo "0") -gt 0 ]]; then
                ((skipped_count++))
                print_success "✅ Repository $repo_index/${#repos[@]} skipped (already exists)!"
            else
                ((success_count++))
                print_success "✅ Repository $repo_index/${#repos[@]} completed successfully!"
            fi
        else
            ((success_count++))
            print_success "✅ Repository $repo_index/${#repos[@]} completed successfully!"
        fi
    else
        ((failure_count++))
        failed_repos+=("$repo")
        print_error "❌ Repository failed. Continuing with next..."
    fi
    
    # Force cleanup between repositories
    cleanup_all_temp
    
    # Short pause to prevent rate limiting
    sleep 2
done

# Final cleanup
cleanup_all_temp

# Summary
echo "===================="
print_status "🎉 Cloning process completed!"
print_success "Successfully processed: $success_count repositories"
if [ $skipped_count -gt 0 ]; then
    print_warning "Skipped (already exist): $skipped_count repositories"
fi

if [ $failure_count -gt 0 ]; then
    print_error "Failed to clone: $failure_count repositories"
    echo "Failed repositories:"
    for failed_repo in "${failed_repos[@]}"; do
        echo "  - $failed_repo"
    done
    echo ""
    print_warning "💡 You can re-run this script to retry failed repositories."
else
    print_success "🎊 All repositories processed successfully!"
fi

echo "===================="
print_status "All repositories are now available in the $ORG_NAME organization on Hugging Face!"
show_disk_usage 