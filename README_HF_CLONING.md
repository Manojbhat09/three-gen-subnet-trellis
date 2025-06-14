# Hugging Face Repository Cloning Scripts

This repository contains scripts to clone all Hugging Face repositories used in the three-gen-subnet-trellis project into the **Manojb** organization.

## ⚠️ SPACE-EFFICIENT DESIGN ⚠️

**These scripts are designed to minimize disk space usage:**
- ✅ Downloads one repository at a time
- ✅ Immediately deletes each repository after upload
- ✅ Monitors disk space continuously
- ✅ Requires minimum 20GB free space
- ✅ Automatic cleanup on interruption or failure
- ✅ Shows repository sizes and remaining space

## Total Repositories: 42

### Repository Categories:

1. **3D Generation Models (14):**
   - microsoft/TRELLIS-text-xlarge
   - microsoft/TRELLIS-text-large
   - microsoft/TRELLIS-text-base
   - microsoft/TRELLIS-image-large
   - jetx/Hunyuan3D-2
   - tencent/Hunyuan3D-2
   - Stable-X/trellis-normal-v0-1
   - Stable-X/stable-normal-v0-1
   - cavargas10/TRELLIS
   - hongfz16/3DTopia
   - frozenburning/3DTopia-XL
   - stabilityai/TripoSR
   - stepfun-ai/Step1X-3D
   - VAST-AI-Research/DetailGen3D

2. **Vision & Language Models (9):**
   - openai/clip-vit-large-patch14
   - openai/clip-vit-base-patch32
   - google/t5-v1_1-large
   - google/t5-v1_1-xl
   - Intel/dpt-large
   - facebook/dinov2-base
   - facebook/dinov2-with-registers-base
   - facebook/dinov2-small-imagenet1k-1-layer

3. **Stable Diffusion & Image Generation (5):**
   - stabilityai/stable-diffusion-2-1-base
   - stabilityai/stable-diffusion-2-base
   - runwayml/stable-diffusion-v1-5
   - gokaygokay/flux-game
   - madebyollin/taesdxl

4. **ControlNet Models (8):**
   - lllyasviel/control_v11p_sd15_normalbae
   - lllyasviel/control_v11f1p_sd15_depth
   - lllyasviel/control_v11e_sd15_ip2p
   - lllyasviel/control_v11p_sd15_inpaint
   - lllyasviel/control_v11e_sd15_depth_aware_inpaint
   - lllyasviel/control_v11p_sd15_openpose
   - lllyasviel/control_v11f1e_sd15_tile
   - lllyasviel/sd-controlnet-canny

5. **Specialized Models (5):**
   - ashawkey/stable-zero123-diffusers
   - ashawkey/zero123-xl-diffusers
   - Peng-Wang/ImageDream
   - MVDream/MVDream
   - Kijai/Hunyuan3D-2_safetensors
   - laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K

6. **Validation & Utility Models (1):**
   - 404-Gen/validation

## Prerequisites

### Disk Space Requirements:
- **Minimum**: 20GB free space (scripts will check and warn you)
- **Recommended**: 50GB+ free space for comfort margin
- **Note**: Only one repository is stored temporarily at a time

### For Both Scripts:
1. **Install Hugging Face CLI:**
   ```bash
   pip install huggingface_hub[cli]
   ```

2. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   Enter your Hugging Face token when prompted.

3. **Ensure you have permissions to create repositories in the Manojb organization.**

### For Python Script (Additional):
```bash
pip install huggingface_hub
```

## Usage

### Option 1: Bash Script (Recommended for Linux/macOS)

```bash
# Make script executable
chmod +x clone_hf_repos_to_manojb.sh

# Run the script
./clone_hf_repos_to_manojb.sh
```

### Option 2: Python Script (Cross-platform)

```bash
# Run the Python script
python3 clone_hf_repos_to_manojb.py
```

## What the Scripts Do

1. **🧹 Initial Cleanup**: Removes any leftover temporary directories
2. **💾 Disk Space Check**: Ensures at least 20GB free space
3. **📥 Download**: Downloads one repository to a temporary directory
4. **📊 Size Report**: Shows the downloaded repository size
5. **🏗️ Create**: Creates a new repository in the Manojb organization
6. **📤 Upload**: Uploads all files to the new repository
7. **🗑️ Immediate Cleanup**: Deletes the temporary directory immediately
8. **🔄 Repeat**: Moves to the next repository

## New Space-Efficient Features

- ✅ **Aggressive Cleanup**: Removes temporary files after each repository
- ✅ **Disk Monitoring**: Shows available space before/after each operation
- ✅ **Size Reporting**: Shows how much space each repository uses
- ✅ **Interrupt Handling**: Cleans up properly if you stop the script (Ctrl+C)
- ✅ **Error Recovery**: Cleans up temporary files even if errors occur
- ✅ **Progress Tracking**: Shows which repository is being processed
- ✅ **Rate Limiting**: Includes pauses to prevent API rate limits

## Output Example

```
[INFO] Starting repository cloning process...
[INFO] Target organization: Manojb
[INFO] Total repositories to clone: 42
====================
[INFO] Cleaning up any existing temporary directories...
[SUCCESS] Cleanup completed
[INFO] Available disk space: 45.2GB
[SUCCESS] Authenticated as: YourUsername
⚠️  IMPORTANT: Each repository will be downloaded, uploaded, and immediately deleted.
⚠️  This process minimizes disk usage but requires stable internet connection.
⚠️  If interrupted, you can safely re-run the script.

[INFO] 🚀 Starting repository 1/42: microsoft/TRELLIS-text-xlarge
[INFO] Processing: microsoft/TRELLIS-text-xlarge -> Manojb/TRELLIS-text-xlarge
[INFO] Current available disk space: 45.2GB
[INFO] Downloading microsoft/TRELLIS-text-xlarge...
[SUCCESS] Downloaded microsoft/TRELLIS-text-xlarge
[INFO] Repository size: 4.2GB
[INFO] Creating repository Manojb/TRELLIS-text-xlarge...
[SUCCESS] Created repository Manojb/TRELLIS-text-xlarge
[INFO] Uploading to Manojb/TRELLIS-text-xlarge...
[SUCCESS] Uploaded to Manojb/TRELLIS-text-xlarge
[INFO] Cleaning up temporary files...
[SUCCESS] ✅ Completed and cleaned up: microsoft/TRELLIS-text-xlarge -> Manojb/TRELLIS-text-xlarge
[INFO] Current available disk space: 45.1GB
[SUCCESS] ✅ Repository 1/42 completed successfully!
---
```

## Important Notes

✅ **Space Efficient**: Maximum temporary storage is ~10GB (largest single repository)  
✅ **Safe Interruption**: You can stop and restart the script safely  
✅ **No Permanent Storage**: No files remain on disk after completion  
✅ **Automatic Cleanup**: All temporary files are removed automatically  

⚠️ **Stable Internet Required**: Process downloads and uploads each repository completely  
⚠️ **Time Required**: Process may take 3-6 hours depending on connection speed  
⚠️ **Organization Permissions**: Ensure you can create repos in Manojb organization  

## Troubleshooting

### Common Issues:

1. **Authentication Error**: Run `huggingface-cli login` again
2. **Permission Denied**: Ensure you have access to create repos in Manojb organization
3. **Out of Space**: The script checks space automatically and will warn you
4. **Network Issues**: Re-run the script; it will skip already completed repositories
5. **Interrupted Process**: Just re-run the script; cleanup is automatic

### Space Management:

The scripts automatically:
- Check for 20GB minimum free space before starting
- Show available space before and after each repository
- Clean up immediately after each repository
- Handle interruptions gracefully with cleanup

### Manual Cleanup (if needed):

```bash
# Remove any leftover temporary directories
rm -rf temp_*

# Check available space
df -h .
```

## Results

After successful completion, all 42 repositories will be available at:
`https://huggingface.co/Manojb/[REPOSITORY_NAME]`

For example:
- `https://huggingface.co/Manojb/TRELLIS-text-xlarge`
- `https://huggingface.co/Manojb/Hunyuan3D-2`
- etc.

## Support

If you encounter any issues, check:
1. Your internet connectivity and stability
2. Hugging Face authentication status: `huggingface-cli whoami`
3. Available disk space: `df -h .`
4. Organization permissions on Hugging Face 