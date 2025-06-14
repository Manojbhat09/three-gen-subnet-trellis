# SuGaR Pipeline Environment Setup

## 🚀 Quick Start

The SuGaR pipeline requires a dedicated conda environment to avoid dependency conflicts. Follow these steps:

### Step 1: Create Environment
```bash
chmod +x create_sugar_environment.sh
./create_sugar_environment.sh
```

This will:
- Create a new conda environment called `sugar-pipeline`
- Install PyTorch with CUDA support
- Install all necessary dependencies
- Test the installation

### Step 2: Activate Environment
```bash
# Method 1: Use the activation script
source activate_sugar_env.sh

# Method 2: Direct conda activation
conda activate sugar-pipeline
```

### Step 3: Start the Server
```bash
python flux_hunyuan_bpt_sugar_generation_server.py --port 8095
```

### Step 4: Test the Pipeline
In a new terminal:
```bash
conda activate sugar-pipeline
python test_sugar_pipeline.py
```

## 🔧 Environment Details

**Environment Name**: `sugar-pipeline`
**Python Version**: 3.9
**Key Dependencies**:
- PyTorch with CUDA 12.1
- FastAPI + Uvicorn
- Trimesh, Open3D, PLYFile
- Diffusers, Transformers
- PyTorch3D (if available)

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8095/health/
```

### Status Check
```bash
curl http://localhost:8095/status/
```

### Generate Test Model
```bash
curl -X POST "http://localhost:8095/generate/" \
  -F "prompt=a red apple" \
  -F "seed=12345" \
  -o test_apple.ply
```

## 🛠️ Troubleshooting

### Environment Issues
If you get import errors:
```bash
conda activate sugar-pipeline
pip install --upgrade pip
pip install missing_package
```

### CUDA Issues
Check CUDA availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Memory Issues
For RTX 4090 with 24GB VRAM, reduce point sampling:
```python
GENERATION_CONFIG['sugar_num_points'] = 25000  # Reduced from 50000
```

### SuGaR Import Issues
The server will automatically fall back to basic mesh-to-GS conversion if SuGaR imports fail. This is expected and the pipeline will still work.

## 📊 Expected Performance

- **Environment Setup**: 5-10 minutes
- **First Generation**: 45-60 seconds (including model loading)
- **Subsequent Generations**: 30-45 seconds
- **Validation Scores**: 0.5783+ (real scores vs 0.0000 with mesh PLY)

## 🔄 Switching Between Environments

### Current environments:
- `hunyuan3d`: Original generation pipeline
- `three-gen-validation`: Validation server
- `sugar-pipeline`: New SuGaR-enhanced pipeline

### Switch to SuGaR environment:
```bash
conda activate sugar-pipeline
```

### Switch back to original:
```bash
conda activate hunyuan3d  # or three-gen-validation
```

## 🎯 Success Indicators

✅ Environment created without errors
✅ All dependencies installed
✅ CUDA available and working
✅ Server starts on port 8095
✅ Health check returns 200
✅ Status shows models_loaded properly
✅ Test generation creates valid PLY files
✅ PLY files have Gaussian Splatting attributes
✅ Validation server accepts generated PLY files

## 🆘 Getting Help

If you encounter issues:

1. **Check environment activation**:
   ```bash
   conda info --envs
   which python
   ```

2. **Check dependencies**:
   ```bash
   pip list | grep torch
   pip list | grep fastapi
   ```

3. **Check CUDA**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

4. **Clear and restart**:
   ```bash
   conda deactivate
   conda env remove -n sugar-pipeline -y
   ./create_sugar_environment.sh
   ```

---

## 🎉 Ready to Go!

Once setup is complete, you'll have a clean, isolated environment for the SuGaR-enhanced generation pipeline that produces validation-compatible Gaussian Splatting PLY files! 