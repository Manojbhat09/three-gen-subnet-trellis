# Score Calculators - Fast CLIP and Alignment Scoring

## 🎯 Overview

The RL optimizer now includes fast score calculators that can compute individual metrics without running the full validation engine. This is much faster for testing and analysis.

## 🚀 New Command Line Options

### 1. Alignment Score Only (`--alignment-score`)
Calculates only the alignment score using the validation system's CLIP model.

```bash
python test_rl_standalone.py "a red car" --alignment-score
```

**What it does:**
- Generates PLY and image using `generate_both` endpoint
- Uses validation system's ConvNeXt Large D CLIP model
- Renders multiple views of the 3D model
- Computes alignment score with normalization

**Output:**
```
🧪 Alignment Score Calculator
==================================================
Prompt: 'a red car'
Endpoint: generate_both/cinema/
==================================================
🔍 Calculating alignment score for: 'a red car...'
🎨 Generating PLY and image...
⏱️ Generation: 15.23s
🔍 Computing alignment score...
⏱️ Alignment computation: 2.45s
✅ Alignment score: 0.6341

🎯 Final Alignment Score: 0.6341
```

### 2. CLIP Score Only (`--clip-score`)
Calculates only the CLIP score using our custom CLIP scorer.

```bash
python test_rl_standalone.py "a red car" --clip-score
```

**What it does:**
- Generates image using `generate_both` endpoint
- Uses ViT-B-32 CLIP model with OpenAI weights
- Computes direct image-text similarity
- No normalization or filtering

**Output:**
```
🧪 CLIP Score Calculator
==================================================
Prompt: 'a red car'
Endpoint: generate_both/cinema/
==================================================
🔍 Calculating CLIP score for: 'a red car...'
🎨 Generating image...
⏱️ Generation: 15.23s
🖼️ Computing CLIP score...
⏱️ CLIP computation: 0.12s
✅ CLIP score: 0.3397
📊 Image size: 245760 bytes

🎯 Final CLIP Score: 0.3397
```

### 3. Both Scores (`--both-scores`)
Calculates both alignment and CLIP scores for comparison.

```bash
python test_rl_standalone.py "a red car" --both-scores
```

### 4. RL Optimization with Alignment Score (`--rl-alignment`)
Runs RL optimization using alignment score for grading iterations instead of full validation score.

```bash
python test_rl_standalone.py "a red car" --rl-alignment
```

**What it does:**
- Generates PLY and image once using `generate_both` endpoint
- Calculates both alignment score and CLIP score
- Shows the difference between the two scores

**Output:**
```
🧪 Both Scores Calculator
==================================================
Prompt: 'a red car'
Endpoint: generate_both/cinema/
==================================================
🔍 Calculating both alignment and CLIP scores for: 'a red car...'
🎨 Generating PLY and image...
⏱️ Generation: 15.23s
🔍 Computing alignment score...
⏱️ Alignment computation: 2.45s
🖼️ Computing CLIP score...
⏱️ CLIP computation: 0.12s
✅ Alignment score: 0.6341
✅ CLIP score: 0.3397
📊 Image size: 245760 bytes
📊 PLY size: 2048576 bytes

🎯 Final Scores:
   Alignment Score: 0.6341
   CLIP Score: 0.3397
   Difference: +0.2944
```

## ⚡ Performance Benefits

### Speed Comparison
- **Full Validation**: ~25-30 seconds (includes quality metrics, SSIM, LPIPS)
- **Alignment Score Only**: ~18-20 seconds (skips quality metrics)
- **CLIP Score Only**: ~16-18 seconds (only image generation + CLIP)
- **Both Scores**: ~18-20 seconds (reuses generation, both calculations)

### Memory Usage
- **Alignment Score**: Loads validation system CLIP model
- **CLIP Score**: Loads custom CLIP scorer model
- **Both Scores**: Loads both models, cleans up after

## 🔍 Understanding the Score Differences

### Why Alignment Score ≠ CLIP Score

1. **Different CLIP Models:**
   - **Alignment**: ConvNeXt Large D + LAION-2B weights
   - **CLIP**: ViT-B-32 + OpenAI weights

2. **Different Image Sources:**
   - **Alignment**: Multiple rendered views of 3D model
   - **CLIP**: Single generated 2D image

3. **Different Processing:**
   - **Alignment**: Geometric mean, outlier filtering, normalization (÷0.35)
   - **CLIP**: Direct similarity, no filtering

4. **Different Aggregation:**
   - **Alignment**: Multiple camera angles
   - **CLIP**: Single image

## 🛠️ Usage Examples

### Quick Score Check
```bash
# Fast alignment score check
python test_rl_standalone.py "a blue bicycle" --alignment-score

# Fast CLIP score check  
python test_rl_standalone.py "a blue bicycle" --clip-score

# Compare both scores
python test_rl_standalone.py "a blue bicycle" --both-scores
```

### Different Endpoints
```bash
# Use different TRELLIS endpoints
python test_rl_standalone.py "a red car" --alignment-score --endpoint "generate_both/"
python test_rl_standalone.py "a red car" --clip-score --endpoint "generate_both/cinema/"
```

### Batch Testing
```bash
# Test multiple prompts quickly
for prompt in "a red car" "a blue bicycle" "a green tree"; do
    echo "Testing: $prompt"
    python test_rl_standalone.py "$prompt" --both-scores
    echo "---"
done
```

## 🔧 Technical Details

### Dependencies
- **Alignment Score**: Requires validation system models
- **CLIP Score**: Requires custom CLIP scorer
- **Both Scores**: Requires both model sets

### Model Loading
- Models are loaded once and reused
- Automatic cleanup after calculation
- GPU memory management included

### Error Handling
- Graceful fallbacks if generation fails
- Model loading error recovery
- Timeout protection (180s for generation)

## 📊 Use Cases

1. **Prompt Optimization**: Quickly test different prompts
2. **Model Comparison**: Compare alignment vs CLIP scores
3. **Performance Analysis**: Measure generation quality
4. **Debugging**: Isolate scoring issues
5. **Research**: Analyze score correlations

## 🎯 Integration with RL Optimizer

These score calculators can be used to:
- Pre-test prompts before full RL optimization
- Analyze why certain prompts score differently
- Debug scoring issues in the RL system
- Compare different generation endpoints

The scores provide insights into:
- **Alignment Score**: 3D model quality and multi-view consistency
- **CLIP Score**: 2D image generation quality
- **Difference**: Gap between 3D and 2D generation quality
