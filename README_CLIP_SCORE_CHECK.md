# Simple CLIP Score Check

A simple tool to compare CLIP scores between text prompts using the exact same model as the subnet accurate validator.

## Features

- **Production-Accurate**: Uses the same `convnext_large_d` model with `laion2b_s26b_b102k_augreg` weights as the production validation system
- **Text Similarity**: Compare similarity between two text prompts
- **Quality Scoring**: Evaluate the quality/clarity of individual prompts
- **Batch Comparison**: Compare multiple prompts against each other
- **Reference Comparison**: Compare prompts against a reference prompt

## Installation

The tool requires the same dependencies as the validation system:

```bash
# Install open_clip and other dependencies
pip install open_clip_torch torch torchvision numpy
```

## Usage

### 1. Simple Two-Prompt Comparison

Compare similarity between two prompts:

```bash
python clip_score_check.py "a blue vase" "a red vase"
```

**Output:**
```
🔍 Comparing prompts:
   Prompt 1: 'a blue vase'
   Prompt 2: 'a red vase'

📊 Similarity Score: 0.8032
🎯 Interpretation: High Similarity

📊 Quality Scores:
   Prompt 1: 1.0000
   Prompt 2: 1.0000
```

### 2. Quality Score Check

Evaluate the quality of a single prompt:

```bash
python clip_score_check.py --quality "a blue ceramic vase with red trim, professional 3D render"
```

**Output:**
```
🔍 Computing quality score for: 'a blue ceramic vase with red trim, professional 3D render'

📊 Quality Score: 1.0000
🎯 Interpretation: Excellent
```

### 3. Multiple Prompt Comparison

Compare multiple prompts against each other:

```bash
python clip_score_check.py --compare "vase" "blue vase" "red vase" "ceramic vase"
```

### 4. Reference-Based Comparison

Compare multiple prompts against a reference prompt:

```bash
python clip_score_check.py --compare "vase" "blue vase" "red vase" --reference "a blue ceramic vase with red trim"
```

**Output:**
```
📊 Computing similarities to reference: 'a blue ceramic vase with red trim'
   'vase' -> reference: 0.4866
   'blue vase' -> reference: 0.7207
   'red vase' -> reference: 0.6602

📋 SUMMARY:
   Best quality: vase
   Most similar to reference: 'blue vase' (score: 0.7207)
   Most similar pair: vase <-> ceramic vase (score: 0.8750)
```

## Score Interpretation

### Similarity Scores
- **0.9+**: Very High Similarity
- **0.7-0.9**: High Similarity  
- **0.5-0.7**: Moderate Similarity
- **0.3-0.5**: Low Similarity
- **<0.3**: Very Low Similarity

### Quality Scores
- **0.8+**: Excellent
- **0.6-0.8**: Good
- **0.4-0.6**: Fair
- **<0.4**: Poor

## API Usage

You can also use the CLIP score checker programmatically:

```python
from simple_clip_score_check import SimpleCLIPScoreChecker

# Initialize checker
checker = SimpleCLIPScoreChecker(verbose=True)

# Compare two prompts
similarity = checker.compute_text_similarity("a blue vase", "a red vase")
print(f"Similarity: {similarity:.4f}")

# Check quality score
quality = checker.compute_prompt_quality_score("a blue ceramic vase")
print(f"Quality: {quality:.4f}")

# Compare multiple prompts
results = checker.compare_multiple_prompts(
    prompts=["vase", "blue vase", "red vase"],
    reference_prompt="a blue ceramic vase"
)

# Cleanup
checker.unload_model()
```

## Model Details

- **Model**: `convnext_large_d`
- **Weights**: `laion2b_s26b_b102k_augreg`
- **Device**: Auto-detects CUDA if available, falls back to CPU
- **Normalization**: Uses production-standard normalization transforms

## Examples

### Example 1: Prompt Optimization Analysis
```bash
# Compare original vs optimized prompts
python clip_score_check.py \
  "a blue vase" \
  "a blue ceramic vase with red trim, professional 3D render, highly detailed, photorealistic"
```

### Example 2: Quality Assessment
```bash
# Check quality of different prompt styles
python clip_score_check.py --quality "vase"
python clip_score_check.py --quality "a blue ceramic vase"
python clip_score_check.py --quality "professional 3D render, Create 3D game asset, isometric view"
```

### Example 3: Batch Analysis
```bash
# Analyze multiple prompt variations
python clip_score_check.py --compare \
  "vase" \
  "blue vase" \
  "ceramic vase" \
  "blue ceramic vase" \
  "professional 3D render of a blue ceramic vase" \
  --reference "a blue ceramic vase with red trim"
```

## Notes

- The tool automatically loads and unloads the CLIP model to manage GPU memory
- All scores are normalized to the [0, 1] range
- The model uses the same preprocessing and normalization as the production validation system
- Quality scores are based on the magnitude of CLIP text features (higher = better CLIP understanding) 