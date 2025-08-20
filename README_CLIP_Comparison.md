# CLIP Alignment and Validation Score Comparison

This repository contains scripts to compare CLIP alignment scores and validation scores between two 3D model generations using the TrellisGenerator class.

## Scripts Overview

### 1. `example_clip_comparison.py` - Simple Example Script
A basic script that demonstrates the core functionality:
- Generates 3D models from original and cleaned prompts
- Extracts images from both generations
- Computes CLIP alignment scores for all combinations
- Provides a simple analysis of the results

### 2. `test_clip_validation_comparison.py` - Comprehensive Test Script with Reproducibility
A full-featured script that includes:
- **Automatic prompt optimization** using the reproducibility system
- 3D model generation from original and optimized prompts
- CLIP alignment score computation
- PLY file validation using direct imports from subnet_accurate_validator_multigpu.py
- Comprehensive comparison and analysis
- Detailed reporting and JSON output

## Prerequisites

Make sure you have the following dependencies installed:
- `trellis_subnit_server_mix_lora_flash.py` - Contains the TrellisGenerator class
- `clip_alignment_with_generation.py` - Contains the CLIPAlignmentWithGeneration class
- `subnet_accurate_validator_multigpu.py` - For PLY validation (required for comprehensive script)
- `llm_close_prompt_reproducibility_test.py` - Contains the LLMClosePromptReproducibility class
- `continuous_trellis_orchestrator_lora_working.py` - Contains the ContinuousTrellisOrchestrator class

### Additional Requirements
- vLLM server running on localhost:9002 (for reproducibility system)
- Episodic logs available for gold prompt extraction

## Usage Examples

### Basic CLIP Comparison

```bash
# Compare two prompts with default settings
python example_clip_comparison.py "a red car" "a red sports car on road"

# Use a specific seed for reproducible results
python example_clip_comparison.py "a red car" "a red sports car on road" 123
```

### Comprehensive Comparison with Reproducibility and Validation

```bash
# Compare with automatic prompt optimization and full validation
python test_clip_validation_comparison.py "a red car"

# Customize reproducibility and generation parameters
python test_clip_validation_comparison.py "a red car" \
    --log-count 10 \
    --min-similarity 0.4 \
    --seed 42 \
    --port 8099 \
    --ss_steps 21 \
    --slat_steps 24 \
    --slat_guidance 4.0 \
    --ss_guidance 9.5
```

## What the Scripts Do

### 1. Generation Phase
```python
generator = TrellisGenerator()

# Generate from original prompt
result1 = generator.generate_3d_model_image(
    original_prompt, seed, 
    num_inference_steps, guidance_scale,
    ss_sampling_steps, slat_sampling_steps,
    slat_guidance_strength, ss_guidance_strength
)

# Generate from cleaned prompt  
result2 = generator.generate_3d_model_image(
    cleaned_prompt, seed,
    num_inference_steps, guidance_scale,
    ss_sampling_steps, slat_sampling_steps,
    slat_guidance_strength, ss_guidance_strength
)

# Extract results
ply_data1, compressed_data1, image1 = result1
ply_data2, compressed_data2, image2 = result2
```

### 2. CLIP Score Computation
```python
clip_analyzer = CLIPAlignmentAnalyzer()

# Compute all possible combinations
score_original_original = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image1)
score_cleaned_cleaned = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, pil_image2)
score_original_cleaned = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image2)
score_cleaned_original = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, pil_image1)
```

### 3. Validation (Comprehensive Script Only)
```python
# Validate both PLY files
original_validation = run_ply_validation(
    ply_data1, original_prompt, endpoint, port, ...)
cleaned_validation = run_ply_validation(
    ply_data2, cleaned_prompt, endpoint, port, ...)
```

## Output Files

### Basic Script Output
- `clip_comparison_example_{seed}.json` - Summary of CLIP scores and analysis

### Comprehensive Script Output  
- `clip_validation_reproducibility_comparison_{port}.json` - Complete results including reproducibility analysis, CLIP scores, and validation scores
- `subnet_validation_results_{port}.json` - Validation results from the validator script

## Understanding the Results

### CLIP Alignment Scores
- **Direct Matches**: How well each prompt aligns with its own generated image
- **Cross Matches**: How well each prompt aligns with the other's generated image
- **Improvement Analysis**: Whether the optimized prompt produces better alignment

### Reproducibility Analysis
- **Gold Similarity**: How similar the original prompt is to the best matching gold prompt
- **Gold Score**: The validation score of the best matching gold prompt
- **Reproducibility Assessment**: How well the optimized prompt performs compared to the gold standard

## Troubleshooting

### CUDA Deterministic Behavior Error
If you encounter this error:
```
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
```

**Solution**: Set the following environment variable before running the script:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python test_clip_validation_comparison.py "your prompt here"
```

Or run it in one line:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python test_clip_validation_comparison.py "your prompt here"
```

### Validation Scores
- **Original PLY Score**: Quality score for the 3D model from original prompt
- **Cleaned PLY Score**: Quality score for the 3D model from cleaned prompt
- **Improvement Analysis**: Whether prompt cleaning improved 3D model quality

### Overall Assessment
- **EXCELLENT**: Both CLIP and validation scores improved significantly
- **GOOD**: One score improved significantly
- **ACCEPTABLE**: Scores maintained at similar levels
- **POOR**: Scores degraded

## Customization Options

### Generation Parameters
- `--num_inference_steps`: Image generation steps (default: 7)
- `--guidance_scale`: Image guidance scale (default: 3.5)
- `--ss_steps`: Sparse-structure sampler steps (default: 21)
- `--slat_steps`: SLAT sampler steps (default: 24)
- `--slat_guidance`: SLAT guidance strength (default: 4.0)
- `--ss_guidance`: Sparse-structure guidance strength (default: 9.5)

### Validation Parameters
- `--endpoint`: Validation endpoint (default: "generate/")
- `--port`: Port for validation (default: 8099)
- `--seed`: Random seed for reproducible results (default: 42)

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all required modules are in your Python path
2. **CUDA Memory**: 3D generation requires significant GPU memory
3. **Validation Failures**: Check if the validator script is accessible and working
4. **CLIP Model Loading**: Ensure CLIP models can be downloaded/loaded

### Debug Mode
Both scripts include detailed logging to help diagnose issues. Check the console output for specific error messages.

## Performance Notes

- **Generation Time**: 3D model generation typically takes 2-5 minutes per model
- **CLIP Computation**: Fast, typically under 10 seconds
- **Validation**: Depends on the validator script, typically 1-3 minutes per PLY
- **Memory Usage**: High GPU memory usage during 3D generation

## Use Cases

1. **Prompt Optimization**: Test if cleaned/optimized prompts produce better results
2. **Quality Assessment**: Compare CLIP alignment vs. validation scores
3. **Cross-Validation**: Test how well different prompts work with different images
4. **Research**: Analyze the relationship between text-image alignment and 3D model quality

## Contributing

Feel free to modify these scripts for your specific needs. Key areas for customization:
- Different CLIP models or scoring methods
- Alternative validation approaches
- Additional analysis metrics
- Integration with other evaluation frameworks
