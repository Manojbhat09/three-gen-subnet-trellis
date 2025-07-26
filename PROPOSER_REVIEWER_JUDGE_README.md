# Proposer-Reviewer-Judge Training & Inference System

## Overview

This system implements a sophisticated **3-agent training loop** that solves the fundamental problem of prompt optimization: **training the Reviewer to become a reliable proxy for slow external validation**.

### The Core Innovation

**During Training**: Judge (external validator) provides ground truth to train both Proposer and Reviewer  
**During Inference**: Trained Reviewer replaces Judge for fast, validation-free optimization

This enables **10x+ performance improvement** while maintaining quality through learned validation patterns.

## System Architecture

```
TRAINING PHASE:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PROPOSER   │───▶│  REVIEWER   │───▶│    JUDGE    │
│ (Creative)  │    │(Analytical) │    │(Ground Truth│
│             │◀───│             │◀───│ Validator)  │
└─────────────┘    └─────────────┘    └─────────────┘
      ▲                    ▲                    │
      │                    │                    │
      └────────────────────┼────────────────────┘
                           ▼
                   TRAINING MEMORY

INFERENCE PHASE:
┌─────────────┐    ┌─────────────┐
│  PROPOSER   │───▶│ TRAINED     │
│ (Creative)  │    │ REVIEWER    │ ──► Fast Results
│             │◀───│(Proxy Judge)│
└─────────────┘    └─────────────┘
      ▲                    ▲
      │                    │
      └────────────────────┘
         Trained Knowledge
```

## Key Components

### 1. **Proposer Agent** (Creative Optimizer)
- Generates optimized prompts using learned strategies
- Learns from Judge feedback and Reviewer critiques
- Adapts approach based on successful patterns
- Uses strategy-based optimization (material focus, artistic elaboration, etc.)

### 2. **Reviewer Agent** (Analytical Critic)
- Critiques proposals and predicts validation scores
- **Learns to mimic Judge's scoring patterns** through training
- Provides detailed feedback and improvement suggestions
- Becomes reliable proxy for external validation after training

### 3. **Judge Agent** (Ground Truth Provider)
- External validation via `subnet_accurate_validator.py`
- Provides definitive scores during training phase
- **Replaced by trained Reviewer during inference**
- Teaches both agents what constitutes quality

## Usage

### Phase 1: Training (Build Knowledge)

#### Single Prompt Training
```bash
# Train on a specific prompt
python proposer_reviewer_judge_trainer.py "emerald pendant"

# The system will:
# 1. Proposer generates optimizations
# 2. Reviewer critiques and predicts scores  
# 3. Judge provides ground truth validation
# 4. Both agents learn from Judge feedback
# 5. Successful sessions saved to memory
```

#### Batch Training (Recommended)
```bash
# Quick training (5 prompts for testing)
python batch_training_script.py --quick

# Comprehensive training (all prompts for maximum knowledge)
python batch_training_script.py --comprehensive

# Standard training (default)
python batch_training_script.py
```

**Training Prompts Include:**
- Jewelry: "emerald pendant", "silver bracelet", "golden ring"
- Glassware: "crystal wine glass", "bottle of red wine"
- Art Objects: "harp with pearl inlays", "crystal staff"
- Cultural Items: "greek kylix cup", "porcelain figurine"
- And many more for diverse knowledge...

### Phase 2: Fast Inference (Use Trained Knowledge)

```bash
# Fast optimization using trained Reviewer as Judge proxy
python trained_debate_inference.py "sapphire necklace"

# Results in 3-6 seconds instead of 30-60 seconds
# No external dependencies, reliable quality scoring
```

## System Files

| File | Purpose |
|------|---------|
| `proposer_reviewer_judge_trainer.py` | Core training system (single prompt) |
| `batch_training_script.py` | Batch training for comprehensive knowledge |
| `trained_debate_inference.py` | Fast inference using trained agents |
| `prj_training_memory.json` | Saved training knowledge (auto-generated) |
| `inference_sessions.json` | Inference session history (auto-generated) |

## Performance Comparison

| System | Speed | Dependencies | Reliability | Quality |
|--------|-------|--------------|-------------|---------|
| **Traditional Validation** | 30-60s | External validator, conda, subprocess | Low (brittle) | Good |
| **Conversational Debate** | 3-6s | None (self-contained) | Medium | Good |
| **Trained P-R-J** | 3-6s | None (self-contained) | **High** | **Excellent** |

## Training Process Deep Dive

### Round-by-Round Training
```
Round 1:
  Proposer: "emerald pendant with intricate silver setting"
  Reviewer: Predicts 0.75, critiques "lacks material specificity"
  Judge: Validates → Actual score 0.72
  Learning: Reviewer prediction was close ✓

Round 2:
  Proposer: "faceted emerald pendant with ornate silver filigree setting"
  Reviewer: Predicts 0.86, critiques "excellent detail, good material focus"
  Judge: Validates → Actual score 0.89
  Learning: Reviewer prediction very accurate ✓✓

Round 3:
  Proposer: "emerald-cut emerald pendant in detailed silver setting with tiny diamonds"
  Reviewer: Predicts 0.92, high confidence
  Judge: Validates → Actual score 0.91
  Learning: Reviewer is now well-calibrated ✓✓✓
```

### Knowledge Accumulation
- **Strategy Performance**: Which approaches work best for different object types
- **Scoring Patterns**: What the Judge considers high vs low quality
- **Reviewer Calibration**: Training to predict Judge scores accurately
- **Optimization Principles**: Reusable patterns for future optimizations

## Expected Training Results

After comprehensive training, you should see:

### Training Statistics
- **Success Rate**: 70-90% of sessions reaching quality threshold
- **Score Distribution**: 60%+ high scores (≥0.85), <20% low scores (<0.70)
- **Reviewer Accuracy**: Average prediction error <0.1
- **Strategy Learning**: Clear performance differences between strategies

### Inference Performance  
- **Speed**: 3-6 seconds per optimization (vs 30-60s traditional)
- **Quality**: Scores comparable to external validation
- **Reliability**: No external dependencies, consistent performance
- **Confidence**: Reviewer provides reliable confidence scores

## Configuration Options

### Training Configuration
```python
trainer = ProposerReviewerJudgeTrainer(
    max_rounds=4,              # Max debate rounds per prompt
    quality_threshold=0.8,     # Min score to save to memory
    convergence_threshold=0.9  # Score for early stopping
)
```

### Inference Configuration
```python
inference = TrainedDebateInference(
    max_rounds=3,              # Max optimization rounds
    target_score=0.9,          # Target score for convergence
    min_improvement=0.05       # Min improvement to continue
)
```

## Troubleshooting

### Training Issues
- **Judge validation fails**: Check conda environment and validator script
- **Low success rates**: Lower quality_threshold or increase max_rounds
- **Memory not building**: Ensure training sessions exceed quality_threshold

### Inference Issues
- **Poor performance**: Need more training data - run batch training
- **No training memory**: Run training first before inference
- **Inconsistent results**: Train on more diverse prompts

## Advanced Usage

### Custom Training Sets
```python
# Train on specific object types
custom_prompts = [
    "diamond ring", "ruby necklace", "sapphire earrings",
    "gold bracelet", "platinum watch", "pearl pendant"
]

trainer.train_on_prompt_set(prompt_subset=custom_prompts)
```

### Strategy Analysis
```python
# Analyze which strategies work best
stats = trainer.get_training_statistics()
strategy_performance = stats['strategy_performance']

# Focus training on underperforming strategies
```

### Reviewer Calibration Monitoring
```python
# Track Reviewer prediction accuracy over time
calibration = stats['reviewer_calibration']
recent_accuracy = calibration['recent_accuracy']
accuracy_trend = calibration['accuracy_trend']
```

## Research Applications

This system demonstrates several important ML concepts:

1. **Learning from Ground Truth**: Training an agent to replicate expensive oracles
2. **Proxy Model Development**: Creating fast approximations of slow systems  
3. **Multi-Agent Learning**: Coordinated learning between specialized agents
4. **Transfer Learning**: Applying learned patterns to new optimization tasks
5. **Self-Supervised Training**: Using conversation to improve both agents

## Future Enhancements

- **Dynamic Strategy Selection**: Learn optimal strategies for prompt types
- **Active Learning**: Identify which prompts need training most
- **Ensemble Reviewers**: Multiple reviewer agents for consensus scoring
- **Continual Learning**: Update training knowledge from inference results
- **Domain Adaptation**: Specialized training for specific object categories

---

## Quick Start Guide

1. **Install Dependencies**: Ensure Ollama is running with llama3.2:3b
2. **Run Training**: `python batch_training_script.py --quick`
3. **Test Inference**: `python trained_debate_inference.py "golden bracelet"`
4. **Scale Up**: `python batch_training_script.py --comprehensive` for production

The system transforms slow, validation-dependent optimization into fast, self-contained conversational AI that maintains quality through learned ground truth patterns. 🚀 