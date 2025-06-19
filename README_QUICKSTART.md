# TRELLIS Mining Quick Start Guide

## 🚀 Quick Start - Choose Your Configuration

### Option 1: CLIP-Optimized (Recommended) ✅

**For stable, consistent mining with minimal failures:**

```bash
# Terminal 1 - Start generation server
conda activate trellis_text
python trellis_submit_server.py

# Terminal 2 - Run miner with CLIP optimization
conda activate trellis_text
bash run_trellis_miner_optim.sh --continuous --harvest --submit
```

**Features:**
- ✅ Automatic prompt optimization using CLIP
- ✅ Detects and fixes problematic prompts
- ✅ <5% failure rate
- ✅ Average scores: 0.7-0.8

---

### Option 2: High-Score Configuration ⚡

**For maximum scores (but with higher risk):**

```bash
# Terminal 1 - Start high-score server
conda activate trellis_text
python trellis_submit_server_highscore.py

# Terminal 2 - Run standard miner
conda activate trellis_text
bash run_trellis_mining.sh --continuous --harvest --submit
```

**Features:**
- 🏆 Can achieve 0.9+ scores
- ⚡ Faster processing
- ⚠️ 15-30% failure rate
- 📊 High variance in results

---

## 📋 File Reference

### CLIP-Optimized Setup:
- `trellis_submit_server.py` - Generation server with dynamic parameters
- `continuous_trellis_orchestrator_optim.py` - Orchestrator with CLIP optimization
- `run_trellis_miner_optim.sh` - Launch script
- `prompt_optimizer_v2.py` - Advanced prompt optimizer
- `clip_prompt_optimizer.py` - CLIP-based optimization

### High-Score Setup:
- `trellis_submit_server_highscore.py` - Optimized generation server
- `continuous_trellis_orchestrator.py` - Standard orchestrator
- `run_trellis_mining.sh` - Standard launch script

---

## 🔍 Monitoring

Check logs for performance:
```bash
# Watch live logs
tail -f continuous_trellis.log

# Check for failures
grep "0.0000" continuous_trellis.log | tail -20

# View statistics
grep "ORCHESTRATOR STATUS" continuous_trellis.log | tail -1 -A 20
```

---

## 💡 Tips

1. **Start with CLIP-Optimized** for consistent results
2. **Monitor failure rates** - if >10%, something's wrong
3. **Check GPU memory** - both configs need ~20GB VRAM
4. **Update validators** - refresh happens every 10 minutes

## Essential Tips

1. **Start Small**: Test with a few validators first
2. **Monitor Logs**: Keep an eye on `continuous_trellis_orchestrator.log`
3. **Check Rewards**: Your average fidelity score determines rewards
4. **Handle Failures**: The system automatically retries failed tasks
5. **Resource Usage**: Each generation uses ~8GB GPU memory

## Troubleshooting Zero Scores

If you're getting many 0.0 fidelity scores, analyze your results:

```bash
# Analyze zero score patterns
python analyze_zero_scores.py --hours 24

# Check specific problematic materials
python analyze_zero_scores.py --db continuous_trellis_tasks.db --hours 48
```

Common causes of zero scores:
- Transparent materials (glass, crystal, gems)
- Multi-object scenes with "with" or "holding"
- Liquids and flowing materials
- Abstract or artistic descriptions
- Complex jewelry or intricate details

---

For detailed configuration options, see [README_TRELLIS_CONFIGURATIONS.md](README_TRELLIS_CONFIGURATIONS.md) 