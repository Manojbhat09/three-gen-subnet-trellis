# LoRA Numbers Quick Reference

Quick reference for using number-based LoRA selection with the CLIP alignment script.

## Number to LoRA Mapping

| Number | LoRA Endpoint | Description |
|--------|---------------|-------------|
| 0 | none | Use default endpoint (isometric_3d) |
| 1 | isometric_3d | Isometric 3D style |
| 2 | live_3d | Live-action 3D style |
| 3 | game_assets | Game asset style |
| 4 | patched_realism | Patched realism style |
| 5 | tf2_style | Team Fortress 2 style |
| 6 | baolei | Baolei artistic style |
| 7 | cartoon_3d | Cartoon 3D style |
| 8 | cinema | Cinematic style |
| 9 | sd15_game_icon | SD1.5 game icon style |

## Usage Examples

### Single Generation
```bash
# Use live_3d LoRA
python clip_alignment_with_generation.py -1 2 "a blue vase"

# Use game_assets LoRA
python clip_alignment_with_generation.py -1 3 "a blue vase"

# Use no LoRA (default endpoint)
python clip_alignment_with_generation.py -1 0 "a blue vase"
```

### Comparative Analysis with Different LoRAs
```bash
# Prefix analysis: isometric_3d for original, live_3d for prefixed
python clip_alignment_with_generation.py --prefix "professional 3D render, " -1 1 -2 2 "a blue vase"

# Suffix analysis: game_assets for original, cartoon_3d for suffixed
python clip_alignment_with_generation.py --suffix ", cartoon style" -1 3 -2 7 "a blue vase"

# Optimization analysis: cinema for original, patched_realism for optimized
python clip_alignment_with_generation.py --optimized "a blue ceramic vase" -1 8 -2 4 "a blue vase"
```

### Same LoRA for Both Generations
```bash
# Both generations use live_3d
python clip_alignment_with_generation.py --prefix "professional 3D render, " -1 2 -2 2 "a blue vase"

# Both generations use default endpoint
python clip_alignment_with_generation.py --prefix "professional 3D render, " -1 0 -2 0 "a blue vase"
```

## Quick Commands

### For Technical/Architectural
```bash
python clip_alignment_with_generation.py -1 1 "technical diagram"  # isometric_3d
python clip_alignment_with_generation.py -1 4 "architectural model"  # patched_realism
```

### For Game Development
```bash
python clip_alignment_with_generation.py -1 3 "game weapon"  # game_assets
python clip_alignment_with_generation.py -1 5 "tf2 character"  # tf2_style
python clip_alignment_with_generation.py -1 7 "cartoon character"  # cartoon_3d
```

### For Realistic Objects
```bash
python clip_alignment_with_generation.py -1 2 "photorealistic object"  # live_3d
python clip_alignment_with_generation.py -1 8 "cinematic scene"  # cinema
```

### For Artistic/Stylized
```bash
python clip_alignment_with_generation.py -1 6 "artistic object"  # baolei
python clip_alignment_with_generation.py -1 9 "game icon"  # sd15_game_icon
```

## Tips

- **-1**: Controls LoRA for first generation (original prompt)
- **-2**: Controls LoRA for second generation (prefixed/suffixed/optimized prompt)
- **0**: Always means "use default endpoint" (no LoRA)
- **Same numbers**: Use same LoRA for both generations
- **Different numbers**: Use different LoRAs for comparison
- **Interactive mode**: If no LoRA specified, script will ask interactively 