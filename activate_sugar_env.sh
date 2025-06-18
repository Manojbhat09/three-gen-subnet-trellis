#!/bin/bash
echo "🍬 Activating SuGaR Pipeline Environment"
eval "$(conda shell.bash hook)"
conda activate sugar-pipeline
echo "✅ Environment activated: sugar-pipeline"
echo "Ready to run SuGaR pipeline!"
