#!/bin/bash
# Run SuGaR Generation Server in the correct environment

echo "🍬 Starting SuGaR Generation Server"
echo "==================================="

# Check if sugar-pipeline environment exists
if ! conda env list | grep -q "sugar-pipeline"; then
    echo "❌ sugar-pipeline environment not found!"
    echo "Please run: ./create_sugar_environment.sh"
    exit 1
fi

echo "Activating sugar-pipeline environment..."
eval "$(conda shell.bash hook)"
conda activate sugar-pipeline

echo "Environment activated: $(conda info --envs | grep '*')"

# Check if server file exists
if [ ! -f "flux_hunyuan_bpt_sugar_generation_server.py" ]; then
    echo "❌ Server file not found!"
    exit 1
fi

echo "Starting server on port 8095..."
python flux_hunyuan_bpt_sugar_generation_server.py --host 0.0.0.0 --port 8095 