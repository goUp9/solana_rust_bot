#!/bin/bash
# Quick setup script for training environment

set -e

echo "🎮 Setting up Game RL Training Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ CUDA not found - training will be slow!"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check API key
if [ -z "$CHUTES_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: CHUTES_API_KEY not set!"
    echo "   Please run: export CHUTES_API_KEY='your-key'"
else
    echo "✓ CHUTES_API_KEY is set"
fi

# Check Wandb key (optional)
if [ -z "$WANDB_API_KEY" ]; then
    echo "ℹ️  Optional: Set WANDB_API_KEY for experiment tracking"
else
    echo "✓ WANDB_API_KEY is set"
fi

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p checkpoints
mkdir -p logs
echo "✓ Directories created"

# Test environment connection
echo ""
echo "🔌 Testing environment connection..."
python3 -c "
import sys
sys.path.insert(0, '..')
try:
    import pyspiel
    print('✓ pyspiel import successful (local OpenSpiel execution ready).')
except Exception as e:
    print(f'❌ Environment connection failed: {e}')
    sys.exit(1)
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start training, run:"
echo "  python train_ppo_lora.py"
echo ""
