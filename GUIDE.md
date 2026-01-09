# Step-by-Step Guide: Training Qwen3-4B with PPO+LoRA on Game Environment

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ GPU with at least 16GB VRAM (24GB+ recommended)
- ✅ Python 3.8 or higher
- ✅ CUDA 11.8 or higher
- ✅ OpenSpiel installed locally (`pyspiel` import works)
- ✅ (Optional) `WANDB_API_KEY` for tracking
- ✅ (Optional) `HF_TOKEN` if the model is gated

### What “local OpenSpiel” means (important)
This guide uses **local in-process OpenSpiel execution** (`execution=local_openspiel` in `config/env_config.json`):
- No vLLM server
- No env server (docker/basilica) required for rollouts
- PPO rollouts always use the **current** training weights (correct PPO)

## 🚀 Installation (5 minutes)

### Step 1: Navigate to training directory
```bash
cd /root/workstation/sn120/stage_4/game_rl_training
```

### Step 2: Run setup script
```bash
bash scripts/setup.sh
```

This will:
- Install all Python dependencies
- Check CUDA availability
- Test `pyspiel` import (OpenSpiel)
- Create necessary directories

### Step 3: Set environment variables
```bash
# Optional (for experiment tracking)
export WANDB_API_KEY="your-wandb-key"

# Optional (for private models)
export HF_TOKEN="your-hf-token"
```

## ⚙️ Configuration (5 minutes)

You have two options for configuration:

### Option A: Interactive Wizard (Recommended for beginners)
```bash
python scripts/configure.py
```

This will walk you through all configuration options.

### Option B: Manual Configuration (Advanced users)

Edit `config/train_config.json`:
```json
{
  "model": {
    "model_name": "Qwen/Qwen3-4B",
    "use_4bit": true  // Set false if you have >24GB VRAM
  },
  "ppo": {
    "batch_size": 16,
    "learning_rate": 5e-6,
    "num_train_steps": 5000,
    "max_seq_length": 1024,
    "max_new_tokens": 8
  }
}
```

### Local execution config
Confirm `config/env_config.json` contains:
- `"execution": "local_openspiel"`

## 🎯 Training (recommended first run: 10–48 hours)

### Start Training

```bash
# Basic training
python train_ppo_lora.py

# Resume from checkpoint
python train_ppo_lora.py --resume_from checkpoints/game_ppo_lora/checkpoint-5000
```

Note: `train_ppo_lora.py` always reads `config/train_config.json` and `config/env_config.json`.

### Monitor Training

**Console Output:**
```
Step 100 | Reward: 0.4523 | Score: 0.6012 | Success: 45.00%
🎓 Curriculum Progression: Advanced to stage 2/3
```

**Weights & Biases:**
- Navigate to https://wandb.ai/your-username/game-rl-training
- View real-time metrics, system stats, and training curves

**Tensorboard (if not using wandb):**
```bash
tensorboard --logdir checkpoints/game_ppo_lora/logs
```

### Expected Timeline

| Steps | Time | Description | Expected Success Rate |
|-------|------|-------------|----------------------|
| 0-500 | ~1–6h | Sanity check (learning starts) | 20–40% |
| 500-2000 | ~6–24h | Stronger play | 35–55% |
| 2000-5000 | ~1–2 days | “Enough” first run | 40–65% |

## 📊 Evaluation

### Evaluate a checkpoint
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/game_ppo_lora/checkpoint-5000 \
  --num_episodes 100 \
  --opponent random
```

### Evaluate against MCTS opponent
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/game_ppo_lora/final \
  --num_episodes 100 \
  --opponent mcts
```

## 💾 Exporting Models

### Merge LoRA adapters with base model
```bash
python scripts/merge.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapters checkpoints/game_ppo_lora/final/model \
  --output merged_models/game_qwen_final
```

### Upload to Hugging Face Hub
```bash
huggingface-cli login
huggingface-cli upload your-username/game-qwen-ppo merged_models/game_qwen_final
```

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

**Symptoms:** CUDA out of memory error during training

**Solutions:**
1. Enable 4-bit quantization: `"use_4bit": true`
2. Reduce batch size: `"batch_size": 4`
3. Reduce max sequence length: `"max_seq_length": 1024`
4. Increase gradient accumulation: `"gradient_accumulation_steps": 8`

```json
// config/train_config.json
{
  "model": {"use_4bit": true},
  "ppo": {
    "batch_size": 4,
    "gradient_accumulation_steps": 8
  }
}
```

### Issue 2: Environment Connection Failed

**Symptoms:** `import pyspiel` fails or OpenSpiel isn't installed

**Solutions:**
1. Run the setup check:
   ```bash
   bash scripts/setup.sh
   ```
2. Install OpenSpiel on the training machine (see the section below).

## 🧩 Installing OpenSpiel (`pyspiel`) on the H100 machine

OpenSpiel installation varies by OS / toolchain. When you get access to the H100 box, tell me:
- Ubuntu version
- CUDA version
- whether you can use conda

And I’ll give you a one-command install path that works reliably for your setup.

### Issue 3: Training Unstable / NaN Loss

**Symptoms:** Loss becomes NaN, rewards collapse

**Solutions:**
1. Reduce learning rate: `1e-5` → `5e-6`
2. Increase PPO epochs: `4` → `6`
3. Adjust KL coefficient: `"init_kl_coef": 0.1`
4. Enable gradient clipping (add to code)

### Issue 4: Slow Progress / Low Success Rate

**Symptoms:** Success rate stuck below 30%

**Solutions:**
1. Increase temperature for more exploration: `0.7` → `0.9`
2. Adjust curriculum progression threshold
3. Increase failure replay probability: `0.3` → `0.5`
4. Check that rules are being enforced by environment

### Issue 5: Checkpoint Too Large

**Symptoms:** Checkpoints consuming too much disk space

**Solutions:**
1. Increase save frequency: `500` → `1000`
2. Only save LoRA adapters (default behavior)
3. Don't save optimizer state (modify save_checkpoint)
4. Use automatic checkpoint cleanup

## 📈 Optimizing Performance

### For Maximum Speed
```json
{
  "model": {"use_4bit": true},
  "ppo": {
    "batch_size": 16,  // Increase if you have VRAM
    "gradient_accumulation_steps": 1,
    "ppo_epochs": 2  // Faster but less stable
  }
}
```

### For Maximum Quality
```json
{
  "model": {"use_4bit": false},  // Full precision
  "lora": {"r": 32},  // Higher rank
  "ppo": {
    "batch_size": 4,  // Smaller but more stable
    "learning_rate": 5e-6,  // Lower LR
    "ppo_epochs": 6  // More optimization
  }
}
```

### For Limited VRAM (<16GB)
```json
{
  "model": {"use_4bit": true},
  "ppo": {
    "batch_size": 2,
    "mini_batch_size": 1,
    "gradient_accumulation_steps": 16
  }
}
```

## 🎓 Understanding the Training Process

### How PPO Works

1. **Rollout Collection**: Model plays games and collects experiences
2. **Advantage Estimation**: Compute how good each action was
3. **Policy Update**: Update model to take better actions
4. **Value Update**: Update value function to predict rewards
5. **KL Constraint**: Keep updates close to reference model

### How LoRA Works

Instead of updating all 3 billion parameters, LoRA:
- Freezes the base model weights
- Adds small "adapter" matrices to attention layers
- Only trains ~15 million parameters (0.5% of total)
- Results in 500MB checkpoints instead of 6GB

### How Curriculum Learning Works

Training progresses through 3 stages:

1. **Easy Stage**: Random opponent, basic games
   - Goal: Learn game mechanics and valid moves
   - Threshold: 60% success rate

2. **Medium Stage**: Random opponent, complex games
   - Goal: Develop strategic thinking
   - Threshold: 50% success rate

3. **Hard Stage**: MCTS opponent, all games
   - Goal: Master advanced tactics
   - Threshold: 40% success rate

### How Failure-Based Sampling Works

- Maintains buffer of 1000 most recent failed/low-scoring tasks
- 30% of training samples come from this buffer
- Priority sampling: Worse performance = higher probability
- Ensures model revisits difficult tasks until mastered

## 📚 Next Steps

### After Training

1. **Evaluate thoroughly**
   ```bash
   python scripts/evaluate.py --checkpoint final --num_episodes 1000
   ```

2. **Merge and export**
   ```bash
   python scripts/merge.py --base Qwen/... --adapters ... --output ...
   ```

3. **Deploy to production**
   - Upload to Hugging Face Hub
   - Create inference endpoint
   - Integrate with your application

4. **Continue training** (optional)
   - Resume with lower learning rate
   - Train on hard stage longer
   - Add more curriculum stages

### Further Improvements

1. **Reward Shaping**: Customize rewards in `env_wrapper.py`
2. **Advanced Curriculum**: Add more stages or dynamic difficulty
3. **Multi-Task Training**: Train on multiple environments
4. **Hyperparameter Tuning**: Use wandb sweeps for optimization

## 🆘 Getting Help

If you encounter issues:

1. Check this guide first
2. Review training logs: `checkpoints/game_ppo_lora/logs/`
3. Check wandb dashboard for anomalies
4. Review README.md for detailed documentation
5. Search GitHub issues

## ✅ Quick Checklist

Before starting training, verify:
- [ ] GPU has sufficient VRAM (16GB+)
- [ ] CUDA is installed and working
- [ ] Python dependencies installed
- [ ] `CHUTES_API_KEY` is set
- [ ] Environment connection successful
- [ ] Configuration files are correct
- [ ] Output directories exist
- [ ] You have ~200GB free disk space
- [ ] Training will run for 80-100 hours

## 🎯 Success Criteria

Your training is successful when:
- ✅ No NaN losses or training crashes
- ✅ Success rate improves over time
- ✅ Model advances through curriculum stages
- ✅ Final evaluation score >0.5 on easy stage
- ✅ Final evaluation score >0.4 on medium stage
- ✅ Final evaluation score >0.3 on hard stage

Good luck with your training! 🚀
