#!/usr/bin/env python3
"""
Stage 3: PPO/RL Training for Trace Environment

This script performs reinforcement learning on the trace task using PPO.
The model learns from actual code execution feedback in the trace environment.

Purpose:
- Fine-tune the model with RL for better trace performance
- Learn from binary rewards (exact match vs no match)
- Optional curriculum learning (easy → medium → hard programs)

Prerequisites:
- Run Stage 1 and Stage 2 first (or at least Stage 2)
- Have the merged Stage 2 model available

Usage:
    # Basic training with Stage 2 model
    python train_stage3_ppo.py
    
    # Use custom config
    python train_stage3_ppo.py --config config/stage3_ppo_config.json
    
    # Specify base model explicitly
    python train_stage3_ppo.py --base_model ./checkpoints/stage2_sft/merged
    
    # Quick test run
    python train_stage3_ppo.py --num_steps 100 --eval_freq 10
    
    # Resume training
    python train_stage3_ppo.py --resume ./checkpoints/stage3_ppo/step_1000
"""

import os
import sys
import json
import random
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add path for trace_task imports
TRACE_ENV_PATH = "/root/workstation/sn120/affine_repo/affinetes/environments/trace"
sys.path.insert(0, TRACE_ENV_PATH)

try:
    from trace_task import (
        TraceTask,
        inject_non_overfittable_prints,
        run_code_sync,
        clean_llm_prediction,
        compare_outputs,
    )
    TRACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import trace_task: {e}")
    TRACE_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG_PATH = "./config/stage3_ppo_config.json"
DEFAULT_BASE_MODEL = "./checkpoints/stage2_sft/merged"
DEFAULT_OUTPUT_DIR = "./checkpoints/stage3_ppo"


@dataclass
class PPOTrainingState:
    """Training state for checkpointing"""
    step: int = 0
    total_reward: float = 0.0
    success_count: int = 0
    total_count: int = 0
    current_difficulty: str = "easy"
    best_success_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "total_reward": self.total_reward,
            "success_count": self.success_count,
            "total_count": self.total_count,
            "current_difficulty": self.current_difficulty,
            "best_success_rate": self.best_success_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PPOTrainingState":
        return cls(**data)


class TraceRewardComputer:
    """Compute rewards for trace task predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.exact_match_reward = config.get("exact_match_reward", 1.0)
        self.partial_match_enabled = config.get("partial_match_enabled", True)
        self.partial_match_max = config.get("partial_match_max", 0.5)
        self.invalid_penalty = config.get("invalid_penalty", -0.1)
        self.timeout_penalty = config.get("timeout_penalty", -0.2)
    
    def compute_reward(
        self,
        prediction: str,
        ground_truth: str,
        error: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for a prediction."""
        
        info = {
            "exact_match": False,
            "partial_score": 0.0,
            "error": error,
        }
        
        if error:
            if "timeout" in error.lower():
                return self.timeout_penalty, info
            return self.invalid_penalty, info
        
        # Clean prediction
        cleaned = clean_llm_prediction(prediction) if prediction else ""
        
        # Check exact match
        if compare_outputs(ground_truth, cleaned):
            info["exact_match"] = True
            return self.exact_match_reward, info
        
        # Partial credit
        if self.partial_match_enabled and cleaned and ground_truth:
            pred_lines = cleaned.strip().split('\n')
            truth_lines = ground_truth.strip().split('\n')
            
            if truth_lines:
                matching = sum(
                    1 for p, t in zip(pred_lines, truth_lines)
                    if p.strip().lower() == t.strip().lower()
                )
                partial = matching / len(truth_lines)
                info["partial_score"] = partial
                return partial * self.partial_match_max, info
        
        return 0.0, info


class TracePPOTrainer:
    """PPO Trainer for Trace Environment"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_model_path: str,
        output_dir: str,
    ):
        self.config = config
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.trace_task = TraceTask() if TRACE_AVAILABLE else None
        self.reward_computer = TraceRewardComputer(config.get("reward", {}))
        self.state = PPOTrainingState()
        
        # Curriculum settings
        curriculum_config = config.get("curriculum", {})
        self.curriculum_enabled = curriculum_config.get("enabled", False)
        self.difficulty_levels = curriculum_config.get("difficulty_levels", ["easy", "medium", "hard"])
        self.promotion_threshold = curriculum_config.get("promotion_threshold", 0.7)
        self.promotion_window = curriculum_config.get("promotion_window", 100)
        self.easy_max_lines = curriculum_config.get("easy_max_lines", 20)
        self.medium_max_lines = curriculum_config.get("medium_max_lines", 50)
        
        # Recent results for curriculum
        self.recent_results: List[bool] = []
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load model with LoRA for PPO training."""
        
        print(f"\n📦 Loading model: {self.base_model_path}")
        
        model_config = self.config.get("model", {})
        lora_config = self.config.get("lora", {})
        ppo_config = self.config.get("ppo", {})
        
        # Quantization
        if model_config.get("use_4bit", True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=model_config.get("use_nested_quant", True),
            )
        else:
            bnb_config = None
        
        # Check attention implementation
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with value head for PPO
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.base_model_path,
            **model_kwargs,
        )
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model.pretrained_model = get_peft_model(self.model.pretrained_model, peft_config)
        
        # Create PPO config
        self.ppo_trainer_config = PPOConfig(
            learning_rate=ppo_config.get("learning_rate", 5e-7),
            batch_size=ppo_config.get("batch_size", 16),
            mini_batch_size=ppo_config.get("mini_batch_size", 4),
            gradient_accumulation_steps=ppo_config.get("gradient_accumulation_steps", 4),
            ppo_epochs=ppo_config.get("ppo_epochs", 2),
            kl_coef=ppo_config.get("init_kl_coef", 0.05),
            cliprange=ppo_config.get("cliprange", 0.2),
            vf_coef=ppo_config.get("vf_coef", 0.1),
            max_grad_norm=ppo_config.get("max_grad_norm", 1.0),
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_trainer_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print(f"   Model loaded with LoRA")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _get_task_ids_for_difficulty(self, difficulty: str, count: int) -> List[int]:
        """Get task IDs appropriate for current difficulty level."""
        
        if not self.trace_task:
            return list(range(count))
        
        dataset_size = len(self.trace_task.dataset)
        
        # Sample task IDs and filter by difficulty
        candidates = random.sample(range(dataset_size), min(count * 10, dataset_size))
        selected = []
        
        for task_id in candidates:
            if len(selected) >= count:
                break
            
            sample = self.trace_task.dataset[task_id]
            program = sample.get("program", "")
            lines = len(program.strip().split('\n'))
            
            if difficulty == "easy" and lines <= self.easy_max_lines:
                selected.append(task_id)
            elif difficulty == "medium" and self.easy_max_lines < lines <= self.medium_max_lines:
                selected.append(task_id)
            elif difficulty == "hard" and lines > self.medium_max_lines:
                selected.append(task_id)
            elif not self.curriculum_enabled:
                selected.append(task_id)
        
        # Fill remaining with random if needed
        while len(selected) < count:
            selected.append(random.randint(0, dataset_size - 1))
        
        return selected
    
    def _maybe_promote_difficulty(self):
        """Check if we should increase difficulty level."""
        
        if not self.curriculum_enabled:
            return
        
        if len(self.recent_results) < self.promotion_window:
            return
        
        recent_success_rate = sum(self.recent_results[-self.promotion_window:]) / self.promotion_window
        
        if recent_success_rate >= self.promotion_threshold:
            current_idx = self.difficulty_levels.index(self.state.current_difficulty)
            if current_idx < len(self.difficulty_levels) - 1:
                self.state.current_difficulty = self.difficulty_levels[current_idx + 1]
                print(f"\n🎯 Promoted to difficulty: {self.state.current_difficulty}")
                self.recent_results = []  # Reset
    
    async def _generate_and_evaluate_batch(
        self,
        task_ids: List[int],
    ) -> List[Tuple[str, str, str, float, Dict]]:
        """Generate challenges and get model predictions for a batch."""
        
        ppo_config = self.config.get("ppo", {})
        max_new_tokens = ppo_config.get("max_new_tokens", 1024)
        temperature = ppo_config.get("temperature", 0.7)
        top_p = ppo_config.get("top_p", 0.9)
        
        results = []
        
        for task_id in task_ids:
            try:
                # Generate challenge
                challenge = await self.trace_task.generate(task_id=task_id)
                prompt = challenge.prompt
                ground_truth = challenge.extra.get("ground_truth", "")
                
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=ppo_config.get("max_seq_length", 4096) - max_new_tokens,
                ).to(self.model.pretrained_model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.pretrained_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Compute reward
                reward, info = self.reward_computer.compute_reward(
                    response, ground_truth
                )
                
                results.append((prompt, response, ground_truth, reward, info))
                
            except Exception as e:
                print(f"   Error on task {task_id}: {e}")
                results.append(("", "", "", self.reward_computer.invalid_penalty, {"error": str(e)}))
        
        return results
    
    def train(
        self,
        num_steps: int,
        eval_freq: int = 50,
        save_freq: int = 100,
        log_freq: int = 1,
    ):
        """Main training loop."""
        
        print("\n" + "=" * 70)
        print("🚀 Stage 3: PPO Training")
        print("=" * 70)
        print(f"   Steps:        {num_steps}")
        print(f"   Eval freq:    {eval_freq}")
        print(f"   Save freq:    {save_freq}")
        print(f"   Curriculum:   {self.curriculum_enabled}")
        print("=" * 70)
        
        ppo_config = self.config.get("ppo", {})
        batch_size = ppo_config.get("batch_size", 16)
        
        # Initialize wandb
        use_wandb = ppo_config.get("use_wandb", False) and WANDB_AVAILABLE
        if use_wandb:
            wandb.init(
                project=ppo_config.get("wandb_project", "trace-rl"),
                name=ppo_config.get("wandb_run_name", "stage3-ppo"),
                config=self.config,
            )
        
        # Training loop
        for step in range(self.state.step, num_steps):
            step_start = time.time()
            
            # Get task IDs for current difficulty
            task_ids = self._get_task_ids_for_difficulty(
                self.state.current_difficulty,
                batch_size,
            )
            
            # Generate and evaluate
            results = asyncio.run(self._generate_and_evaluate_batch(task_ids))
            
            # Prepare for PPO update
            queries = []
            responses = []
            rewards = []
            
            for prompt, response, ground_truth, reward, info in results:
                if prompt and response:
                    query_tensor = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                    response_tensor = self.tokenizer(response, return_tensors="pt")["input_ids"][0]
                    
                    queries.append(query_tensor)
                    responses.append(response_tensor)
                    rewards.append(torch.tensor([reward]))
                    
                    # Track success
                    success = info.get("exact_match", False)
                    self.recent_results.append(success)
                    self.state.success_count += int(success)
                    self.state.total_count += 1
            
            # PPO update
            if queries and responses:
                try:
                    stats = self.ppo_trainer.step(queries, responses, rewards)
                except Exception as e:
                    print(f"   PPO step error: {e}")
                    stats = {}
            else:
                stats = {}
            
            # Update state
            self.state.step = step + 1
            self.state.total_reward += sum(r.item() for r in rewards)
            
            # Check curriculum promotion
            self._maybe_promote_difficulty()
            
            # Logging
            if step % log_freq == 0:
                success_rate = self.state.success_count / max(self.state.total_count, 1)
                avg_reward = self.state.total_reward / max(self.state.total_count, 1)
                step_time = time.time() - step_start
                
                print(f"Step {step+1}/{num_steps} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"Success: {success_rate:.1%} | "
                      f"Difficulty: {self.state.current_difficulty} | "
                      f"Time: {step_time:.1f}s")
                
                if use_wandb:
                    wandb.log({
                        "step": step + 1,
                        "avg_reward": avg_reward,
                        "success_rate": success_rate,
                        "difficulty": self.state.current_difficulty,
                        "step_time": step_time,
                        **stats,
                    })
            
            # Save checkpoint
            if step % save_freq == 0 and step > 0:
                self._save_checkpoint(step)
            
            # Evaluation
            if step % eval_freq == 0 and step > 0:
                self._evaluate(num_samples=50)
        
        # Final save
        self._save_checkpoint(num_steps, final=True)
        
        if use_wandb:
            wandb.finish()
        
        print("\n" + "=" * 70)
        print("✅ Stage 3 Complete!")
        print("=" * 70)
        print(f"   Final success rate: {self.state.success_count / max(self.state.total_count, 1):.1%}")
        print(f"   Best success rate:  {self.state.best_success_rate:.1%}")
        print(f"   Checkpoints:        {self.output_dir}")
        print("=" * 70)
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save training checkpoint."""
        
        suffix = "final" if final else f"step_{step}"
        checkpoint_dir = self.output_dir / suffix
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save state
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        print(f"   💾 Saved checkpoint: {checkpoint_dir}")
    
    def _evaluate(self, num_samples: int = 50):
        """Run evaluation."""
        
        print(f"\n📊 Evaluating on {num_samples} samples...")
        
        task_ids = random.sample(range(len(self.trace_task.dataset)), num_samples)
        results = asyncio.run(self._generate_and_evaluate_batch(task_ids))
        
        successes = sum(1 for _, _, _, _, info in results if info.get("exact_match", False))
        success_rate = successes / num_samples
        
        print(f"   Eval success rate: {success_rate:.1%} ({successes}/{num_samples})")
        
        if success_rate > self.state.best_success_rate:
            self.state.best_success_rate = success_rate
            print(f"   🎉 New best!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main(args):
    """Main function."""
    
    if not TRACE_AVAILABLE:
        print("ERROR: trace_task module not available.")
        print(f"Please ensure {TRACE_ENV_PATH}/trace_task.py exists.")
        sys.exit(1)
    
    # Load config
    config_path = args.config or DEFAULT_CONFIG_PATH
    if Path(config_path).exists():
        config = load_config(config_path)
        print(f"📋 Loaded config: {config_path}")
    else:
        print(f"⚠️  Config not found: {config_path}")
        print("   Using default configuration.")
        config = {}
    
    # Override with command line args
    base_model = args.base_model or config.get("model", {}).get("model_name", DEFAULT_BASE_MODEL)
    output_dir = args.output_dir or config.get("ppo", {}).get("output_dir", DEFAULT_OUTPUT_DIR)
    num_steps = args.num_steps or config.get("ppo", {}).get("num_train_steps", 5000)
    eval_freq = args.eval_freq or config.get("ppo", {}).get("eval_freq", 50)
    save_freq = args.save_freq or config.get("ppo", {}).get("save_freq", 100)
    
    # Check base model
    if not Path(base_model).exists():
        print(f"⚠️  Base model not found: {base_model}")
        fallback = config.get("model", {}).get("base_model_fallback", "Qwen/Qwen3-4B")
        print(f"   Using fallback: {fallback}")
        base_model = fallback
    
    # Create trainer
    trainer = TracePPOTrainer(
        config=config,
        base_model_path=base_model,
        output_dir=output_dir,
    )
    
    # Resume if specified
    if args.resume:
        resume_path = Path(args.resume)
        state_file = resume_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                trainer.state = PPOTrainingState.from_dict(json.load(f))
            print(f"📋 Resumed from step {trainer.state.step}")
    
    # Train
    trainer.train(
        num_steps=num_steps,
        eval_freq=eval_freq,
        save_freq=save_freq,
        log_freq=args.log_freq,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: PPO Training for Trace Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    main(args)
