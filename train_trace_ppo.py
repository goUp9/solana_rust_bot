#!/usr/bin/env python3
"""
PPO + LoRA Training for Trace Environment (Code Tracing Task)
Model: Qwen3-4B
Training: PPO (Proximal Policy Optimization) + LoRA (Low-Rank Adaptation)
Framework: TRL + PEFT
Environment: Affinetes Trace environment (Docker-based)
Task: Predict exact stdout output of Python programs with injected debug prints
"""

import os
import sys
import json
import torch
import asyncio
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logging, save_checkpoint, load_checkpoint

# Try to import affinetes
try:
    import affinetes as af_env
    AFFINETES_AVAILABLE = True
except ImportError:
    AFFINETES_AVAILABLE = False
    print("Warning: affinetes not available. Install with: pip install -e /root/workspace/affinetes")


def _load_json_if_exists(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@dataclass
class TurnSample:
    """One LLM turn sample for PPO."""
    prompt_text: str
    response_text: str
    reward: float
    query_tensor: Optional[torch.Tensor] = None
    response_tensor: Optional[torch.Tensor] = None


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True


@dataclass
class LoRAConfigData:
    """LoRA configuration"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class PPOTrainingConfig:
    """PPO training configuration (TRL 0.27.0 compatible)"""
    # Batch settings
    batch_size: int = 4  # per_device_train_batch_size
    mini_batch_size: int = 4  # Used to calculate num_mini_batches
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    ppo_epochs: int = 2  # num_ppo_epochs in TRL 0.27.0
    
    # KL and clipping (TRL 0.27.0: init_kl_coef -> kl_coef, target_kl/adap_kl_ctrl removed)
    init_kl_coef: float = 0.05  # Maps to kl_coef
    target_kl: float = 6.0  # Kept for config compatibility, not used in TRL 0.27.0
    adap_kl_ctrl: bool = True  # Kept for config compatibility, not used in TRL 0.27.0
    
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # GAE parameters
    gamma: float = 1.0
    lam: float = 0.95
    max_grad_norm: float = 1.0
    
    num_train_steps: int = 2000
    save_freq: int = 50
    eval_freq: int = 25
    log_freq: int = 1
    
    max_seq_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    output_dir: str = "./checkpoints/trace_ppo_lora"
    resume_from: Optional[str] = None
    
    use_wandb: bool = True
    wandb_project: str = "trace-rl-training"
    wandb_run_name: Optional[str] = None
    
    include_invalid_samples: bool = False
    invalid_penalty: float = -0.5
    sft_warmup_steps: int = 50


class TracePPOTrainer:
    """PPO trainer for Trace environment using affinetes"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfigData,
        ppo_config: PPOTrainingConfig,
        env_config: Dict[str, Any],
        train_config_path: str = None,
        env_config_path: str = None,
    ):
        self.model_config = model_config
        self.lora_config_data = lora_config
        self.ppo_config = ppo_config
        self.env_config = env_config
        
        self.train_config_path = train_config_path
        self.env_config_path = env_config_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = setup_logging()
        
        # Will be initialized in _setup()
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.ppo_trainer = None
        self.env = None
        
        self._setup()
    
    def _setup(self):
        """Initialize all components"""
        self.logger.info("Setting up PPO+LoRA training for Trace environment...")
        
        # Initialize wandb
        if self.ppo_config.use_wandb:
            wandb.login()
            wandb.init(
                project=self.ppo_config.wandb_project,
                name=self.ppo_config.wandb_run_name or f"trace-ppo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config_data.__dict__,
                    "ppo": self.ppo_config.__dict__,
                    "env": self.env_config,
                }
            )
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.logger.info(f"Loading base model: {self.model_config.model_name}")
        bnb_config = None
        if self.model_config.use_4bit:
            compute_dtype = getattr(torch, self.model_config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_config.use_nested_quant,
            )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.model_config.use_4bit else torch.float16,
        )
        
        # Apply LoRA
        self.logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=self.lora_config_data.r,
            lora_alpha=self.lora_config_data.lora_alpha,
            target_modules=self.lora_config_data.target_modules,
            lora_dropout=self.lora_config_data.lora_dropout,
            bias=self.lora_config_data.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        peft_model = get_peft_model(base_model, lora_config)
        
        # Wrap with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
        
        # Create reference model
        self.logger.info("Creating reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.model_config.use_4bit else torch.float16,
        )
        
        # Initialize PPO trainer
        self.logger.info("Initializing PPO trainer...")
        # TRL 0.27.0 PPOConfig API
        ppo_config_obj = PPOConfig(
            output_dir=self.ppo_config.output_dir,
            learning_rate=self.ppo_config.learning_rate,
            per_device_train_batch_size=self.ppo_config.batch_size,
            gradient_accumulation_steps=self.ppo_config.gradient_accumulation_steps,
            num_ppo_epochs=self.ppo_config.ppo_epochs,
            num_mini_batches=max(1, self.ppo_config.batch_size // self.ppo_config.mini_batch_size),
            kl_coef=self.ppo_config.init_kl_coef,
            cliprange=self.ppo_config.cliprange,
            cliprange_value=self.ppo_config.cliprange_value,
            vf_coef=self.ppo_config.vf_coef,
            gamma=self.ppo_config.gamma,
            lam=self.ppo_config.lam,
            max_grad_norm=self.ppo_config.max_grad_norm,
            whiten_rewards=True,
            temperature=self.ppo_config.temperature,
            response_length=self.ppo_config.max_new_tokens,
            report_to="wandb" if self.ppo_config.use_wandb else None,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config_obj,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Initialize trace task (local execution - no Docker needed)
        self.logger.info("Using local trace execution (no Docker/affinetes required)")
        self.env = None  # Not using affinetes for local execution
        
        # Resume from checkpoint if specified
        if self.ppo_config.resume_from:
            self._resume_from_checkpoint(self.ppo_config.resume_from)
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            metadata = load_checkpoint(
                self.model,
                self.tokenizer,
                Path(checkpoint_path)
            )
            self.start_step = metadata.get("step", 0)
            self.logger.info(f"Resumed from step {self.start_step}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    async def _run_trace_episode(
        self,
        task_id: int,
        seed: int,
    ) -> Tuple[List[TurnSample], Dict[str, Any]]:
        """Run a single Trace episode"""
        
        if self.env is None:
            # Mock episode for testing
            return [], {"score": 0.0, "success": False, "error": "No environment"}
        
        # Generate prompt from environment
        try:
            result = await self.env.evaluate(
                model="local",  # We'll handle inference ourselves
                base_url="http://localhost:8000",
                timeout=self.env_config.get("timeout", 120),
                temperature=self.ppo_config.temperature,
                seed=seed,
                task_id=task_id,
            )
        except Exception as e:
            self.logger.error(f"Environment error: {e}")
            return [], {"score": 0.0, "success": False, "error": str(e)}
        
        # For now, we need to modify the approach since the env.evaluate() 
        # calls the LLM internally. We need to intercept the prompt.
        # This requires modifications to the trace environment or using a different approach.
        
        # Alternative approach: Generate challenge, get prompt, run our model, then evaluate
        # This would require the environment to expose generate() and evaluate() separately
        
        score = result.get("score", 0.0)
        conversation = result.get("extra", {}).get("conversation", [])
        
        turn_samples = []
        if conversation and len(conversation) >= 2:
            prompt = conversation[0].get("content", "")
            response = conversation[1].get("content", "")
            
            if prompt and response:
                turn_samples.append(TurnSample(
                    prompt_text=prompt,
                    response_text=response,
                    reward=score,
                ))
        
        episode_info = {
            "score": score,
            "success": score > 0,
            "task_id": task_id,
            "seed": seed,
        }
        
        return turn_samples, episode_info
    
    async def _run_local_trace_episode(
        self,
        task_id: int,
        seed: int,
    ) -> Tuple[List[TurnSample], Dict[str, Any]]:
        """
        Run a Trace episode with local model inference.
        
        Uses the local_trace module to:
        1. Generate challenge from the dataset
        2. Run inference with our model
        3. Evaluate the response
        """
        try:
            # Import local trace module
            from local_trace import LocalTraceTask, run_local_trace_episode
            
            # Initialize trace task if not already done
            if not hasattr(self, '_trace_task'):
                self._trace_task = LocalTraceTask(
                    dataset_name=self.env_config.get("dataset_name", "satpalsr/rl-python"),
                    dataset_split=self.env_config.get("dataset_split", "train"),
                    hf_token=self.env_config.get("hf_token"),
                )
            
            # Run episode with local model inference
            # deterministic=True ensures same task_id produces same challenge (required for PPO)
            turn_samples, episode_info, episode_data = run_local_trace_episode(
                model=self.model,
                tokenizer=self.tokenizer,
                task_id=task_id,
                seed=None,  # Let it be derived from task_id for determinism
                temperature=self.ppo_config.temperature,
                top_p=self.ppo_config.top_p,
                max_new_tokens=self.ppo_config.max_new_tokens,
                max_seq_length=self.ppo_config.max_seq_length,
                device=str(self.device),
                trace_task=self._trace_task,
                deterministic=self.env_config.get("deterministic_seed", True),
            )
            
            return turn_samples, episode_info
            
        except Exception as e:
            self.logger.error(f"Episode error: {e}")
            import traceback
            traceback.print_exc()
            return [], {"score": 0.0, "success": False, "error": str(e)}
    
    async def collect_rollouts(self, batch_size: int, step: int) -> Dict:
        """Collect rollouts for PPO training - TRL compatible format"""
        
        all_samples = []
        episode_infos = []
        metadata = []
        
        # Generate task configs
        task_id_range = self.env_config.get("task_id_range", [0, 100000])
        
        # Collect more episodes to ensure enough valid samples
        max_attempts = batch_size * 5
        for i in range(max_attempts):
            task_id = random.randint(task_id_range[0], task_id_range[1])
            seed = random.randint(0, 2**31 - 1)
            
            samples, info = await self._run_local_trace_episode(task_id, seed)
            
            if samples:  # Only add if we got valid samples
                all_samples.extend(samples)
                episode_infos.append(info)
                metadata.append({
                    "score": info.get("score", 0.0),
                    "success": info.get("success", False),
                    "task_id": task_id,
                })
            
            if len(all_samples) >= batch_size:
                break
        
        # Prepare tensors in TRL PPOTrainer.step() compatible format
        # PPO trainer expects:
        # - queries: List of 1D tensors (prompt token IDs)
        # - responses: List of 1D tensors (response token IDs)
        # - rewards: List of scalar tensors
        queries = []
        responses = []
        rewards = []
        
        for sample in all_samples[:batch_size]:
            # Tokenize prompt - apply chat template if available
            messages = [{"role": "user", "content": sample.prompt_text}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = sample.prompt_text
            
            # Tokenize query (prompt) - 1D tensor
            query_ids = self.tokenizer.encode(
                formatted_prompt,
                truncation=True,
                max_length=self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens,
                add_special_tokens=False,
            )
            
            # Tokenize response - 1D tensor
            response_ids = self.tokenizer.encode(
                sample.response_text,
                truncation=True,
                max_length=self.ppo_config.max_new_tokens,
                add_special_tokens=False,
            )
            
            # Convert to tensors (1D, on device)
            queries.append(torch.tensor(query_ids, dtype=torch.long, device=self.device))
            responses.append(torch.tensor(response_ids, dtype=torch.long, device=self.device))
            
            # Reward as scalar tensor
            rewards.append(torch.tensor(sample.reward, dtype=torch.float32))
        
        # Calculate stats
        scores = [info.get("score", 0.0) for info in episode_infos]
        success_rate = sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0
        
        return {
            "queries": queries,
            "responses": responses,
            "rewards": rewards,
            "metadata": metadata,  # Required for train_step stats
            "episode_infos": episode_infos,
            "stats": {
                "mean_reward": sum(scores) / len(scores) if scores else 0.0,
                "success_rate": success_rate,
                "num_episodes": len(episode_infos),
                "num_samples": len(queries),
            }
        }
    
    def train_step(self, rollouts: Dict) -> Dict:
        """Single PPO training step - same format as game training"""
        queries = rollouts["queries"]
        responses = rollouts["responses"]
        rewards = rollouts["rewards"]
        
        # PPO trainer expects batch_size samples; process in batches
        all_stats = []
        batch_size = self.ppo_config.batch_size
        num_samples = len(queries)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_queries = queries[i:end_idx]
            batch_responses = responses[i:end_idx]
            batch_rewards = rewards[i:end_idx]
            
            # Skip incomplete batches at the end
            if len(batch_queries) < batch_size:
                continue
            
            # Run PPO step on this batch
            stats = self.ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
            all_stats.append(stats)
        
        # Aggregate stats from all batches
        if all_stats:
            stats = {}
            for k in all_stats[0].keys():
                values = [s.get(k, 0) for s in all_stats]
                if isinstance(values[0], (int, float)):
                    stats[k] = sum(values) / len(values)
                elif hasattr(values[0], 'item'):
                    try:
                        stats[k] = sum(v.item() if hasattr(v, 'item') else v for v in values) / len(values)
                    except:
                        pass
        else:
            stats = {}
        
        # Add custom metrics
        if rewards:
            stats["mean_reward"] = torch.mean(torch.stack(rewards)).item()
        else:
            stats["mean_reward"] = 0.0
        
        if rollouts.get("metadata"):
            stats["mean_score"] = sum(m["score"] for m in rollouts["metadata"]) / len(rollouts["metadata"])
            stats["success_rate"] = sum(m["success"] for m in rollouts["metadata"]) / len(rollouts["metadata"])
        else:
            stats["mean_score"] = 0.0
            stats["success_rate"] = 0.0
        
        stats["num_samples"] = num_samples
        stats["num_batches"] = len(all_stats)
        
        return stats
    
    async def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training for Trace environment...")
        
        start_step = getattr(self, 'start_step', 0)
        output_dir = Path(self.ppo_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for step in range(start_step, self.ppo_config.num_train_steps):
            step_start = time.time()
            
            # Collect more episodes than batch_size to ensure enough valid samples
            num_episodes = max(16, self.ppo_config.batch_size * 4)
            self.logger.info(f"Step {step + 1}/{self.ppo_config.num_train_steps}: Collecting rollouts...")
            
            rollout_data = await self.collect_rollouts(num_episodes, step)
            
            num_samples = len(rollout_data["queries"])
            if num_samples < self.ppo_config.batch_size:
                self.logger.warning(f"Step {step + 1}: Only {num_samples} valid samples (need {self.ppo_config.batch_size}). Skipping.")
                continue
            
            # PPO update using train_step (same as game training)
            self.logger.info(f"Training with {num_samples} samples...")
            try:
                stats = self.train_step(rollout_data)
            except Exception as e:
                self.logger.error(f"PPO step failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Logging
            if (step + 1) % self.ppo_config.log_freq == 0:
                step_time = time.time() - step_start
                self.logger.info(
                    f"Step {step + 1} | "
                    f"Reward: {stats.get('mean_reward', 0):.4f} | "
                    f"Score: {stats.get('mean_score', 0):.4f} | "
                    f"Success: {stats.get('success_rate', 0):.2%} | "
                    f"Time: {step_time:.1f}s"
                )
                
                if self.ppo_config.use_wandb:
                    wandb.log({
                        "step": step + 1,
                        **stats,
                        "step_time": step_time,
                    })
            
            # Save checkpoint
            if (step + 1) % self.ppo_config.save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint-{step + 1}"
                save_checkpoint(
                    self.model,
                    self.tokenizer,
                    checkpoint_path,
                    training_args={"step": step + 1},
                )
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Final save
        final_path = output_dir / "final"
        save_checkpoint(self.model, self.tokenizer, final_path)
        self.logger.info(f"Training complete! Final model saved to {final_path}")
        
        if self.ppo_config.use_wandb:
            wandb.finish()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.env is not None:
            try:
                await self.env.cleanup()
            except Exception as e:
                self.logger.warning(f"Environment cleanup error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO+LoRA for Trace environment")
    parser.add_argument("--train-config", type=str, default="config/trace_train_config.json")
    parser.add_argument("--env-config", type=str, default="config/trace_env_config.json")
    args = parser.parse_args()
    
    # Load configs
    train_config_path = Path(args.train_config)
    env_config_path = Path(args.env_config)
    
    train_config = _load_json_if_exists(train_config_path)
    env_config = _load_json_if_exists(env_config_path)
    
    # Create config objects
    model_config = ModelConfig(**train_config.get("model", {}))
    lora_config = LoRAConfigData(**train_config.get("lora", {}))
    ppo_config = PPOTrainingConfig(**train_config.get("ppo", {}))
    
    # Create trainer
    trainer = TracePPOTrainer(
        model_config=model_config,
        lora_config=lora_config,
        ppo_config=ppo_config,
        env_config=env_config,
        train_config_path=str(train_config_path),
        env_config_path=str(env_config_path),
    )
    
    # Run training
    try:
        asyncio.run(trainer.train())
    finally:
        asyncio.run(trainer.cleanup())


if __name__ == "__main__":
    main()
