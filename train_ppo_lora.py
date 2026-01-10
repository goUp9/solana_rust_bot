#!/usr/bin/env python3
"""
PPO + LoRA Training for Game Environment
Model: Qwen3-4B
Training: PPO (Proximal Policy Optimization) + LoRA (Low-Rank Adaptation)
Framework: TRL + PEFT
Environment: Live game server (task-ID aware)
Sampling: Curriculum + failure-based sampling
Rules: Enforced by environment, not model
"""

import os
import sys
import json
import torch
import asyncio
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Conditional imports - only needed for non-local execution
try:
    from affine.core.environments import create_environment
    from game_rl_training.env_wrapper import GameEnvironmentWrapper
except ImportError:
    create_environment = None
    GameEnvironmentWrapper = None
from game_rl_training.curriculum import CurriculumSampler
from game_rl_training.utils import setup_logging, save_checkpoint
from game_rl_training.local_openspiel import run_local_openspiel_episode, EpisodeData

def _load_json_if_exists(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Using Qwen2.5-3B as Qwen3-4B might not be released
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling factor
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class PPOTrainingConfig:
    """PPO training configuration"""
    # PPO hyperparameters
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    
    # PPO-specific
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    
    # Clipping
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # Training
    num_train_steps: int = 10000
    save_freq: int = 500
    eval_freq: int = 100
    log_freq: int = 10
    
    # Environment
    max_seq_length: int = 2048
    # Critical for speed: action-id only generation should be tiny (e.g., "5")
    max_new_tokens: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Paths
    output_dir: str = "./checkpoints/game_ppo_lora"
    resume_from: Optional[str] = None
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "game-rl-training"
    wandb_run_name: Optional[str] = None


class GamePPOTrainer:
    """PPO Trainer for Game Environment"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        ppo_config: PPOTrainingConfig,
        env_name: str = "game",
        env_mode: str = "basilica",
        env_config: Optional[Dict] = None,
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.ppo_config = ppo_config
        self.env_name = env_name
        self.env_mode = env_mode
        self.env_config = env_config or {}
        
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.ppo_trainer = None
        self.env_wrapper = None
        self.curriculum_sampler = None
        self._execution = str(self.env_config.get("execution", "docker")).lower()
        
        self._setup()
    
    def _setup(self):
        """Setup all components"""
        self.logger.info("Setting up PPO+LoRA training...")
        
        # Initialize wandb
        if self.ppo_config.use_wandb:
            wandb.init(
                project=self.ppo_config.wandb_project,
                name=self.ppo_config.wandb_run_name,
                config={
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "ppo": self.ppo_config.__dict__,
                }
            )
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config
        quantization_config = None
        if self.model_config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_config.use_nested_quant,
            )
        
        # Load base model
        self.logger.info(f"Loading base model: {self.model_config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Apply LoRA
        self.logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Create model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            peft_config=peft_config,
        )
        
        # Create reference model (frozen copy for KL divergence)
        self.logger.info("Creating reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.ref_model.eval()
        
        # Setup PPO config
        ppo_config_obj = PPOConfig(
            batch_size=self.ppo_config.batch_size,
            mini_batch_size=self.ppo_config.mini_batch_size,
            gradient_accumulation_steps=self.ppo_config.gradient_accumulation_steps,
            learning_rate=self.ppo_config.learning_rate,
            ppo_epochs=self.ppo_config.ppo_epochs,
            init_kl_coef=self.ppo_config.init_kl_coef,
            target=self.ppo_config.target_kl,
            adap_kl_ctrl=self.ppo_config.adap_kl_ctrl,
            cliprange=self.ppo_config.cliprange,
            cliprange_value=self.ppo_config.cliprange_value,
            vf_coef=self.ppo_config.vf_coef,
            log_with="wandb" if self.ppo_config.use_wandb else None,
            # Reduce KL divergence issues
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=10.0,
        )
        
        # Initialize PPO trainer
        self.logger.info("Initializing PPO trainer...")
        self.ppo_trainer = PPOTrainer(
            config=ppo_config_obj,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Initialize environment wrapper
        if self._execution == "local_openspiel":
            self.logger.info("Using local OpenSpiel execution (in-process inference, no env server, no vLLM).")
            self.env_wrapper = None
        else:
            self.logger.info(f"Initializing environment: {self.env_name} (mode={self.env_mode})")
            env = create_environment(self.env_name, mode=self.env_mode)
            self.env_wrapper = GameEnvironmentWrapper(
                env=env,
                tokenizer=self.tokenizer,
                max_length=self.ppo_config.max_seq_length,
                reward_shaping=bool(self.env_config.get("reward_shaping", True)),
                llm_base_url=str(self.env_config.get("llm_base_url", "http://127.0.0.1:8000/v1")),
                llm_model=str(self.env_config.get("llm_model", self.model_config.model_name)),
                llm_temperature=float(self.env_config.get("llm_temperature", self.ppo_config.temperature)),
                llm_timeout=int(self.env_config.get("timeout", 7200)),
                llm_api_key=self.env_config.get("llm_api_key"),
            )
        
        # Initialize curriculum sampler
        self.logger.info("Initializing curriculum sampler...")
        # Restrict PPO training to OpenSpiel Tier1+Tier2 games:
        # idx 0..7 in `affinetes/environments/openspiel/game_config.py::AVAILABLE_GAMES`
        # 0 goofspiel, 1 liars_dice, 2 leduc_poker, 3 gin_rummy,
        # 4 othello, 5 backgammon, 6 hex, 7 clobber
        allowed_game_indices = list(range(8))
        max_task_id = (max(allowed_game_indices) + 1) * 100_000_000 - 1  # 799,999,999
        self.curriculum_sampler = CurriculumSampler(
            env_name=self.env_name,
            failure_buffer_size=1000,
            curriculum_stages=[
                # Stage ranges kept for readability; actual sampling is constrained by allowed_game_indices.
                {"name": "easy", "task_range": (0, max_task_id), "opponent": "random"},
                {"name": "medium", "task_range": (0, max_task_id), "opponent": "random"},
                {"name": "hard", "task_range": (0, max_task_id), "opponent": "mcts"},
            ]
            ,allowed_game_indices=allowed_game_indices
        )
        
        # Setup episode logging directory
        self.episode_log_dir = Path(self.ppo_config.output_dir) / "episode_logs"
        self.episode_log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_counter = 0
        
        self.logger.info("Setup complete!")
    
    def _save_episode_data(self, episode_data: EpisodeData, step: int):
        """Save episode data to JSON file for logging and analysis."""
        self.episode_counter += 1
        filename = f"step{step:05d}_ep{self.episode_counter:06d}_{episode_data.game_name}.json"
        filepath = self.episode_log_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(episode_data.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save episode data: {e}")

    def _save_step_summary(self, step: int, episodes: List[EpisodeData], stats: Dict):
        """Save summary of all episodes in a step."""
        summary = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_episodes": len(episodes),
            "stats": stats,
            "episodes_summary": [
                {
                    "task_id": ep.task_id,
                    "game_name": ep.game_name,
                    "seed": ep.seed,
                    "opponent": ep.opponent,
                    "final_reward": ep.final_reward,
                    "total_turns": ep.total_turns,
                    "valid_turns": ep.valid_turns,
                    "invalid_turns": ep.invalid_turns,
                    "valid_rate": ep.valid_turns / max(1, ep.total_turns),
                }
                for ep in episodes
            ],
            "aggregated": {
                "total_valid_turns": sum(ep.valid_turns for ep in episodes),
                "total_invalid_turns": sum(ep.invalid_turns for ep in episodes),
                "avg_valid_rate": sum(ep.valid_turns / max(1, ep.total_turns) for ep in episodes) / max(1, len(episodes)),
                "games_played": {game: sum(1 for ep in episodes if ep.game_name == game) for game in set(ep.game_name for ep in episodes)},
            }
        }
        
        filepath = self.episode_log_dir / f"step{step:05d}_summary.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save step summary: {e}")

    async def collect_rollouts(self, batch_size: int, step: int = 0) -> Dict:
        """Collect rollouts from environment, ensuring at least 1 valid sample per game type"""
        rollouts = {
            "queries": [],
            "responses": [],
            "rewards": [],
            "task_ids": [],
            "metadata": [],
        }
        episode_data_list: List[EpisodeData] = []
        
        # Track which games have valid samples
        games_with_valid_samples: Dict[str, int] = {}
        
        # Get list of available games from curriculum sampler
        available_games = list(self.curriculum_sampler.task_pool.keys()) if hasattr(self.curriculum_sampler, 'task_pool') else []
        if not available_games:
            # Fallback: use known game list
            available_games = ["goofspiel", "liars_dice", "leduc_poker", "gin_rummy", 
                            "othello", "backgammon", "hex", "clobber"]
        
        # Maximum attempts to prevent infinite loop
        max_total_attempts = batch_size * 10
        total_attempts = 0
        
        # Keep collecting until we have at least 1 valid sample per game OR hit max attempts
        while total_attempts < max_total_attempts:
            total_attempts += 1
            
            # Determine which game to sample from
            # Prioritize games without valid samples yet
            games_needing_samples = [g for g in available_games if games_with_valid_samples.get(g, 0) == 0]
            
            if games_needing_samples:
                # Force sample from a game that needs valid samples
                target_game = games_needing_samples[total_attempts % len(games_needing_samples)]
                task_config = self.curriculum_sampler.sample_task_for_game(target_game) if hasattr(self.curriculum_sampler, 'sample_task_for_game') else self.curriculum_sampler.sample_task()
            else:
                # All games have at least 1 valid sample, check if we have enough total
                if len(rollouts["queries"]) >= batch_size:
                    break
                task_config = self.curriculum_sampler.sample_task()
            
            task_id = task_config.task_id if hasattr(task_config, "task_id") else task_config["task_id"]

            if self._execution == "local_openspiel":
                opponent = getattr(task_config, "opponent", None) if not isinstance(task_config, dict) else task_config.get("opponent")
                opponent = opponent or str(self.env_config.get("opponent", "mcts"))
                seed = getattr(task_config, "seed", None) if not isinstance(task_config, dict) else task_config.get("seed")
                seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())

                turn_samples, episode_info, episode_data = run_local_openspiel_episode(
                    model=self.model.pretrained_model if hasattr(self.model, "pretrained_model") else self.model,
                    tokenizer=self.tokenizer,
                    task_id=int(task_id),
                    seed=seed,
                    opponent=opponent,
                    temperature=float(self.ppo_config.temperature),
                    top_p=float(self.ppo_config.top_p),
                    max_new_tokens=int(self.ppo_config.max_new_tokens),
                    max_seq_length=int(self.ppo_config.max_seq_length),
                    gamma=float(self.env_config.get("gamma", 0.99)),
                    device=str(self.device),
                )

                # Save episode data for logging
                episode_data_list.append(episode_data)
                self._save_episode_data(episode_data, step)

                # Use final_reward as score proxy for curriculum progression
                score = float(episode_info.get("final_reward", 0.0))
                success = bool(score > 0.5)
                game_name = episode_info.get("game_name", "unknown")
                self.curriculum_sampler.update(task_id=task_id, score=score, success=success)

                # Track valid samples per game
                valid_turns_this_episode = episode_info.get("valid_turns", 0)
                if valid_turns_this_episode > 0:
                    games_with_valid_samples[game_name] = games_with_valid_samples.get(game_name, 0) + valid_turns_this_episode

                # Add each LLM decision turn as a PPO sample
                for ts in turn_samples:
                    q = self.tokenizer.encode(
                        ts.prompt_text,
                        return_tensors="pt",
                        max_length=max(32, self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens),
                        truncation=True,
                    ).to(self.device)

                    # Encode response tokens (tiny)
                    r = self.tokenizer.encode(ts.response_text, return_tensors="pt", truncation=True).to(self.device)

                    # PPO trainer expects 1D tensors (squeeze batch dimension)
                    rollouts["queries"].append(q.squeeze(0))
                    rollouts["responses"].append(r.squeeze(0))
                    rollouts["rewards"].append(torch.tensor(ts.reward))
                    rollouts["task_ids"].append(task_id)
                    rollouts["metadata"].append(
                        {
                            "score": score,
                            "success": success,
                            "task_config": task_config,
                            "episode": {k: episode_info.get(k) for k in ("game_name", "num_turns", "final_reward", "opponent", "valid_turns", "invalid_turns")},
                        }
                    )
        
        # Log collection summary
        self.logger.info(f"Collected {len(rollouts['queries'])} valid samples from {total_attempts} episodes. "
                        f"Games with valid samples: {games_with_valid_samples}")
            else:
                # Legacy server-based flow (docker/basilica env server + HTTP LLM)
                state_dict = await self.env_wrapper.reset(task_id=task_id)

                query_tensors = self.tokenizer.encode(
                    state_dict["prompt"],
                    return_tensors="pt",
                    max_length=max(32, self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens),
                    truncation=True,
                ).to(self.device)

                with torch.no_grad():
                    response_tensors = self.ppo_trainer.generate(
                        query_tensors,
                        max_new_tokens=self.ppo_config.max_new_tokens,
                        temperature=self.ppo_config.temperature,
                        top_p=self.ppo_config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                response_text = self.tokenizer.decode(
                    response_tensors[0][len(query_tensors[0]):],
                    skip_special_tokens=True
                )
                import re
                m = re.search(r"\d+", response_text)
                response_text = m.group(0) if m else "INVALID"

                result = await self.env_wrapper.step(action=response_text, task_id=task_id, task_config=task_config)
                self.curriculum_sampler.update(task_id=task_id, score=result["score"], success=result["success"])

                rollouts["queries"].append(query_tensors)
                rollouts["responses"].append(response_tensors[0][len(query_tensors[0]):])
                rollouts["rewards"].append(torch.tensor(result["reward"]))
                rollouts["task_ids"].append(task_id)
                rollouts["metadata"].append({"score": result["score"], "success": result["success"], "task_config": task_config})
        
        # Store episode data list in rollouts for step summary
        rollouts["episode_data_list"] = episode_data_list
        
        return rollouts
    
    def train_step(self, rollouts: Dict) -> Dict:
        """Single PPO training step"""
        # Prepare data for PPO
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
        
        # Aggregate stats from all batches (only scalar values)
        if all_stats:
            stats = {}
            for k in all_stats[0].keys():
                values = [s.get(k, 0) for s in all_stats]
                # Only aggregate scalar values
                if isinstance(values[0], (int, float)):
                    stats[k] = sum(values) / len(values)
                elif hasattr(values[0], 'item'):
                    try:
                        stats[k] = sum(v.item() if hasattr(v, 'item') else v for v in values) / len(values)
                    except:
                        pass  # Skip non-scalar tensors
        else:
            stats = {}
        
        # Add custom metrics (handle empty rewards case)
        if rewards:
            stats["mean_reward"] = torch.mean(torch.stack(rewards)).item()
        else:
            stats["mean_reward"] = 0.0
        
        if rollouts["metadata"]:
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
        self.logger.info("Starting training...")
        
        global_step = 0
        
        for step in range(self.ppo_config.num_train_steps):
            # Collect rollouts - collect many episodes to ensure enough VALID samples
            # Only valid responses are used for training (to avoid KL divergence issues)
            # Untrained models have low valid response rate, so we need many episodes
            num_episodes = max(16, self.ppo_config.batch_size * 4)  # Collect 16+ episodes per step
            self.logger.info(f"Step {step + 1}/{self.ppo_config.num_train_steps}: Collecting rollouts from {num_episodes} episodes...")
            rollouts = await self.collect_rollouts(num_episodes, step=step + 1)
            
            # Get episode data for logging
            episode_data_list = rollouts.pop("episode_data_list", [])
            
            # Check if we have enough samples for training
            num_samples = len(rollouts["queries"])
            if num_samples < self.ppo_config.batch_size:
                self.logger.warning(f"Step {step + 1}: Only {num_samples} valid samples collected (need {self.ppo_config.batch_size}). Skipping training step.")
                # Still save step summary even if skipping
                if episode_data_list:
                    self._save_step_summary(step + 1, episode_data_list, {"skipped": True, "num_samples": num_samples})
                continue
            
            # Train
            self.logger.info(f"Training with {num_samples} samples...")
            stats = self.train_step(rollouts)
            
            # Add episode statistics to stats
            if episode_data_list:
                stats["total_valid_turns"] = sum(ep.valid_turns for ep in episode_data_list)
                stats["total_invalid_turns"] = sum(ep.invalid_turns for ep in episode_data_list)
                stats["avg_valid_rate"] = sum(ep.valid_turns / max(1, ep.total_turns) for ep in episode_data_list) / max(1, len(episode_data_list))
            
            # Save step summary with episode data
            if episode_data_list:
                self._save_step_summary(step + 1, episode_data_list, stats)
            
            # Log
            if (step + 1) % self.ppo_config.log_freq == 0:
                valid_rate = stats.get('avg_valid_rate', 0)
                self.logger.info(
                    f"Step {step + 1} | "
                    f"Reward: {stats['mean_reward']:.4f} | "
                    f"Score: {stats['mean_score']:.4f} | "
                    f"Success: {stats['success_rate']:.2%} | "
                    f"ValidRate: {valid_rate:.2%}"
                )
                
                if self.ppo_config.use_wandb:
                    wandb.log(stats, step=step + 1)
            
            # Save checkpoint
            if (step + 1) % self.ppo_config.save_freq == 0:
                save_path = Path(self.ppo_config.output_dir) / f"checkpoint-{step + 1}"
                self.logger.info(f"Saving checkpoint to {save_path}")
                save_checkpoint(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    save_path=save_path,
                    curriculum_state=self.curriculum_sampler.get_state(),
                )
            
            # Evaluate
            if (step + 1) % self.ppo_config.eval_freq == 0:
                self.logger.info("Running evaluation...")
                eval_stats = await self.evaluate()
                self.logger.info(f"Eval | Score: {eval_stats['mean_score']:.4f}")
                
                if self.ppo_config.use_wandb:
                    wandb.log({"eval/" + k: v for k, v in eval_stats.items()}, step=step + 1)
            
            global_step += 1
        
        # Final save
        final_path = Path(self.ppo_config.output_dir) / "final"
        self.logger.info(f"Saving final model to {final_path}")
        save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            save_path=final_path,
            curriculum_state=self.curriculum_sampler.get_state(),
        )
        
        self.logger.info("Training complete!")
    
    async def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        eval_results = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_episodes):
                # Sample random task
                task_config = self.curriculum_sampler.sample_task(eval_mode=True)
                task_id = task_config.task_id if hasattr(task_config, "task_id") else task_config["task_id"]
                
                # Reset environment
                state_dict = await self.env_wrapper.reset(task_id=task_id)
                
                # Generate response
                query_tensors = self.tokenizer.encode(
                    state_dict["prompt"],
                    return_tensors="pt",
                    max_length=self.ppo_config.max_seq_length // 2,
                    truncation=True,
                ).to(self.device)
                
                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=self.ppo_config.max_new_tokens,
                    temperature=0.0,  # Greedy for evaluation
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                response_text = self.tokenizer.decode(
                    response_tensors[0][len(query_tensors[0]):],
                    skip_special_tokens=True
                )
                import re
                m = re.search(r"\d+", response_text)
                response_text = m.group(0) if m else "INVALID"
                
                # Evaluate
                result = await self.env_wrapper.step(
                    action=response_text,
                    task_id=task_id,
                    task_config=task_config
                )
                
                eval_results.append(result)
        
        self.model.train()
        
        # Aggregate stats
        stats = {
            "mean_score": sum(r["score"] for r in eval_results) / len(eval_results),
            "mean_reward": sum(r["reward"] for r in eval_results) / len(eval_results),
            "success_rate": sum(r["success"] for r in eval_results) / len(eval_results),
        }
        
        return stats


def main():
    """Main entry point"""
    # Load config from file if exists
    config_path = Path("config/train_config.json")
    config_dict = _load_json_if_exists(config_path)
    model_config = ModelConfig(**config_dict.get("model", {})) if config_dict else ModelConfig()
    lora_config = LoRAConfig(**config_dict.get("lora", {})) if config_dict else LoRAConfig()
    ppo_config = PPOTrainingConfig(**config_dict.get("ppo", {})) if config_dict else PPOTrainingConfig()

    # Load environment config (LLM endpoint etc.)
    env_config_path = Path("config/env_config.json")
    env_cfg = _load_json_if_exists(env_config_path)
    
    # Create trainer
    trainer = GamePPOTrainer(
        model_config=model_config,
        lora_config=lora_config,
        ppo_config=ppo_config,
        env_name="game",
        env_mode=str(env_cfg.get("env_mode", "basilica")) if env_cfg else "basilica",  # or "docker"
        env_config=env_cfg,
    )
    
    # Train
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
