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
from game_rl_training.utils import setup_logging, save_checkpoint, load_checkpoint
from game_rl_training.local_openspiel import (
    run_local_openspiel_episode, 
    run_vllm_openspiel_episode,
    run_parallel_episodes,
    run_parallel_episodes_cpu,
    EpisodeData,
)

def _load_json_if_exists(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen3-4B-Instruct"  # Using Qwen2.5-3B as Qwen3-4B might not be released
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
    resume_from: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "game-rl-training"
    wandb_run_name: Optional[str] = None
    
    # Invalid sample handling
    include_invalid_samples: bool = True  # Include invalid responses with penalty
    invalid_penalty: float = -0.5  # Penalty for invalid responses
    
    # SFT warmup
    sft_warmup_steps: int = 0  # Number of SFT warmup steps before PPO


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
            # Use "abs" KL penalty to avoid negative KL issues
            kl_penalty="abs",
            # Whiten rewards for more stable training
            whiten_rewards=True,
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
        # Game indices from `affinetes/environments/openspiel/game_config.py::AVAILABLE_GAMES`:
        # 0 goofspiel (7.8k tokens), 1 liars_dice (1.1k), 2 leduc_poker (1.3k), 3 gin_rummy (167k - SLOW)
        # 4 othello (105k - SLOW), 5 backgammon (347k - VERY SLOW), 6 hex (13.9k), 7 clobber (16.9k)
        # 
        # For faster training, focus on quick games first:
        # Fast: liars_dice, leduc_poker, goofspiel, hex, clobber
        # Slow: gin_rummy, othello, backgammon (skip initially)
        fast_game_indices = [0, 1, 2, 6, 7]  # goofspiel, liars_dice, leduc_poker, hex, clobber
        allowed_game_indices = self.env_config.get("allowed_game_indices", fast_game_indices)
        max_task_id = (max(allowed_game_indices) + 1) * 100_000_000 - 1
        # Get opponent from env_config, default to "mcts"
        default_opponent = str(self.env_config.get("opponent", "mcts"))
        self.curriculum_sampler = CurriculumSampler(
            env_name=self.env_name,
            failure_buffer_size=1000,
            curriculum_stages=[
                # Stage ranges kept for readability; actual sampling is constrained by allowed_game_indices.
                # Opponent is now configurable via env_config.json
                {"name": "easy", "task_range": (0, max_task_id), "opponent": default_opponent},
                {"name": "medium", "task_range": (0, max_task_id), "opponent": default_opponent},
                {"name": "hard", "task_range": (0, max_task_id), "opponent": default_opponent},
            ]
            ,allowed_game_indices=allowed_game_indices
        )
        
        # Setup episode logging directories (separate for success/failure)
        self.episode_log_dir = Path(self.ppo_config.output_dir) / "episode_logs"
        self.episode_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate directories for successful and unsuccessful episodes
        self.success_log_dir = Path(self.ppo_config.output_dir) / "episode_logs_success"
        self.failure_log_dir = Path(self.ppo_config.output_dir) / "episode_logs_failure"
        self.success_log_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_counter = 0
        
        # Track resume state
        self.start_step = 0
        self.resumed_from = None
        
        # Resume from checkpoint if specified
        if self.ppo_config.resume_from:
            self._resume_from_checkpoint(self.ppo_config.resume_from)
        
        self.logger.info("Setup complete!")
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint metadata and model weights
        try:
            metadata = load_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                load_path=checkpoint_path,
            )
            
            # Restore curriculum state if available
            if "curriculum_state" in metadata:
                self.curriculum_sampler.load_state(metadata["curriculum_state"])
                self.logger.info("Restored curriculum sampler state")
            
            # Extract step number from checkpoint name (e.g., "checkpoint-500" -> 500)
            checkpoint_name = checkpoint_path.name
            if checkpoint_name.startswith("checkpoint-"):
                try:
                    self.start_step = int(checkpoint_name.split("-")[1])
                    self.logger.info(f"Resuming from step {self.start_step}")
                except (IndexError, ValueError):
                    self.logger.warning(f"Could not parse step from checkpoint name: {checkpoint_name}")
            elif checkpoint_name == "final":
                # If resuming from final checkpoint, we need to read the step from metadata
                self.start_step = metadata.get("training_args", {}).get("step", 0)
                self.logger.info(f"Resuming from final checkpoint at step {self.start_step}")
            
            # Update episode counter based on existing logs
            existing_logs = list(self.episode_log_dir.glob("step*_ep*.json"))
            if existing_logs:
                # Extract max episode number from existing logs
                max_ep = 0
                for log_file in existing_logs:
                    try:
                        ep_num = int(log_file.stem.split("_ep")[1].split("_")[0])
                        max_ep = max(max_ep, ep_num)
                    except (IndexError, ValueError):
                        pass
                self.episode_counter = max_ep
                self.logger.info(f"Resuming episode counter from {self.episode_counter}")
            
            self.resumed_from = str(checkpoint_path)
            self.logger.info(f"Successfully resumed from checkpoint: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _save_episode_data(self, episode_data: EpisodeData, step: int):
        """Save episode data to JSON file for logging and analysis.
        
        Episodes are saved to separate directories based on success:
        - episode_logs_success/: Episodes where final_reward > 0.5 (win)
        - episode_logs_failure/: Episodes where final_reward <= 0.5 (loss/draw)
        - episode_logs/: All episodes (for backward compatibility)
        """
        self.episode_counter += 1
        filename = f"step{step:05d}_ep{self.episode_counter:06d}_{episode_data.game_name}.json"
        
        # Determine if this episode was successful (win = reward > 0.5)
        is_success = episode_data.final_reward > 0.5
        
        # Choose directory based on success
        if is_success:
            target_dir = self.success_log_dir
        else:
            target_dir = self.failure_log_dir
        
        filepath = target_dir / filename
        
        try:
            episode_dict = episode_data.to_dict()
            # Add success flag to the data
            episode_dict["is_success"] = is_success
            
            with open(filepath, 'w') as f:
                json.dump(episode_dict, f, indent=2, default=str)
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
                "success_count": sum(1 for ep in episodes if ep.final_reward > 0.5),
                "failure_count": sum(1 for ep in episodes if ep.final_reward <= 0.5),
                "success_rate": sum(1 for ep in episodes if ep.final_reward > 0.5) / max(1, len(episodes)),
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
        
        # Check if we should use vLLM for parallel collection
        use_vllm = self.env_config.get("use_vllm", False)
        vllm_base_url = str(self.env_config.get("vllm_base_url", "http://localhost:8000"))
        max_concurrent = int(self.env_config.get("max_concurrent_episodes", 8))
        
        if use_vllm and self._execution == "local_openspiel":
            return await self._collect_rollouts_vllm(batch_size, step, vllm_base_url, max_concurrent)
        
        # Check if we should use parallel CPU workers for episode collection
        num_workers = int(self.env_config.get("num_episode_workers", 1))
        if num_workers > 1 and self._execution == "local_openspiel":
            return await self._collect_rollouts_parallel(batch_size, step, num_workers)
        
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
                
                # Get MCTS config (higher = stronger opponent, slower training)
                mcts_simulations = int(self.env_config.get("mcts_simulations", 100))
                mcts_rollouts = self.env_config.get("mcts_rollouts", None)
                if mcts_rollouts is not None:
                    mcts_rollouts = int(mcts_rollouts)
                mcts_workers = self.env_config.get("mcts_workers", None)
                if mcts_workers is not None:
                    mcts_workers = int(mcts_workers)

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
                    include_invalid=bool(self.ppo_config.include_invalid_samples),
                    invalid_penalty=float(self.ppo_config.invalid_penalty),
                    mcts_simulations=mcts_simulations,
                    mcts_rollouts=mcts_rollouts,
                    mcts_workers=mcts_workers,
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
                    # Use pre-computed token IDs if available (avoids re-encoding issues)
                    if ts.query_ids is not None and ts.response_ids is not None:
                        q = torch.tensor(ts.query_ids, dtype=torch.long, device=self.device)
                        r = torch.tensor(ts.response_ids, dtype=torch.long, device=self.device)
                    else:
                        # Fallback: re-encode (may cause KL divergence issues)
                        q = self.tokenizer.encode(
                            ts.prompt_text,
                            return_tensors="pt",
                            max_length=max(32, self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens),
                            truncation=True,
                            add_special_tokens=False,  # Don't add special tokens
                        ).to(self.device).squeeze(0)

                        r = self.tokenizer.encode(
                            ts.response_text, 
                            return_tensors="pt", 
                            truncation=True,
                            add_special_tokens=False,  # Don't add special tokens
                        ).to(self.device).squeeze(0)

                    # PPO trainer expects 1D tensors
                    rollouts["queries"].append(q)
                    rollouts["responses"].append(r)
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
        
        # Log collection summary
        self.logger.info(f"Collected {len(rollouts['queries'])} valid samples from {total_attempts} episodes. "
                        f"Games with valid samples: {games_with_valid_samples}")
        
        # Store episode data list in rollouts for step summary
        rollouts["episode_data_list"] = episode_data_list
        
        return rollouts
    
    async def _collect_rollouts_vllm(self, batch_size: int, step: int, vllm_base_url: str, max_concurrent: int) -> Dict:
        """Collect rollouts using vLLM API for parallel episode collection (FAST)"""
        rollouts = {
            "queries": [],
            "responses": [],
            "rewards": [],
            "task_ids": [],
            "metadata": [],
        }
        episode_data_list: List[EpisodeData] = []
        games_with_valid_samples: Dict[str, int] = {}
        
        # Get available games
        available_games = list(self.curriculum_sampler.task_pool.keys()) if hasattr(self.curriculum_sampler, 'task_pool') else []
        if not available_games:
            available_games = ["goofspiel", "liars_dice", "leduc_poker", "hex", "clobber"]
        
        opponent = str(self.env_config.get("opponent", "random"))
        model_name = self.model_config.model_name
        
        # Collect episodes in batches until we have enough samples
        max_rounds = 10
        round_num = 0
        
        while len(rollouts["queries"]) < batch_size and round_num < max_rounds:
            round_num += 1
            
            # Generate task configs for parallel episodes
            num_episodes = max(max_concurrent, batch_size // 2)
            task_configs = []
            
            for i in range(num_episodes):
                # Prioritize games without samples
                games_needing = [g for g in available_games if games_with_valid_samples.get(g, 0) == 0]
                if games_needing:
                    target_game = games_needing[i % len(games_needing)]
                    task_config = self.curriculum_sampler.sample_task_for_game(target_game) if hasattr(self.curriculum_sampler, 'sample_task_for_game') else self.curriculum_sampler.sample_task()
                else:
                    task_config = self.curriculum_sampler.sample_task()
                
                task_id = task_config.task_id if hasattr(task_config, "task_id") else task_config["task_id"]
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                
                task_configs.append({
                    "task_id": int(task_id),
                    "seed": seed,
                    "task_config": task_config,
                })
            
            self.logger.info(f"Round {round_num}: Running {len(task_configs)} parallel episodes via vLLM...")
            
            # Run episodes in parallel
            results = await run_parallel_episodes(
                tokenizer=self.tokenizer,
                task_configs=task_configs,
                vllm_base_url=vllm_base_url,
                model_name=model_name,
                opponent=opponent,
                temperature=float(self.ppo_config.temperature),
                top_p=float(self.ppo_config.top_p),
                max_new_tokens=int(self.ppo_config.max_new_tokens),
                max_seq_length=int(self.ppo_config.max_seq_length),
                gamma=float(self.env_config.get("gamma", 0.99)),
                max_concurrent=max_concurrent,
            )
            
            # Process results
            for idx, (turn_samples, episode_info, episode_data) in enumerate(results):
                task_config = task_configs[idx]["task_config"]
                task_id = task_configs[idx]["task_id"]
                
                # Save episode data
                episode_data_list.append(episode_data)
                self._save_episode_data(episode_data, step)
                
                # Update curriculum
                score = float(episode_info.get("final_reward", 0.0))
                success = bool(score > 0.5)
                game_name = episode_info.get("game_name", "unknown")
                self.curriculum_sampler.update(task_id=task_id, score=score, success=success)
                
                # Track valid samples
                valid_turns = episode_info.get("valid_turns", 0)
                if valid_turns > 0:
                    games_with_valid_samples[game_name] = games_with_valid_samples.get(game_name, 0) + valid_turns
                
                # Add turn samples to rollouts
                for ts in turn_samples:
                    q = self.tokenizer.encode(
                        ts.prompt_text,
                        return_tensors="pt",
                        max_length=max(32, self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens),
                        truncation=True,
                    ).to(self.device)
                    r = self.tokenizer.encode(ts.response_text, return_tensors="pt", truncation=True).to(self.device)
                    
                    rollouts["queries"].append(q.squeeze(0))
                    rollouts["responses"].append(r.squeeze(0))
                    rollouts["rewards"].append(torch.tensor(ts.reward))
                    rollouts["task_ids"].append(task_id)
                    rollouts["metadata"].append({
                        "score": score,
                        "success": success,
                        "task_config": task_config,
                        "episode": {k: episode_info.get(k) for k in ("game_name", "num_turns", "final_reward", "opponent", "valid_turns", "invalid_turns")},
                    })
            
            self.logger.info(f"Round {round_num}: Collected {len(rollouts['queries'])} total samples so far")
        
        self.logger.info(f"vLLM collection complete: {len(rollouts['queries'])} samples from {len(episode_data_list)} episodes. "
                        f"Games: {games_with_valid_samples}")
        
        rollouts["episode_data_list"] = episode_data_list
        return rollouts
    
    async def _collect_rollouts_parallel(self, batch_size: int, step: int, num_workers: int) -> Dict:
        """Collect rollouts using parallel workers for faster MCTS episode collection.
        
        Uses ThreadPoolExecutor to run multiple episodes concurrently.
        MCTS C++ code releases the GIL, allowing true parallelism for opponent moves.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        rollouts = {
            "queries": [],
            "responses": [],
            "rewards": [],
            "task_ids": [],
            "metadata": [],
        }
        episode_data_list: List[EpisodeData] = []
        games_with_valid_samples: Dict[str, int] = {}
        
        # Get config
        opponent = str(self.env_config.get("opponent", "mcts"))
        mcts_simulations = int(self.env_config.get("mcts_simulations", 100))
        mcts_rollouts = self.env_config.get("mcts_rollouts", None)
        if mcts_rollouts is not None:
            mcts_rollouts = int(mcts_rollouts)
        mcts_workers = self.env_config.get("mcts_workers", None)
        if mcts_workers is not None:
            mcts_workers = int(mcts_workers)
        
        def run_single_episode(task_config):
            """Run a single episode - model inference runs without lock for parallelism."""
            task_id = task_config.task_id if hasattr(task_config, "task_id") else task_config["task_id"]
            seed = task_config.seed if hasattr(task_config, "seed") else int(torch.randint(0, 2**31 - 1, (1,)).item())
            ep_opponent = getattr(task_config, "opponent", None) or opponent
            
            # Run episode - PyTorch model.generate() is thread-safe for inference
            turn_samples, episode_info, episode_data = run_local_openspiel_episode(
                    model=self.model.pretrained_model if hasattr(self.model, "pretrained_model") else self.model,
                    tokenizer=self.tokenizer,
                    task_id=int(task_id),
                    seed=seed,
                    opponent=ep_opponent,
                    temperature=float(self.ppo_config.temperature),
                    top_p=float(self.ppo_config.top_p),
                    max_new_tokens=int(self.ppo_config.max_new_tokens),
                    max_seq_length=int(self.ppo_config.max_seq_length),
                    gamma=float(self.env_config.get("gamma", 0.99)),
                    device=str(self.device),
                    include_invalid=bool(self.ppo_config.include_invalid_samples),
                    invalid_penalty=float(self.ppo_config.invalid_penalty),
                    mcts_simulations=mcts_simulations,
                    mcts_rollouts=mcts_rollouts,
                    mcts_workers=mcts_workers,
                )
            
            return task_id, task_config, turn_samples, episode_info, episode_data
        
        # Generate task configs
        num_episodes_to_collect = max(batch_size * 3, 16)  # Collect more to ensure enough valid samples
        task_configs = []
        for _ in range(num_episodes_to_collect):
            task_config = self.curriculum_sampler.sample_task()
            task_configs.append(task_config)
        
        # Run episodes in parallel
        self.logger.info(f"Running {num_episodes_to_collect} episodes with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_single_episode, cfg): cfg for cfg in task_configs}
            
            for future in as_completed(futures):
                try:
                    task_id, task_config, turn_samples, episode_info, episode_data = future.result()
                    
                    # Save episode data
                    episode_data_list.append(episode_data)
                    self._save_episode_data(episode_data, step)
                    
                    # Update curriculum
                    score = float(episode_info.get("final_reward", 0.0))
                    success = bool(score > 0.5)
                    game_name = episode_info.get("game_name", "unknown")
                    self.curriculum_sampler.update(task_id=task_id, score=score, success=success)
                    
                    # Track valid samples
                    valid_turns = episode_info.get("valid_turns", 0)
                    if valid_turns > 0:
                        games_with_valid_samples[game_name] = games_with_valid_samples.get(game_name, 0) + valid_turns
                    
                    # Add turn samples to rollouts
                    for ts in turn_samples:
                        if ts.query_ids is not None and ts.response_ids is not None:
                            q = torch.tensor(ts.query_ids, dtype=torch.long, device=self.device)
                            r = torch.tensor(ts.response_ids, dtype=torch.long, device=self.device)
                        else:
                            q = self.tokenizer.encode(
                                ts.prompt_text,
                                return_tensors="pt",
                                max_length=max(32, self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens),
                                truncation=True,
                                add_special_tokens=False,
                            ).to(self.device).squeeze(0)
                            r = self.tokenizer.encode(
                                ts.response_text,
                                return_tensors="pt",
                                truncation=True,
                                add_special_tokens=False,
                            ).to(self.device).squeeze(0)
                        
                        rollouts["queries"].append(q)
                        rollouts["responses"].append(r)
                        rollouts["rewards"].append(torch.tensor(ts.reward))
                        rollouts["task_ids"].append(task_id)
                        rollouts["metadata"].append({
                            "score": score,
                            "success": success,
                            "task_config": task_config,
                            "episode": {k: episode_info.get(k) for k in ("game_name", "num_turns", "final_reward", "opponent", "valid_turns", "invalid_turns")},
                        })
                    
                    # Check if we have enough samples
                    if len(rollouts["queries"]) >= batch_size:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Episode failed: {e}")
        
        self.logger.info(f"Parallel collection complete: {len(rollouts['queries'])} samples from {len(episode_data_list)} episodes. "
                        f"Games: {games_with_valid_samples}")
        
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
    
    def sft_warmup(self, num_steps: int = 50):
        """
        Run SFT warmup to teach the model the output format.
        Creates synthetic examples with correct action ID format.
        """
        if num_steps <= 0:
            return
        
        self.logger.info(f"Running SFT warmup for {num_steps} steps...")
        
        from torch.optim import AdamW
        
        # Get trainable parameters (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.ppo_config.learning_rate * 5)  # Higher LR for SFT
        
        # Import game components
        try:
            from game_rl_training.game_config import create_game, AVAILABLE_GAMES
            from game_rl_training.agents import GAME_AGENTS
        except ImportError as e:
            self.logger.warning(f"Could not import game components for SFT: {e}")
            return
        
        # Get available games from config
        allowed_indices = self.env_config.get("allowed_game_indices", [0, 1, 2])
        
        self.model.train()
        total_loss = 0.0
        
        for step in range(num_steps):
            # Generate synthetic training example
            game_idx = allowed_indices[step % len(allowed_indices)]
            task_id = game_idx * 100_000_000 + (step * 7919) % 1000  # Varied config
            
            try:
                game, game_cfg = create_game(task_id)
                game_name = game_cfg["game_name"]
                agent_class = GAME_AGENTS.get(game_name)
                if not agent_class:
                    continue
                agent = agent_class()
                
                # Create a game state and get legal actions
                state = game.new_initial_state()
                
                # Apply some random actions to get to a non-trivial state
                import numpy as np
                rng = np.random.RandomState(step)
                for _ in range(rng.randint(1, 5)):
                    if state.is_terminal():
                        break
                    cur_player = state.current_player()
                    if cur_player < 0:  # Chance node
                        outcomes = state.chance_outcomes()
                        actions, probs = zip(*outcomes)
                        state.apply_action(rng.choice(actions, p=probs))
                    else:
                        legal = state.legal_actions(cur_player)
                        if legal:
                            state.apply_action(rng.choice(legal))
                
                if state.is_terminal():
                    continue
                
                # Get legal actions for current player
                player_id = state.current_player()
                if player_id < 0:
                    continue
                legal_actions = state.legal_actions(player_id)
                if not legal_actions:
                    continue
                
                # Create prompt and target
                system_prompt = agent.generate_system_prompt()
                user_prompt = agent.generate_user_prompt(state=state, player_id=player_id, legal_actions=legal_actions)
                
                # Target is just a valid action ID
                target_action = str(rng.choice(legal_actions))
                
                # Build chat format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                full_text = prompt_text + target_action
                
                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.ppo_config.max_seq_length,
                ).to(self.device)
                
                # Create labels (mask prompt, only compute loss on target)
                prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
                labels = inputs["input_ids"].clone()
                labels[0, :prompt_len] = -100  # Mask prompt tokens
                
                # Forward pass - use the pretrained_model (base LM) for SFT
                # AutoModelForCausalLMWithValueHead wraps the base model
                base_model = self.model.pretrained_model if hasattr(self.model, "pretrained_model") else self.model
                outputs = base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels,
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Ensure loss is a scalar
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / (step + 1)
                    self.logger.info(f"SFT step {step + 1}/{num_steps} | Loss: {avg_loss:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"SFT step {step} failed: {e}")
                continue
        
        self.logger.info(f"SFT warmup complete. Avg loss: {total_loss / max(1, num_steps):.4f}")
    
    async def train(self):
        """Main training loop"""
        # Run SFT warmup if configured and not resuming
        if not self.resumed_from and self.ppo_config.sft_warmup_steps > 0:
            self.sft_warmup(self.ppo_config.sft_warmup_steps)
        
        if self.resumed_from:
            self.logger.info(f"Resuming training from step {self.start_step}...")
        else:
            self.logger.info("Starting PPO training...")
        
        global_step = self.start_step
        
        for step in range(self.start_step, self.ppo_config.num_train_steps):
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
                    training_args={"step": step + 1, "global_step": global_step + 1},
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
            training_args={"step": self.ppo_config.num_train_steps, "global_step": global_step},
        )
        
        self.logger.info("Training complete!")
    
    async def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        eval_results = []
        
        # Check if using local OpenSpiel execution (no env_wrapper)
        if self._execution == "local_openspiel":
            # Use local OpenSpiel for evaluation
            self.model.eval()
            for _ in range(num_episodes):
                task_config = self.curriculum_sampler.sample_task(eval_mode=True)
                task_id = task_config.task_id if hasattr(task_config, "task_id") else task_config["task_id"]
                seed = task_config.seed if hasattr(task_config, "seed") else task_config.get("seed", 42)
                opponent = task_config.opponent if hasattr(task_config, "opponent") else task_config.get("opponent", "random")
                
                try:
                    mcts_simulations = int(self.env_config.get("mcts_simulations", 100))
                    mcts_rollouts = self.env_config.get("mcts_rollouts", None)
                    if mcts_rollouts is not None:
                        mcts_rollouts = int(mcts_rollouts)
                    mcts_workers = self.env_config.get("mcts_workers", None)
                    if mcts_workers is not None:
                        mcts_workers = int(mcts_workers)
                    turn_samples, episode_info, _ = run_local_openspiel_episode(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        task_id=task_id,
                        seed=seed,
                        opponent=opponent,
                        temperature=0.1,  # Low temperature for more deterministic eval
                        top_p=0.95,
                        max_new_tokens=self.ppo_config.max_new_tokens,
                        max_seq_length=self.ppo_config.max_seq_length,
                        gamma=self.env_config.get("gamma", 0.99),
                        device=self.device,
                        mcts_simulations=mcts_simulations,
                        mcts_rollouts=mcts_rollouts,
                        mcts_workers=mcts_workers,
                    )
                    eval_results.append({
                        "score": episode_info.get("final_reward", 0.0),
                        "reward": episode_info.get("final_reward", 0.0),
                        "success": episode_info.get("final_reward", 0.0) > 0.5,
                    })
                except Exception as e:
                    self.logger.warning(f"Eval episode failed: {e}")
                    eval_results.append({"score": 0.0, "reward": 0.0, "success": False})
            
            self.model.train()
        else:
            # Use env_wrapper for evaluation
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
        if not eval_results:
            return {"mean_score": 0.0, "mean_reward": 0.0, "success_rate": 0.0}
        
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
