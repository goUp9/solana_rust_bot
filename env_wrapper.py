#!/usr/bin/env python3
"""
Environment Wrapper for RL Training
Wraps the game environment to provide a standard RL interface
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents the current game state"""
    prompt: str
    task_id: int
    step: int
    done: bool
    info: Dict[str, Any]


class GameEnvironmentWrapper:
    """
    Wrapper around the game environment for RL training
    
    This wrapper:
    1. Converts environment observations to text prompts
    2. Converts agent actions (text) to environment format
    3. Computes rewards based on game outcomes
    4. Handles task-ID awareness and episode management
    """
    
    def __init__(
        self,
        env,
        tokenizer,
        max_length: int = 2048,
        reward_shaping: bool = True,
        llm_base_url: str = "http://127.0.0.1:8000/v1",
        llm_model: str = "Qwen/Qwen3-4B",
        llm_temperature: float = 0.8,
        llm_timeout: int = 7200,
        llm_api_key: Optional[str] = None,
    ):
        """
        Initialize environment wrapper
        
        Args:
            env: The game environment instance (from create_environment)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            reward_shaping: Whether to apply reward shaping
        """
        self.env = env
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reward_shaping = reward_shaping

        # LLM endpoint used by the GAME/OpenSpiel environment actor (OpenAI-compatible)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_timeout = llm_timeout
        self.llm_api_key = llm_api_key
        
        self.current_task_id = None
        self.current_step = 0
        self.episode_history = []
    
    async def reset(self, task_id: int) -> Dict[str, Any]:
        """
        Reset environment for a new episode
        
        Args:
            task_id: Task ID to run (determines game configuration)
        
        Returns:
            Dictionary with initial state information:
            - prompt: Text prompt for the model
            - task_id: Current task ID
            - info: Additional metadata
        """
        self.current_task_id = task_id
        self.current_step = 0
        self.episode_history = []
        
        # The game environment doesn't have a traditional "reset" that returns state
        # Instead, we construct the initial prompt based on the task
        initial_prompt = self._create_initial_prompt(task_id)
        
        return {
            "prompt": initial_prompt,
            "task_id": task_id,
            "info": {
                "step": 0,
                "game_config": self._parse_task_id(task_id),
            }
        }
    
    async def step(
        self,
        action: str,
        task_id: int,
        task_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute one step in the environment
        
        Args:
            action: Model's action (text response)
            task_id: Current task ID
            task_config: Additional task configuration (from curriculum sampler)
        
        Returns:
            Dictionary with step results:
            - score: Game score (0.0-1.0)
            - reward: Shaped reward for RL
            - success: Whether the episode was successful
            - done: Whether the episode is done
            - info: Additional metadata
        """
        # In the game environment, each evaluation is a full episode
        # So we run the complete game and get the result
        
        # Prepare evaluation parameters
        def _get(cfg: Any, key: str, default=None):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        eval_kwargs = {
            "task_id": task_id,
            "seed": _get(task_config, "seed", None),
            # IMPORTANT: point the env at local vLLM (OpenAI-compatible)
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "timeout": self.llm_timeout,
            "temperature": self.llm_temperature,
        }
        if self.llm_api_key is not None:
            eval_kwargs["api_key"] = self.llm_api_key
        
        # Add opponent configuration if available
        opponent = _get(task_config, "opponent", None)
        if opponent:
            eval_kwargs["opponent"] = opponent
        
        # The environment expects a miner with model/base_url
        # For training, we need to mock this since we're using the model directly
        # Instead, we'll need to set up a local inference endpoint or use the model directly
        # For now, we'll simulate this by using the environment's evaluation
        
        try:
            # Call environment evaluation
            # Note: This is a simplified version - you may need to set up a local inference server
            # NOTE: For GAME/OpenSpiel, miner=None is OK because Actor.evaluate accepts
            # base_url/model directly (see affinetes/environments/openspiel/env.py).
            result = await self.env.evaluate(miner=None, **eval_kwargs)
            
            # Extract results
            score = result.score if hasattr(result, 'score') else result.get("score", 0.0)
            success = result.success if hasattr(result, 'success') else result.get("success", False)
            extra = result.extra if hasattr(result, 'extra') else result.get("extra", {})
            
            # Compute reward
            reward = self._compute_reward(score, success, extra)
            
            # Update episode history
            self.current_step += 1
            self.episode_history.append({
                "action": action,
                "score": score,
                "reward": reward,
                "success": success,
            })
            
            return {
                "score": score,
                "reward": reward,
                "success": success,
                "done": True,  # Each game episode is atomic
                "info": {
                    "step": self.current_step,
                    "extra": extra,
                },
            }
            
        except Exception as e:
            # Handle errors gracefully
            return {
                "score": 0.0,
                "reward": -1.0,  # Penalty for errors
                "success": False,
                "done": True,
                "info": {
                    "error": str(e),
                    "step": self.current_step,
                },
            }
    
    def _create_initial_prompt(self, task_id: int) -> str:
        """
        Create initial prompt for the task
        
        Args:
            task_id: Task ID
        
        Returns:
            Initial prompt string
        """
        game_config = self._parse_task_id(task_id)
        
        prompt = f"""You are playing a strategic game. Your goal is to make optimal decisions to win.

Game: {game_config['game_name']}
Configuration: {game_config['config_id']}

You will be presented with game states and must choose actions. Think strategically and consider:
1. Current game state
2. Available actions
3. Opponent behavior
4. Long-term strategy

CRITICAL OUTPUT FORMAT (for speed + correctness):
- Respond with ONLY the action ID integer (example: 5)
- No extra words, no punctuation, no explanation.
"""
        
        return prompt.strip()
    
    def _parse_task_id(self, task_id: int) -> Dict[str, Any]:
        """
        Parse task_id into game configuration
        
        task_id format: GGGGCCCCCCCC
        - GGGG: Game index (4 digits)
        - CCCCCCCC: Configuration variant (8 digits)
        
        Args:
            task_id: 12-digit task ID
        
        Returns:
            Dictionary with game configuration
        """
        # IMPORTANT: Keep parsing logic consistent with the actual OpenSpiel env.
        # The authoritative mapping is `affinetes.environments.openspiel.game_config`.
        try:
            from affinetes.environments.openspiel.game_config import decode_task_id  # type: ignore

            cfg = decode_task_id(int(task_id))
            return {
                # `game_idx` is the raw index derived from task_id (may be > number of games).
                # The selected game name is computed using modulo by the env.
                "game_id": int(cfg["game_idx"]),
                "game_name": str(cfg["game_name"]),
                "config_id": int(cfg["config_id"]),
            }
        except Exception:
            # Fallback: preserve previous behavior if OpenSpiel env isn't importable
            task_id_str = str(task_id).zfill(12)
            game_id = int(task_id_str[:4])
            config_id = int(task_id_str[4:])

            game_names = [
                "leduc_poker",
                "liars_dice",
                "battleship",
                "dark_hex",
                "phantom_ttt",
                "sheriff",
                "goofspiel",
                "tiny_bridge",
                "hearts",
                "cribbage",
                "euchre",
                "pig",
                "oh_hell",
                "chinese_checkers",
                "maedn",
                "bargaining",
                "negotiation",
                "trade_comm",
                "colored_trails",
                "hanabi",
            ]

            game_name = game_names[game_id % len(game_names)]
            return {"game_id": game_id, "game_name": game_name, "config_id": config_id}
    
    def _compute_reward(
        self,
        score: float,
        success: bool,
        extra: Dict[str, Any]
    ) -> float:
        """
        Compute reward for RL training
        
        Args:
            score: Game score (0.0-1.0)
            success: Whether the episode was successful
            extra: Additional information from environment
        
        Returns:
            Shaped reward value
        """
        if not self.reward_shaping:
            # Simple reward: just use the score
            return score
        
        # Apply reward shaping for better learning
        reward = 0.0
        
        # Base reward from score
        reward += score
        
        # Success bonus
        if success:
            reward += 0.5
        
        # Penalties for errors or timeouts
        if extra.get("error"):
            reward -= 1.0
        
        if extra.get("timeout"):
            reward -= 0.3
        
        # Normalize to reasonable range [-2, 2]
        reward = np.clip(reward, -2.0, 2.0)
        
        return float(reward)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episode_history:
            return {}
        
        return {
            "total_steps": len(self.episode_history),
            "total_reward": sum(h["reward"] for h in self.episode_history),
            "final_score": self.episode_history[-1]["score"],
            "success": self.episode_history[-1]["success"],
        }


class LocalInferenceWrapper:
    """
    Wrapper to use the trained model for inference during environment interaction
    
    This allows the environment to call the model being trained without needing
    to deploy it as a separate service.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize local inference wrapper
        
        Args:
            model: The model being trained
            tokenizer: Tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
