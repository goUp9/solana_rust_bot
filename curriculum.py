#!/usr/bin/env python3
"""
Curriculum Learning and Failure-Based Sampling
Implements adaptive task sampling based on:
1. Curriculum stages (easy -> hard progression)
2. Failure history (prioritize failed tasks for replay)
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from pathlib import Path


@dataclass
class TaskConfig:
    """Configuration for a single task"""
    task_id: int
    difficulty: str  # "easy", "medium", "hard"
    opponent: str  # "random", "mcts"
    seed: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class CurriculumStage:
    """A stage in the curriculum"""
    name: str
    task_range: Tuple[int, int]  # (min_task_id, max_task_id)
    opponent: str  # "random" or "mcts"
    weight: float = 1.0  # Sampling weight
    min_success_rate: float = 0.6  # Success rate to advance to next stage


class FailureBuffer:
    """
    Buffer to store and sample from failed tasks
    Implements experience replay with prioritization for failed tasks
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize failure buffer
        
        Args:
            max_size: Maximum number of failed tasks to remember
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.task_stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
    
    def add(self, task_id: int, score: float, success: bool):
        """
        Add a task result to the buffer
        
        Args:
            task_id: Task ID
            score: Task score (0.0-1.0)
            success: Whether the task was successful
        """
        # Update statistics
        self.task_stats[task_id]["attempts"] += 1
        if success:
            self.task_stats[task_id]["successes"] += 1
        else:
            self.task_stats[task_id]["failures"] += 1
        
        # Add to buffer if failed or low score
        if not success or score < 0.5:
            self.buffer.append({
                "task_id": task_id,
                "score": score,
                "success": success,
                "priority": 1.0 - score,  # Lower scores = higher priority
            })
    
    def sample(self, k: int = 1) -> List[int]:
        """
        Sample task IDs from the buffer with priority
        
        Args:
            k: Number of tasks to sample
        
        Returns:
            List of task IDs
        """
        if not self.buffer:
            return []
        
        # Get priorities
        priorities = np.array([item["priority"] for item in self.buffer])
        
        # Normalize to probabilities
        probs = priorities / priorities.sum()
        
        # Sample with replacement
        indices = np.random.choice(
            len(self.buffer),
            size=min(k, len(self.buffer)),
            replace=False,
            p=probs
        )
        
        return [self.buffer[i]["task_id"] for i in indices]
    
    def get_task_stats(self, task_id: int) -> Dict:
        """Get statistics for a specific task"""
        stats = self.task_stats[task_id]
        success_rate = stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
        return {
            "attempts": stats["attempts"],
            "successes": stats["successes"],
            "failures": stats["failures"],
            "success_rate": success_rate,
        }
    
    def get_overall_stats(self) -> Dict:
        """Get overall buffer statistics"""
        total_attempts = sum(s["attempts"] for s in self.task_stats.values())
        total_successes = sum(s["successes"] for s in self.task_stats.values())
        
        return {
            "buffer_size": len(self.buffer),
            "unique_tasks": len(self.task_stats),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
        }


class CurriculumSampler:
    """
    Curriculum-based task sampler with failure replay
    
    Features:
    1. Multi-stage curriculum (easy -> hard)
    2. Automatic stage progression based on success rate
    3. Failure-based replay buffer
    4. Mixed sampling (curriculum + failures)
    """
    
    def __init__(
        self,
        env_name: str,
        curriculum_stages: List[Dict],
        failure_buffer_size: int = 1000,
        failure_replay_prob: float = 0.3,
        progression_threshold: float = 0.7,
        eval_window: int = 100,
        allowed_game_indices: Optional[List[int]] = None,
        game_block_size: int = 100_000_000,
    ):
        """
        Initialize curriculum sampler
        
        Args:
            env_name: Environment name
            curriculum_stages: List of curriculum stage configurations
            failure_buffer_size: Size of failure replay buffer
            failure_replay_prob: Probability of sampling from failure buffer
            progression_threshold: Success rate threshold to advance stage
            eval_window: Window size for computing success rate
        """
        self.env_name = env_name
        self.failure_replay_prob = failure_replay_prob
        self.progression_threshold = progression_threshold
        self.eval_window = eval_window
        # Optional: restrict sampling to a subset of OpenSpiel "game_idx" blocks.
        # In the OpenSpiel env, game_idx is derived as task_id // game_block_size.
        self.allowed_game_indices = list(allowed_game_indices) if allowed_game_indices else None
        self.game_block_size = int(game_block_size)
        
        # Initialize stages
        self.stages = [
            CurriculumStage(**stage) for stage in curriculum_stages
        ]
        self.current_stage_idx = 0
        
        # Initialize failure buffer
        self.failure_buffer = FailureBuffer(max_size=failure_buffer_size)
        
        # Track recent performance for stage progression
        self.recent_results = deque(maxlen=eval_window)
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "stage_changes": 0,
            "failure_replays": 0,
            "curriculum_samples": 0,
        }
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    def sample_task(self, eval_mode: bool = False) -> TaskConfig:
        """
        Sample a task using curriculum + failure replay
        
        Args:
            eval_mode: If True, sample uniformly for evaluation
        
        Returns:
            TaskConfig for the sampled task
        """
        self.stats["total_samples"] += 1
        
        # Evaluation mode: sample uniformly from current stage
        if eval_mode:
            return self._sample_from_curriculum()
        
        # Training mode: mix curriculum and failure replay
        if random.random() < self.failure_replay_prob and len(self.failure_buffer.buffer) > 0:
            # Sample from failure buffer
            task_ids = self.failure_buffer.sample(k=1)
            if task_ids:
                # If we're restricting games, skip disallowed tasks from replay.
                if self.allowed_game_indices is not None and not self._is_task_allowed(task_ids[0]):
                    # Fall back to curriculum sampling
                    return self._sample_from_curriculum()
                self.stats["failure_replays"] += 1
                return TaskConfig(
                    task_id=task_ids[0],
                    difficulty=self.current_stage.name,
                    opponent=self.current_stage.opponent,
                    seed=random.randint(0, 2**31 - 1),
                    metadata={"source": "failure_buffer"}
                )
        
        # Sample from current curriculum stage
        self.stats["curriculum_samples"] += 1
        return self._sample_from_curriculum()
    
    def _task_id_to_game_idx(self, task_id: int) -> int:
        return int(task_id) // self.game_block_size

    def _is_task_allowed(self, task_id: int) -> bool:
        if self.allowed_game_indices is None:
            return True
        return self._task_id_to_game_idx(task_id) in set(self.allowed_game_indices)

    def _sample_allowed_task_id(self) -> int:
        """
        Sample a task_id restricted to allowed_game_indices.
        task_id = game_idx * block + config_id
        """
        if not self.allowed_game_indices:
            raise ValueError("allowed_game_indices is empty; cannot sample task_id.")
        game_idx = random.choice(self.allowed_game_indices)
        config_id = random.randint(0, self.game_block_size - 1)
        return int(game_idx) * self.game_block_size + int(config_id)

    def _sample_from_curriculum(self) -> TaskConfig:
        """Sample a task from the current curriculum stage"""
        stage = self.current_stage
        
        # If configured, restrict sampling to specific OpenSpiel game indices
        if self.allowed_game_indices is not None:
            task_id = self._sample_allowed_task_id()
        else:
            # Generate random task_id in stage range
            min_id, max_id = stage.task_range
            task_id = random.randint(min_id, max_id)
        
        return TaskConfig(
            task_id=task_id,
            difficulty=stage.name,
            opponent=stage.opponent,
            seed=random.randint(0, 2**31 - 1),
            metadata={"source": "curriculum", "stage": stage.name}
        )
    
    def update(self, task_id: int, score: float, success: bool):
        """
        Update sampler with task result
        
        Args:
            task_id: Task ID
            score: Task score (0.0-1.0)
            success: Whether task was successful
        """
        # Add to failure buffer
        self.failure_buffer.add(task_id, score, success)
        
        # Track recent results
        self.recent_results.append(success)
        
        # Check for stage progression
        if len(self.recent_results) >= self.eval_window:
            self._check_stage_progression()
    
    def _check_stage_progression(self):
        """Check if we should advance to the next curriculum stage"""
        # Compute success rate over recent window
        success_rate = sum(self.recent_results) / len(self.recent_results)
        
        # Check if we meet the threshold and there's a next stage
        if (success_rate >= self.progression_threshold and
            self.current_stage_idx < len(self.stages) - 1):
            
            self.current_stage_idx += 1
            self.stats["stage_changes"] += 1
            
            # Clear recent results for fresh evaluation
            self.recent_results.clear()
            
            print(f"🎓 Curriculum Progression: Advanced to stage {self.current_stage_idx + 1}/{len(self.stages)}")
            print(f"   Stage: {self.current_stage.name} | Opponent: {self.current_stage.opponent}")
            print(f"   Success rate: {success_rate:.2%}")
    
    def get_state(self) -> Dict:
        """
        Get current sampler state for checkpointing
        
        Returns:
            Dictionary with sampler state
        """
        return {
            "current_stage_idx": self.current_stage_idx,
            "stats": self.stats,
            "buffer_stats": self.failure_buffer.get_overall_stats(),
            "recent_results": list(self.recent_results),
        }
    
    def load_state(self, state: Dict):
        """
        Load sampler state from checkpoint
        
        Args:
            state: Dictionary with sampler state
        """
        self.current_stage_idx = state.get("current_stage_idx", 0)
        self.stats = state.get("stats", self.stats)
        
        recent_results = state.get("recent_results", [])
        self.recent_results.clear()
        self.recent_results.extend(recent_results)
    
    def get_curriculum_info(self) -> Dict:
        """
        Get information about current curriculum state
        
        Returns:
            Dictionary with curriculum information
        """
        current_stage = self.current_stage
        
        info = {
            "stage_idx": self.current_stage_idx,
            "stage_name": current_stage.name,
            "stage_opponent": current_stage.opponent,
            "total_stages": len(self.stages),
            "recent_success_rate": (
                sum(self.recent_results) / len(self.recent_results)
                if self.recent_results else 0.0
            ),
            "samples_in_window": len(self.recent_results),
            "progression_threshold": self.progression_threshold,
        }
        
        return info


class DynamicCurriculumSampler(CurriculumSampler):
    """
    Advanced curriculum sampler with dynamic difficulty adjustment
    
    Extends CurriculumSampler with:
    1. Automatic difficulty adjustment based on performance
    2. Task-level difficulty tracking
    3. Adaptive opponent strength
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Track per-task difficulty
        self.task_difficulties = defaultdict(lambda: {"score_history": []})
    
    def update(self, task_id: int, score: float, success: bool):
        """Update with additional task difficulty tracking"""
        super().update(task_id, score, success)
        
        # Track task-specific difficulty
        self.task_difficulties[task_id]["score_history"].append(score)
        
        # Keep only recent history
        if len(self.task_difficulties[task_id]["score_history"]) > 10:
            self.task_difficulties[task_id]["score_history"].pop(0)
    
    def get_task_difficulty(self, task_id: int) -> float:
        """
        Estimate task difficulty based on historical performance
        
        Args:
            task_id: Task ID
        
        Returns:
            Estimated difficulty (0.0=easy, 1.0=hard)
        """
        history = self.task_difficulties[task_id]["score_history"]
        
        if not history:
            return 0.5  # Unknown difficulty
        
        # Difficulty = 1 - average score
        avg_score = sum(history) / len(history)
        return 1.0 - avg_score
    
    def sample_task(self, eval_mode: bool = False) -> TaskConfig:
        """Sample task with dynamic difficulty consideration"""
        task_config = super().sample_task(eval_mode)
        
        # Add difficulty estimate to metadata
        difficulty_estimate = self.get_task_difficulty(task_config.task_id)
        task_config.metadata["estimated_difficulty"] = difficulty_estimate
        
        return task_config


def save_curriculum_checkpoint(sampler: CurriculumSampler, path: Path):
    """
    Save curriculum sampler state to file
    
    Args:
        sampler: CurriculumSampler instance
        path: Path to save checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = sampler.get_state()
    
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def load_curriculum_checkpoint(sampler: CurriculumSampler, path: Path):
    """
    Load curriculum sampler state from file
    
    Args:
        sampler: CurriculumSampler instance
        path: Path to load checkpoint from
    """
    if not path.exists():
        return
    
    with open(path, 'r') as f:
        state = json.load(f)
    
    sampler.load_state(state)
