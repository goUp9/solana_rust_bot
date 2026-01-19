#!/usr/bin/env python3
"""
Local (in-process) OpenSpiel rollout runner.

This bypasses the docker/basilica environment server and calls the model directly,
so PPO rollouts always use the *current* policy weights.

Supports two inference modes:
1. Direct model inference (slower, uses training model)
2. vLLM API inference (faster, uses separate vLLM server)

Requires: open_spiel / pyspiel to be installed in the training environment.
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TurnSample:
    """One LLM decision turn sample for PPO."""
    prompt_text: str
    response_text: str
    reward: float
    info: Dict[str, Any]
    # Token IDs for PPO training (avoids re-encoding issues)
    query_ids: Optional[List[int]] = None
    response_ids: Optional[List[int]] = None


@dataclass
class EpisodeData:
    """Complete episode data for logging and analysis."""
    task_id: int
    seed: int
    game_name: str
    opponent: str
    llm_player_id: int
    num_players: int
    # Game outcome
    final_reward: float
    llm_return: float
    all_returns: List[float]
    # Turn-by-turn data
    conversation: List[Dict[str, str]]  # Full chat history
    action_history: List[Dict[str, Any]]  # All actions taken
    llm_turns: List[Dict[str, Any]]  # Detailed LLM turn info
    # Statistics
    total_turns: int
    valid_turns: int
    invalid_turns: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "game_name": self.game_name,
            "opponent": self.opponent,
            "llm_player_id": self.llm_player_id,
            "num_players": self.num_players,
            "final_reward": self.final_reward,
            "llm_return": self.llm_return,
            "all_returns": self.all_returns,
            "conversation": self.conversation,
            "action_history": self.action_history,
            "llm_turns": self.llm_turns,
            "total_turns": self.total_turns,
            "valid_turns": self.valid_turns,
            "invalid_turns": self.invalid_turns,
        }


def _import_openspiel_components():
    # Lazily import to keep module importable even if open_spiel isn't installed.
    import pyspiel  # type: ignore
    from open_spiel.python.algorithms import mcts  # type: ignore
    from open_spiel.python.bots import uniform_random  # type: ignore

    # Import from local game_config and agents modules
    from game_rl_training.game_config import create_game
    from game_rl_training.agents import GAME_AGENTS

    return pyspiel, mcts, uniform_random, create_game, GAME_AGENTS


def _build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Build a chat prompt for Qwen-style chat models.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: plain concatenation
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"{role.upper()}: {m.get('content','')}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _parse_action_id(text: str) -> Optional[int]:
    import re
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# Global process pool for MCTS rollouts (created lazily)
_MCTS_PROCESS_POOL = None
_MCTS_POOL_SIZE = None


def _get_mcts_process_pool(num_workers: int = None):
    """Get or create a global process pool for MCTS rollouts."""
    global _MCTS_PROCESS_POOL, _MCTS_POOL_SIZE
    import os
    from concurrent.futures import ProcessPoolExecutor
    
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    # Recreate pool if size changed
    if _MCTS_PROCESS_POOL is None or _MCTS_POOL_SIZE != num_workers:
        if _MCTS_PROCESS_POOL is not None:
            _MCTS_PROCESS_POOL.shutdown(wait=False)
        _MCTS_PROCESS_POOL = ProcessPoolExecutor(max_workers=num_workers)
        _MCTS_POOL_SIZE = num_workers
    
    return _MCTS_PROCESS_POOL


def _run_single_rollout_worker(args: Tuple[str, str, int, int]) -> List[float]:
    """
    Worker function for parallel rollouts. Runs in a separate process.
    
    Args:
        args: (game_name, state_str, seed, num_rollouts_per_worker)
    
    Returns:
        Sum of returns across all rollouts for this worker
    """
    game_name, state_str, seed, num_rollouts = args
    
    # Import OpenSpiel in the worker process
    import pyspiel
    import numpy as np
    
    game = pyspiel.load_game(game_name)
    state = game.deserialize_state(state_str)
    num_players = state.num_players()
    rng = np.random.RandomState(seed)
    
    total_returns = np.zeros(num_players)
    for _ in range(num_rollouts):
        working_state = state.clone()
        while not working_state.is_terminal():
            legal_actions = working_state.legal_actions()
            if not legal_actions:
                break
            action = rng.choice(legal_actions)
            working_state.apply_action(action)
        total_returns += np.array(working_state.returns())
    
    return total_returns.tolist()


def _create_opponent_bot(opponent: str, player_id: int, seed: int, game, agent, mcts_simulations: int = None, mcts_rollouts: int = None, mcts_workers: int = None):
    pyspiel, mcts, uniform_random, _, _ = _import_openspiel_components()

    game_type = game.get_type()
    if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        return uniform_random.UniformRandomBot(player_id=player_id, rng=np.random.RandomState(seed + 2))

    if opponent == "random":
        return uniform_random.UniformRandomBot(player_id=player_id, rng=np.random.RandomState(seed + 2))

    if opponent == "mcts":
        mcts_config = agent.get_mcts_config()
        if mcts_config is None:
            return uniform_random.UniformRandomBot(player_id=player_id, rng=np.random.RandomState(seed + 2))

        default_simulations, default_rollouts = mcts_config
        # Use provided mcts_simulations/mcts_rollouts or fall back to agent's defaults
        max_simulations = mcts_simulations if mcts_simulations is not None else default_simulations
        n_rollouts = mcts_rollouts if mcts_rollouts is not None else default_rollouts

        class _ParallelRandomRolloutEvaluator(mcts.Evaluator):
            """
            MCTS evaluator with parallel rollouts using multiprocessing.
            
            For high rollout counts (e.g., 200), this distributes work across
            all available CPU cores for significant speedup.
            """
            
            def __init__(self, n_rollouts=1, random_state=None, num_workers=None, game_name=None):
                import os
                self._n_rollouts = n_rollouts
                self._random_state = random_state or np.random.RandomState()
                self._num_workers = num_workers or (os.cpu_count() or 4)
                self._game_name = game_name
                # Only use multiprocessing for significant rollout counts
                self._use_multiprocessing = n_rollouts >= 16 and self._num_workers > 1
                
            def evaluate(self, state):
                if state.is_terminal():
                    return state.returns()
                legal_actions = state.legal_actions()
                if not legal_actions:
                    return state.returns()
                
                num_players = state.num_players()
                
                # For small rollout counts, use sequential (faster due to no overhead)
                if not self._use_multiprocessing or self._n_rollouts < 16:
                    total_returns = np.zeros(num_players)
                    for _ in range(self._n_rollouts):
                        working_state = state.clone()
                        while not working_state.is_terminal():
                            legal_actions = working_state.legal_actions()
                            if not legal_actions:
                                break
                            action = self._random_state.choice(legal_actions)
                            working_state.apply_action(action)
                        total_returns += working_state.returns()
                    return total_returns / self._n_rollouts
                
                # Distribute rollouts across workers
                state_str = state.serialize()
                base_seed = self._random_state.randint(0, 2**31)
                
                # Split rollouts among workers
                rollouts_per_worker = self._n_rollouts // self._num_workers
                remainder = self._n_rollouts % self._num_workers
                
                worker_args = []
                for i in range(self._num_workers):
                    n = rollouts_per_worker + (1 if i < remainder else 0)
                    if n > 0:
                        worker_args.append((self._game_name, state_str, base_seed + i * 1000, n))
                
                # Use global process pool
                pool = _get_mcts_process_pool(self._num_workers)
                
                total_returns = np.zeros(num_players)
                try:
                    results = list(pool.map(_run_single_rollout_worker, worker_args))
                    for result in results:
                        total_returns += np.array(result)
                except Exception as e:
                    # Fallback to sequential on error
                    for _ in range(self._n_rollouts):
                        working_state = state.clone()
                        while not working_state.is_terminal():
                            legal_actions = working_state.legal_actions()
                            if not legal_actions:
                                break
                            action = self._random_state.choice(legal_actions)
                            working_state.apply_action(action)
                        total_returns += working_state.returns()
                
                return total_returns / self._n_rollouts

            def prior(self, state):
                legal_actions = state.legal_actions()
                if not legal_actions:
                    return []
                prob = 1.0 / len(legal_actions)
                return [(a, prob) for a in legal_actions]

        # Get game name for serialization in workers
        game_name = game.get_type().short_name
        
        evaluator = _ParallelRandomRolloutEvaluator(
            n_rollouts=n_rollouts, 
            random_state=np.random.RandomState(seed + 3),
            num_workers=mcts_workers,
            game_name=game_name
        )
        return mcts.MCTSBot(
            game=game,
            uct_c=1.414,
            max_simulations=max_simulations,
            evaluator=evaluator,
            random_state=np.random.RandomState(seed + 4),
        )

    raise ValueError(f"Unknown opponent type: {opponent}")


def run_local_openspiel_episode(
    *,
    model,
    tokenizer,
    task_id: int,
    seed: int,
    opponent: str = "mcts",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8,
    max_seq_length: int = 1024,
    gamma: float = 0.99,
    device: str = "cuda",
    include_invalid: bool = False,
    invalid_penalty: float = -0.5,
    mcts_simulations: int = None,
    mcts_rollouts: int = None,
    mcts_workers: int = None,
) -> Tuple[List[TurnSample], Dict[str, Any], EpisodeData]:
    """
    Play a full OpenSpiel episode locally and return per-turn samples.

    Rewards: final outcome is discounted backwards across turns (Monte-Carlo style).
    
    Args:
        include_invalid: If True, include invalid responses with penalty reward
        invalid_penalty: Reward to assign to invalid responses (default -0.5)
        mcts_simulations: Number of MCTS simulations (lower = weaker opponent)
        mcts_rollouts: Number of rollouts per MCTS evaluation (higher = stronger opponent)
        mcts_workers: Number of CPU workers for parallel MCTS rollouts (default: all CPUs)
    
    Returns:
        turn_samples: List of TurnSample for PPO training
        episode_info: Dict with summary info for curriculum updates
        episode_data: EpisodeData with complete episode details for logging
    """
    pyspiel, _, _, create_game, GAME_AGENTS = _import_openspiel_components()

    game, game_cfg = create_game(task_id)
    game_name = game_cfg["game_name"]
    num_players = game.num_players()
    llm_player_id = (seed % num_players)

    agent_class = GAME_AGENTS.get(game_name)
    if not agent_class:
        raise ValueError(f"No agent found for game: {game_name}")
    agent = agent_class()

    # Build bots: model plays one seat, opponents fill the rest.
    bots = [None] * num_players
    for pid in range(num_players):
        if pid != llm_player_id:
            bots[pid] = _create_opponent_bot(opponent, pid, seed + 2 + pid, game, agent, mcts_simulations=mcts_simulations, mcts_rollouts=mcts_rollouts, mcts_workers=mcts_workers)

    state = game.new_initial_state()
    messages: List[Dict[str, str]] = [{"role": "system", "content": agent.generate_system_prompt()}]
    action_history: List[Dict[str, Any]] = []
    turn_samples: List[TurnSample] = []
    llm_turns: List[Dict[str, Any]] = []  # Detailed LLM turn info for logging
    valid_count = 0
    invalid_count = 0

    # Local RNG for fallback action selection
    rng = np.random.RandomState(seed + 1)

    while not state.is_terminal():
        cur_player = state.current_player()

        # Chance nodes (dice / cards dealt etc.)
        if cur_player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            a = rng.choice(actions, p=probs)
            state.apply_action(a)
            continue

        # Non-LLM players use opponent bots
        if cur_player != llm_player_id:
            # Handle simultaneous move games where cur_player might be SIMULTANEOUS
            if cur_player < 0 or cur_player >= num_players:
                # Simultaneous move - all players act at once
                # For simplicity, treat as LLM's turn and let opponents use random
                legal_actions = state.legal_actions()
                if legal_actions:
                    a = int(rng.choice(legal_actions))
                    state.apply_action(a)
                continue
            
            if bots[cur_player] is None:
                # Fallback: use random action if bot not initialized
                legal_actions = state.legal_actions(cur_player)
                a = int(rng.choice(legal_actions)) if legal_actions else 0
            else:
                a = bots[cur_player].step(state)
            state.apply_action(a)
            action_history.append({"player_id": int(cur_player), "action": int(a), "is_llm": False})
            continue

        # LLM turn
        legal_actions = state.legal_actions(llm_player_id)
        user_prompt = agent.generate_user_prompt(state=state, player_id=llm_player_id, legal_actions=legal_actions)
        messages.append({"role": "user", "content": user_prompt})

        prompt_text = _build_chat_prompt(tokenizer, messages)

        # Tokenize and generate (keep it short)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(32, max_seq_length - max_new_tokens),
        ).to(device)

        # Use inference_mode for faster generation (no gradient tracking)
        import torch
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Get the actual generated tokens (not decoded text) for PPO
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        action_id = _parse_action_id(response_text)

        # Track if this was a valid model response or a fallback
        is_valid_response = action_id is not None and action_id in legal_actions

        # Enforce validity; fallback to random legal action if invalid
        actual_action_id = action_id if is_valid_response else (int(rng.choice(legal_actions)) if legal_actions else 0)

        # Apply action
        state.apply_action(actual_action_id)
        action_history.append({"player_id": int(llm_player_id), "action": int(actual_action_id), "is_llm": True, "valid": is_valid_response})
        messages.append({"role": "assistant", "content": str(actual_action_id)})

        # Track valid/invalid counts
        if is_valid_response:
            valid_count += 1
        else:
            invalid_count += 1

        # Record detailed LLM turn info for logging
        llm_turns.append({
            "turn_index": len(llm_turns),
            "user_prompt": user_prompt,
            "raw_model_output": response_text,
            "parsed_action_id": action_id,
            "actual_action_id": actual_action_id,
            "legal_actions": legal_actions[:20] if len(legal_actions) > 20 else legal_actions,  # Truncate for readability
            "num_legal_actions": len(legal_actions),
            "is_valid": is_valid_response,
        })

        # Include responses for PPO training based on validity
        # Valid responses: reward will be filled with game outcome
        # Invalid responses: immediate penalty reward (if include_invalid=True)
        if is_valid_response or include_invalid:
            turn_samples.append(
                TurnSample(
                    prompt_text=prompt_text,
                    response_text=response_text.strip(),  # Use actual model output
                    reward=0.0 if is_valid_response else invalid_penalty,  # Invalid gets immediate penalty
                    info={
                        "game_name": game_name,
                        "task_id": task_id,
                        "seed": seed,
                        "opponent": opponent,
                        "llm_player_id": llm_player_id,
                        "valid_response": is_valid_response,
                    },
                    # Pass actual token IDs to avoid re-encoding issues with KL divergence
                    query_ids=inputs["input_ids"][0].tolist(),
                    response_ids=response_ids.tolist(),
                )
            )

    returns = state.returns()
    llm_return = float(returns[llm_player_id])

    # Convert to score-like reward in [0,1] for most games
    # (simple heuristic: for 2p zero-sum games with min=-1 max=1, map to [0,1])
    try:
        min_u = game.min_utility()
        max_u = game.max_utility()
        if max_u > min_u:
            final_reward = (llm_return - min_u) / (max_u - min_u)
        else:
            final_reward = 0.0
    except Exception:
        final_reward = llm_return

    # Assign rewards to each turn sample (discounted game outcome, Monte-Carlo style)
    # Only update rewards for valid responses; invalid responses keep their penalty
    for i in range(len(turn_samples) - 1, -1, -1):
        if turn_samples[i].info.get("valid_response", True):
            turn_samples[i].reward = float((gamma ** (len(turn_samples) - 1 - i)) * final_reward)
        # Invalid responses keep their penalty reward (already set)

    episode_info = {
        "game_name": game_name,
        "task_id": task_id,
        "seed": seed,
        "opponent": opponent,
        "llm_player_id": llm_player_id,
        "final_reward": float(final_reward),
        "llm_return": float(llm_return),
        "num_turns": len(turn_samples),
        "action_history": action_history,
        "valid_turns": valid_count,
        "invalid_turns": invalid_count,
    }

    # Create complete episode data for logging
    episode_data = EpisodeData(
        task_id=task_id,
        seed=seed,
        game_name=game_name,
        opponent=opponent,
        llm_player_id=llm_player_id,
        num_players=num_players,
        final_reward=float(final_reward),
        llm_return=float(llm_return),
        all_returns=[float(r) for r in returns],
        conversation=messages,
        action_history=action_history,
        llm_turns=llm_turns,
        total_turns=valid_count + invalid_count,
        valid_turns=valid_count,
        invalid_turns=invalid_count,
    )

    return turn_samples, episode_info, episode_data


async def _vllm_generate(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    vllm_base_url: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Call vLLM OpenAI-compatible API for text generation."""
    url = f"{vllm_base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"vLLM API error {resp.status}: {text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except asyncio.TimeoutError:
        raise RuntimeError("vLLM API timeout")


async def run_vllm_openspiel_episode(
    *,
    tokenizer,  # Still needed for prompt formatting
    task_id: int,
    seed: int,
    vllm_base_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen3-4B",
    opponent: str = "random",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8,
    max_seq_length: int = 1024,
    gamma: float = 0.99,
    session: Optional[aiohttp.ClientSession] = None,
    mcts_simulations: int = None,
    mcts_rollouts: int = None,
    mcts_workers: int = None,
) -> Tuple[List[TurnSample], Dict[str, Any], EpisodeData]:
    """
    Play a full OpenSpiel episode using vLLM API for inference.
    
    This is much faster than direct model inference because:
    1. vLLM uses continuous batching
    2. vLLM uses PagedAttention for efficient memory
    3. Multiple episodes can run in parallel
    
    Returns:
        turn_samples: List of TurnSample for PPO training (only valid responses)
        episode_info: Dict with summary info for curriculum updates
        episode_data: EpisodeData with complete episode details for logging
    """
    pyspiel, _, _, create_game, GAME_AGENTS = _import_openspiel_components()

    game, game_cfg = create_game(task_id)
    game_name = game_cfg["game_name"]
    num_players = game.num_players()
    llm_player_id = (seed % num_players)

    agent_class = GAME_AGENTS.get(game_name)
    if not agent_class:
        raise ValueError(f"No agent found for game: {game_name}")
    agent = agent_class()

    # Build bots: model plays one seat, opponents fill the rest.
    bots = [None] * num_players
    for pid in range(num_players):
        if pid != llm_player_id:
            bots[pid] = _create_opponent_bot(opponent, pid, seed + 2 + pid, game, agent, mcts_simulations=mcts_simulations, mcts_rollouts=mcts_rollouts, mcts_workers=mcts_workers)

    state = game.new_initial_state()
    messages: List[Dict[str, str]] = [{"role": "system", "content": agent.generate_system_prompt()}]
    action_history: List[Dict[str, Any]] = []
    turn_samples: List[TurnSample] = []
    llm_turns: List[Dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0

    # Local RNG for fallback action selection
    rng = np.random.RandomState(seed + 1)
    
    # Create session if not provided
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        while not state.is_terminal():
            cur_player = state.current_player()

            # Chance nodes (dice / cards dealt etc.)
            if cur_player == pyspiel.PlayerId.CHANCE:
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                a = rng.choice(actions, p=probs)
                state.apply_action(a)
                continue

            # Non-LLM players use opponent bots
            if cur_player != llm_player_id:
                if cur_player < 0 or cur_player >= num_players:
                    legal_actions = state.legal_actions()
                    if legal_actions:
                        a = int(rng.choice(legal_actions))
                        state.apply_action(a)
                    continue
                
                if bots[cur_player] is None:
                    legal_actions = state.legal_actions(cur_player)
                    a = int(rng.choice(legal_actions)) if legal_actions else 0
                else:
                    a = bots[cur_player].step(state)
                state.apply_action(a)
                action_history.append({"player_id": int(cur_player), "action": int(a), "is_llm": False})
                continue

            # LLM turn - use vLLM API
            legal_actions = state.legal_actions(llm_player_id)
            user_prompt = agent.generate_user_prompt(state=state, player_id=llm_player_id, legal_actions=legal_actions)
            messages.append({"role": "user", "content": user_prompt})

            # Call vLLM API
            response_text = await _vllm_generate(
                session=session,
                messages=messages,
                vllm_base_url=vllm_base_url,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            
            action_id = _parse_action_id(response_text)

            # Track if this was a valid model response or a fallback
            is_valid_response = action_id is not None and action_id in legal_actions

            # Enforce validity; fallback to random legal action if invalid
            actual_action_id = action_id if is_valid_response else (int(rng.choice(legal_actions)) if legal_actions else 0)

            # Apply action
            state.apply_action(actual_action_id)
            action_history.append({"player_id": int(llm_player_id), "action": int(actual_action_id), "is_llm": True, "valid": is_valid_response})
            messages.append({"role": "assistant", "content": str(actual_action_id)})

            # Track valid/invalid counts
            if is_valid_response:
                valid_count += 1
            else:
                invalid_count += 1

            # Record detailed LLM turn info for logging
            llm_turns.append({
                "turn_index": len(llm_turns),
                "user_prompt": user_prompt,
                "raw_model_output": response_text,
                "parsed_action_id": action_id,
                "actual_action_id": actual_action_id,
                "legal_actions": legal_actions[:20] if len(legal_actions) > 20 else legal_actions,
                "num_legal_actions": len(legal_actions),
                "is_valid": is_valid_response,
            })

            # Build prompt text for PPO (needed for tokenization later)
            prompt_text = _build_chat_prompt(tokenizer, messages[:-1])  # Exclude assistant response
            
            # Only include VALID responses for PPO training
            if is_valid_response:
                turn_samples.append(
                    TurnSample(
                        prompt_text=prompt_text,
                        response_text=response_text.strip(),
                        reward=0.0,
                        info={
                            "game_name": game_name,
                            "task_id": task_id,
                            "seed": seed,
                            "opponent": opponent,
                            "llm_player_id": llm_player_id,
                            "valid_response": True,
                        },
                    )
                )

    finally:
        if own_session:
            await session.close()

    returns = state.returns()
    llm_return = float(returns[llm_player_id])

    # Convert to score-like reward in [0,1]
    try:
        min_u = game.min_utility()
        max_u = game.max_utility()
        if max_u > min_u:
            final_reward = (llm_return - min_u) / (max_u - min_u)
        else:
            final_reward = 0.0
    except Exception:
        final_reward = llm_return

    # Assign rewards to each turn sample (discounted game outcome)
    for i in range(len(turn_samples) - 1, -1, -1):
        turn_samples[i].reward = float((gamma ** (len(turn_samples) - 1 - i)) * final_reward)

    episode_info = {
        "game_name": game_name,
        "task_id": task_id,
        "seed": seed,
        "opponent": opponent,
        "llm_player_id": llm_player_id,
        "final_reward": float(final_reward),
        "llm_return": float(llm_return),
        "num_turns": len(turn_samples),
        "action_history": action_history,
        "valid_turns": valid_count,
        "invalid_turns": invalid_count,
    }

    episode_data = EpisodeData(
        task_id=task_id,
        seed=seed,
        game_name=game_name,
        opponent=opponent,
        llm_player_id=llm_player_id,
        num_players=num_players,
        final_reward=float(final_reward),
        llm_return=float(llm_return),
        all_returns=[float(r) for r in returns],
        conversation=messages,
        action_history=action_history,
        llm_turns=llm_turns,
        total_turns=valid_count + invalid_count,
        valid_turns=valid_count,
        invalid_turns=invalid_count,
    )

    return turn_samples, episode_info, episode_data


async def run_parallel_episodes(
    *,
    tokenizer,
    task_configs: List[Dict[str, Any]],
    vllm_base_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen3-4B",
    opponent: str = "random",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8,
    max_seq_length: int = 1024,
    gamma: float = 0.99,
    max_concurrent: int = 8,
) -> List[Tuple[List[TurnSample], Dict[str, Any], EpisodeData]]:
    """
    Run multiple episodes in parallel using vLLM API.
    
    This provides significant speedup by:
    1. Running multiple games concurrently
    2. vLLM batches the inference requests automatically
    
    Args:
        task_configs: List of dicts with 'task_id' and 'seed' keys
        max_concurrent: Maximum number of concurrent episodes
        
    Returns:
        List of (turn_samples, episode_info, episode_data) tuples
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task_config):
        async with semaphore:
            return await run_vllm_openspiel_episode(
                tokenizer=tokenizer,
                task_id=task_config["task_id"],
                seed=task_config["seed"],
                vllm_base_url=vllm_base_url,
                model_name=model_name,
                opponent=opponent,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                max_seq_length=max_seq_length,
                gamma=gamma,
            )
    
    # Create shared session for connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [run_with_semaphore(cfg) for cfg in task_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and log them
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Episode {i} failed: {result}")
        else:
            valid_results.append(result)
    
    return valid_results


def _run_episode_worker(args: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, Any], Dict[str, Any]]:
    """
    Worker function for parallel episode collection.
    Runs a single episode without model - returns game states for batched inference.
    
    This is a CPU-only worker that handles MCTS opponent moves.
    Model inference happens in the main process after collecting states.
    """
    import numpy as np
    
    task_id = args["task_id"]
    seed = args["seed"]
    opponent = args["opponent"]
    mcts_simulations = args.get("mcts_simulations", 100)
    mcts_rollouts = args.get("mcts_rollouts", None)
    mcts_workers = args.get("mcts_workers", None)
    
    pyspiel, mcts_mod, uniform_random, create_game, GAME_AGENTS = _import_openspiel_components()
    
    game, game_cfg = create_game(task_id)
    game_name = game_cfg["game_name"]
    num_players = game.num_players()
    llm_player_id = (seed % num_players)
    
    agent_class = GAME_AGENTS.get(game_name)
    if not agent_class:
        raise ValueError(f"No agent found for game: {game_name}")
    agent = agent_class()
    
    # Build opponent bots
    bots = [None] * num_players
    for pid in range(num_players):
        if pid != llm_player_id:
            bots[pid] = _create_opponent_bot(opponent, pid, seed + 2 + pid, game, agent, mcts_simulations=mcts_simulations, mcts_rollouts=mcts_rollouts, mcts_workers=mcts_workers)
    
    state = game.new_initial_state()
    rng = np.random.RandomState(seed + 1)
    
    # Collect states where LLM needs to make decisions
    llm_decision_points = []
    action_history = []
    system_prompt = agent.generate_system_prompt()
    
    while not state.is_terminal():
        cur_player = state.current_player()
        
        # Chance nodes
        if cur_player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            a = rng.choice(actions, p=probs)
            state.apply_action(a)
            continue
        
        # Non-LLM players (MCTS/random opponents)
        if cur_player != llm_player_id:
            if cur_player < 0 or cur_player >= num_players:
                legal_actions = state.legal_actions()
                if legal_actions:
                    a = int(rng.choice(legal_actions))
                    state.apply_action(a)
                continue
            
            if bots[cur_player] is None:
                legal_actions = state.legal_actions(cur_player)
                a = int(rng.choice(legal_actions)) if legal_actions else 0
            else:
                a = bots[cur_player].step(state)
            state.apply_action(a)
            action_history.append({"player_id": int(cur_player), "action": int(a), "is_llm": False})
            continue
        
        # LLM turn - save state for later batch inference
        legal_actions = state.legal_actions(llm_player_id)
        user_prompt = agent.generate_user_prompt(state=state, player_id=llm_player_id, legal_actions=legal_actions)
        
        llm_decision_points.append({
            "state_str": str(state),
            "user_prompt": user_prompt,
            "legal_actions": legal_actions,
            "state_clone": state.serialize(),  # Serialize for later
        })
        
        # For now, use random action as placeholder (will be replaced by model inference)
        placeholder_action = int(rng.choice(legal_actions)) if legal_actions else 0
        state.apply_action(placeholder_action)
        action_history.append({"player_id": int(llm_player_id), "action": placeholder_action, "is_llm": True, "placeholder": True})
    
    episode_info = {
        "game_name": game_name,
        "task_id": task_id,
        "seed": seed,
        "opponent": opponent,
        "llm_player_id": llm_player_id,
        "num_players": num_players,
        "system_prompt": system_prompt,
        "final_returns": list(state.returns()),
    }
    
    return llm_decision_points, episode_info, {"action_history": action_history}


def run_parallel_episodes_cpu(
    num_episodes: int,
    task_configs: List[Dict[str, Any]],
    num_workers: int = 4,
) -> List[Tuple[List[Dict], Dict[str, Any], Dict[str, Any]]]:
    """
    Run multiple episodes in parallel using CPU workers for MCTS.
    
    This parallelizes the MCTS opponent computations across multiple CPU cores.
    Model inference still happens in the main process.
    
    Args:
        num_episodes: Number of episodes to run
        task_configs: List of task configurations
        num_workers: Number of parallel workers
        
    Returns:
        List of (llm_decision_points, episode_info, metadata) tuples
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_episode_worker, cfg): i for i, cfg in enumerate(task_configs[:num_episodes])}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Episode {idx} failed: {e}")
    
    # Sort by original index and return just the results
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

