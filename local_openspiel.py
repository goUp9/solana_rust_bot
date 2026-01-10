#!/usr/bin/env python3
"""
Local (in-process) OpenSpiel rollout runner.

This bypasses the docker/basilica environment server and calls the model directly,
so PPO rollouts always use the *current* policy weights.

Requires: open_spiel / pyspiel to be installed in the training environment.
"""

from __future__ import annotations

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
    import sys
    import pyspiel  # type: ignore
    from open_spiel.python.algorithms import mcts  # type: ignore
    from open_spiel.python.bots import uniform_random  # type: ignore

    # Add the actual affinetes repo to path for direct import
    affinetes_openspiel_path = "/root/workplace/affinetes/environments/openspiel"
    if affinetes_openspiel_path not in sys.path:
        sys.path.insert(0, affinetes_openspiel_path)
    
    from game_config import create_game  # type: ignore
    from agents import GAME_AGENTS  # type: ignore

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


def _create_opponent_bot(opponent: str, player_id: int, seed: int, game, agent):
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

        max_simulations, n_rollouts = mcts_config

        class _SafeRandomRolloutEvaluator(mcts.Evaluator):
            def __init__(self, n_rollouts=1, random_state=None):
                self._n_rollouts = n_rollouts
                self._random_state = random_state or np.random.RandomState()

            def evaluate(self, state):
                if state.is_terminal():
                    return state.returns()
                legal_actions = state.legal_actions()
                if not legal_actions:
                    return state.returns()
                total_returns = np.zeros(state.num_players())
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

        evaluator = _SafeRandomRolloutEvaluator(n_rollouts=n_rollouts, random_state=np.random.RandomState(seed + 3))
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
) -> Tuple[List[TurnSample], Dict[str, Any], EpisodeData]:
    """
    Play a full OpenSpiel episode locally and return per-turn samples.

    Rewards: final outcome is discounted backwards across turns (Monte-Carlo style).
    
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
            bots[pid] = _create_opponent_bot(opponent, pid, seed + 2 + pid, game, agent)

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

        # Only include VALID responses for PPO training
        # Invalid responses cause KL divergence issues (policy ratio explosion)
        # Use the ACTUAL model output (not cleaned) to maintain KL consistency
        if is_valid_response:
            # Use the actual model output (trimmed) to match what the model generated
            # This prevents KL divergence issues from mismatched tokens
            turn_samples.append(
                TurnSample(
                    prompt_text=prompt_text,
                    response_text=response_text.strip(),  # Use actual model output
                    reward=0.0,  # Will be filled with game outcome
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

