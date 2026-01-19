#!/usr/bin/env python3
"""
OpenSpiel game configuration and creation utilities.

This module provides the game definitions and task ID encoding/decoding
for the RL training environment.
"""

from typing import Any, Dict, List, Tuple, Optional
import pyspiel  # type: ignore


# Available games with their OpenSpiel names and configurations
# Format: (openspiel_name, default_params, num_players)
AVAILABLE_GAMES: List[Tuple[str, Dict[str, Any], int]] = [
    # Index 0: Goofspiel - bidding card game
    ("goofspiel", {"num_cards": 4, "points_order": "descending"}, 2),
    # Index 1: Liar's Dice - bluffing dice game  
    ("liars_dice", {"numdice": 1}, 2),
    # Index 2: Leduc Poker - simplified poker
    ("leduc_poker", {}, 2),
    # Index 3: Gin Rummy - card game (slow)
    ("gin_rummy", {}, 2),
    # Index 4: Othello/Reversi - board game (slow)
    ("othello", {}, 2),
    # Index 5: Backgammon - classic board game (very slow)
    ("backgammon", {}, 2),
    # Index 6: Hex - connection board game
    ("hex", {"board_size": 5}, 2),
    # Index 7: Clobber - capture board game
    ("clobber", {"rows": 4, "columns": 5}, 2),
    # Index 8: Hearts - trick-taking card game
    ("hearts", {}, 4),
    # Index 9: Euchre - trick-taking card game
    ("euchre", {}, 4),
    # Index 10: Dots and Boxes - connection game
    ("dots_and_boxes", {"num_rows": 2, "num_cols": 2}, 2),
    # Index 11: Go - classic board game
    ("go", {"board_size": 5, "komi": 5.5}, 2),
]

# Game index to name mapping
GAME_INDEX_TO_NAME = {i: game[0] for i, game in enumerate(AVAILABLE_GAMES)}
GAME_NAME_TO_INDEX = {game[0]: i for i, game in enumerate(AVAILABLE_GAMES)}

# Block size for task ID encoding (100 million per game)
GAME_BLOCK_SIZE = 100_000_000


def decode_task_id(task_id: int) -> Dict[str, Any]:
    """
    Decode a task ID into game configuration.
    
    Task ID encoding:
    - game_index = task_id // GAME_BLOCK_SIZE
    - config_id = task_id % GAME_BLOCK_SIZE
    
    Args:
        task_id: The encoded task identifier
        
    Returns:
        Dict with game_name, game_index, config_id, seed
    """
    game_index = task_id // GAME_BLOCK_SIZE
    config_id = task_id % GAME_BLOCK_SIZE
    
    if game_index < 0 or game_index >= len(AVAILABLE_GAMES):
        raise ValueError(f"Invalid game index {game_index} from task_id {task_id}")
    
    game_name = GAME_INDEX_TO_NAME[game_index]
    
    return {
        "game_name": game_name,
        "game_index": game_index,
        "config_id": config_id,
        "seed": config_id,  # Use config_id as seed for determinism
    }


def encode_task_id(game_index: int, config_id: int) -> int:
    """
    Encode game index and config into a task ID.
    
    Args:
        game_index: Index into AVAILABLE_GAMES
        config_id: Configuration/seed within the game
        
    Returns:
        Encoded task ID
    """
    return game_index * GAME_BLOCK_SIZE + config_id


def create_game(task_id: int) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an OpenSpiel game from a task ID.
    
    Args:
        task_id: The encoded task identifier
        
    Returns:
        Tuple of (pyspiel.Game, game_config_dict)
    """
    config = decode_task_id(task_id)
    game_index = config["game_index"]
    
    openspiel_name, params, num_players = AVAILABLE_GAMES[game_index]
    
    # Create the game with parameters
    if params:
        param_str = ",".join(f"{k}={v}" for k, v in params.items())
        game_string = f"{openspiel_name}({param_str})"
    else:
        game_string = openspiel_name
    
    try:
        game = pyspiel.load_game(game_string)
    except Exception as e:
        # Fallback: try without parameters
        try:
            game = pyspiel.load_game(openspiel_name)
        except Exception:
            raise RuntimeError(f"Failed to load game '{openspiel_name}': {e}")
    
    config["openspiel_name"] = openspiel_name
    config["params"] = params
    config["num_players"] = game.num_players()
    
    return game, config


def get_game_info(game_index: int) -> Dict[str, Any]:
    """
    Get information about a game by index.
    
    Args:
        game_index: Index into AVAILABLE_GAMES
        
    Returns:
        Dict with game information
    """
    if game_index < 0 or game_index >= len(AVAILABLE_GAMES):
        raise ValueError(f"Invalid game index: {game_index}")
    
    openspiel_name, params, num_players = AVAILABLE_GAMES[game_index]
    
    return {
        "game_index": game_index,
        "game_name": openspiel_name,
        "openspiel_name": openspiel_name,
        "params": params,
        "num_players": num_players,
    }


def list_available_games() -> List[Dict[str, Any]]:
    """
    List all available games with their configurations.
    
    Returns:
        List of game info dicts
    """
    return [get_game_info(i) for i in range(len(AVAILABLE_GAMES))]
