#!/usr/bin/env python3
"""
Game-specific agent classes for OpenSpiel games.

Each agent provides:
- System prompt generation (game rules explanation)
- User prompt generation (current state + legal actions)
- MCTS configuration for opponent bots
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseGameAgent(ABC):
    """Base class for game-specific agents."""
    
    @abstractmethod
    def generate_system_prompt(self) -> str:
        """Generate the system prompt explaining the game rules."""
        pass
    
    @abstractmethod
    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        """Generate the user prompt for the current game state."""
        pass
    
    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        """
        Get MCTS configuration for opponent bot.
        
        Returns:
            Tuple of (max_simulations, n_rollouts) or None if MCTS not supported
        """
        return (100, 1)  # Default: 100 simulations, 1 rollout per evaluation
    
    def format_action(self, action: int, state: Any) -> str:
        """Format an action for display."""
        try:
            return state.action_to_string(action)
        except:
            return str(action)


class GoofspielAgent(BaseGameAgent):
    """Agent for Goofspiel (bidding card game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Goofspiel, a bidding card game.

RULES:
- Each player has a hand of bid cards (numbered 1 to N).
- Prize cards are revealed one at a time.
- Players simultaneously bid using one card from their hand.
- Highest bid wins the prize card (ties: prize discarded).
- Goal: Win prizes with highest total value.

STRATEGY:
- Bid high on valuable prizes.
- Save high cards for important prizes.
- Consider opponent's remaining cards.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class LiarsDiceAgent(BaseGameAgent):
    """Agent for Liar's Dice (bluffing dice game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Liar's Dice, a bluffing game.

RULES:
- Each player rolls dice secretly.
- Players take turns making claims about total dice showing a face value.
- Claims must increase (higher quantity or higher face value).
- Call "Liar" to challenge the previous claim.
- If challenged: claim is checked against ALL dice.
  - Claim true: challenger loses a die.
  - Claim false: claimer loses a die.
- Last player with dice wins.

STRATEGY:
- Bluff strategically based on your dice.
- Track probability of claims being true.
- Challenge suspicious claims.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""

    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        return (3000, 200)  # Strong MCTS opponent for Liar's Dice


class LeducPokerAgent(BaseGameAgent):
    """Agent for Leduc Poker (simplified poker)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Leduc Poker, a simplified poker game.

RULES:
- Deck: 6 cards (2 Jacks, 2 Queens, 2 Kings).
- Each player gets 1 private card.
- Betting round, then 1 community card revealed.
- Another betting round, then showdown.
- Pair beats high card. Higher rank wins ties.

ACTIONS:
- Fold: Give up the hand.
- Call: Match the current bet.
- Raise: Increase the bet.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class GinRummyAgent(BaseGameAgent):
    """Agent for Gin Rummy (card game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Gin Rummy, a card matching game.

RULES:
- Goal: Form melds (sets of 3+ same rank, or runs of 3+ same suit).
- Draw from deck or discard pile, then discard one card.
- "Knock" when unmatched cards (deadwood) total ≤10 points.
- "Gin" = no deadwood (bonus points).
- Face cards = 10 points, others = face value.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""
    
    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        return (50, 1)  # Fewer simulations for slower game


class OthelloAgent(BaseGameAgent):
    """Agent for Othello/Reversi (board game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Othello (Reversi), a strategy board game.

RULES:
- 8x8 board, players are Black (X) and White (O).
- Place a disc to outflank opponent's discs.
- Outflanked discs flip to your color.
- Must make a valid move if possible.
- Game ends when neither player can move.
- Most discs wins.

STRATEGY:
- Control corners (can't be flipped).
- Avoid giving corners to opponent.
- Maximize stable discs (edges, corners).

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""
    
    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        return (50, 1)  # Fewer simulations for slower game


class BackgammonAgent(BaseGameAgent):
    """Agent for Backgammon (board game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Backgammon, a classic board game.

RULES:
- Move checkers based on dice rolls.
- Goal: Bear off all 15 checkers first.
- Can hit opponent's single checker (blot).
- Hit checkers go to the bar, must re-enter.
- Can only bear off when all checkers in home board.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""
    
    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        return (30, 1)  # Fewer simulations for very slow game


class HexAgent(BaseGameAgent):
    """Agent for Hex (connection board game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Hex, a connection board game.

RULES:
- Players take turns placing stones on hexagonal grid.
- Player 1 (X): Connect top to bottom.
- Player 2 (O): Connect left to right.
- No draws possible - someone always wins.

STRATEGY:
- Control the center.
- Build connected groups.
- Block opponent's connections.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class ClobberAgent(BaseGameAgent):
    """Agent for Clobber (capture board game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Clobber, a capture board game.

RULES:
- Grid with alternating black and white stones.
- Move: Capture adjacent opponent stone (orthogonally).
- Your stone replaces the captured stone.
- Player who cannot move loses.

STRATEGY:
- Maintain mobility (keep move options).
- Trap opponent's stones.
- Control key positions.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class HeartsAgent(BaseGameAgent):
    """Agent for Hearts (trick-taking card game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Hearts, a trick-taking card game.

RULES:
- 4 players, standard 52-card deck.
- Goal: Avoid points (hearts=1pt, Queen of Spades=13pts).
- Must follow suit if possible.
- Highest card of led suit wins trick.
- "Shooting the Moon": Take ALL hearts + QoS = 0 pts, others get 26.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class EuchreAgent(BaseGameAgent):
    """Agent for Euchre (trick-taking card game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Euchre, a trick-taking card game.

RULES:
- 4 players in 2 teams, 24-card deck (9-A).
- Trump suit determined by bidding.
- Jack of trump (Right Bower) is highest.
- Jack of same color (Left Bower) is second highest.
- Win tricks to score points for your team.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class DotsAndBoxesAgent(BaseGameAgent):
    """Agent for Dots and Boxes (connection game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Dots and Boxes, a connection game.

RULES:
- Grid of dots, players take turns drawing lines.
- Complete a box = score 1 point + extra turn.
- Game ends when all boxes completed.
- Most boxes wins.

STRATEGY:
- Avoid giving opponent chains of boxes.
- Force opponent to open chains.
- Count and plan ahead.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""


class GoAgent(BaseGameAgent):
    """Agent for Go (classic board game)."""
    
    def generate_system_prompt(self) -> str:
        return """You are playing Go, a classic strategy board game.

RULES:
- Place stones to control territory.
- Surround opponent's stones to capture them.
- Stones need liberties (adjacent empty points) to survive.
- Game ends when both pass.
- Score = territory + captures - komi (white's compensation).

STRATEGY:
- Balance territory and influence.
- Keep groups connected and alive.
- Attack weak groups.

OUTPUT FORMAT: Respond with ONLY the action number, nothing else."""

    def generate_user_prompt(self, state: Any, player_id: int, legal_actions: List[int]) -> str:
        state_str = str(state)
        actions_str = ", ".join(str(a) for a in legal_actions)
        return f"""Current state:
{state_str}

You are Player {player_id}.
Legal actions: [{actions_str}]

Choose your action (respond with just the number):"""
    
    def get_mcts_config(self) -> Optional[Tuple[int, int]]:
        return (50, 1)  # Fewer simulations for complex game


# Registry mapping game names to agent classes
GAME_AGENTS: Dict[str, type] = {
    "goofspiel": GoofspielAgent,
    "liars_dice": LiarsDiceAgent,
    "leduc_poker": LeducPokerAgent,
    "gin_rummy": GinRummyAgent,
    "othello": OthelloAgent,
    "backgammon": BackgammonAgent,
    "hex": HexAgent,
    "clobber": ClobberAgent,
    "hearts": HeartsAgent,
    "euchre": EuchreAgent,
    "dots_and_boxes": DotsAndBoxesAgent,
    "go": GoAgent,
}


def get_agent(game_name: str) -> BaseGameAgent:
    """
    Get an agent instance for the specified game.
    
    Args:
        game_name: Name of the game
        
    Returns:
        Agent instance for the game
    """
    agent_class = GAME_AGENTS.get(game_name)
    if agent_class is None:
        raise ValueError(f"No agent found for game: {game_name}")
    return agent_class()
