"""
ai_engine.py
============
The "brain" of the project: a reusable statistical opponent-modeling engine,
plus two AI players (Rock-Paper-Scissors and Tic-Tac-Toe) built on top of it.

--------------------------------------------------------------------------
How the opponent modeling works
--------------------------------------------------------------------------
`PatternAnalyzer` watches a stream of the human player's moves and blends
two classic prediction techniques:

1. Frequency analysis
   The simplest possible model: "what has the player played the most,
   overall?" Kept as a baseline / fallback for when there isn't enough
   history yet for a sequential model to be reliable.

2. Markov chain / n-gram analysis
   Humans are bad at being random -- they fall into habits like
   "always switch after losing" or "repeat the last move after a win".
   The analyzer keeps a table of `P(next_move | last N moves)` and updates
   it after every round. With N=2 (the default) it is effectively
   learning short 3-move sequences the player tends to repeat.

The two predictions are blended together, with the Markov chain's
influence increasing as more history accumulates for the *current*
context (i.e. once the AI has actually seen the player's last two moves
before, it trusts the sequential pattern more than the raw frequency).

An exponential decay factor (`decay < 1`) is applied to old observations
every time a new one is recorded, so the model adapts if a player changes
their strategy mid-game instead of being anchored to old habits forever.

Both game AIs are intentionally NOT unbeatable in the same way:
- RPSAI actively uses the prediction to counter the player, with a small
  exploration rate so it isn't itself trivially exploitable.
- TicTacToeAI plays *optimal* minimax (with alpha-beta pruning) so it can
  never lose -- pattern modeling can't out-strategize perfect information
  tic-tac-toe. Instead, the pattern analyzer is used to break ties between
  multiple equally-good minimax moves, biasing the AI towards the square
  the player is statistically most likely to want next. This is disclosed
  honestly in the README rather than oversold as "the AI reads your mind
  in tic-tac-toe".
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

from game_logic import (
    RPS_BEATS,
    RPS_MOVES,
    check_winner,
    empty_cells,
    is_board_full,
)


# ---------------------------------------------------------------------------
# Generic statistical pattern analyzer
# ---------------------------------------------------------------------------

class PatternAnalyzer:
    """
    Tracks a sequence of discrete "moves" from some fixed alphabet and
    predicts the next one using a frequency + Markov-chain blend.

    Generic over the alphabet so the same class powers both the RPS AI
    (alphabet = {"rock", "paper", "scissors"}) and the Tic-Tac-Toe AI
    (alphabet = {0, 1, ..., 8}).
    """

    def __init__(self, alphabet: Sequence[Hashable], order: int = 2, decay: float = 0.98):
        """
        alphabet: every possible move value
        order:    how many previous moves the Markov chain conditions on
        decay:    multiplier (<=1) applied to old counts on every new
                  observation, so recent behaviour is weighted more heavily
        """
        self.alphabet: Tuple[Hashable, ...] = tuple(alphabet)
        self.order = order
        self.decay = decay

        self.history: List[Hashable] = []
        self.frequency: Counter = Counter({s: 0.0 for s in self.alphabet})
        # markov[context][next_move] = weighted count
        self.markov: Dict[Tuple[Hashable, ...], Counter] = defaultdict(
            lambda: Counter({s: 0.0 for s in self.alphabet})
        )

    def record(self, move: Hashable) -> None:
        """Register a move the player actually made."""
        if move not in self.alphabet:
            raise ValueError(f"{move!r} is not in this analyzer's alphabet")

        # Decay all frequency counts slightly before adding the new one,
        # so older observations gradually matter less.
        for key in self.frequency:
            self.frequency[key] *= self.decay
        self.frequency[move] += 1.0

        if len(self.history) >= self.order:
            context = tuple(self.history[-self.order:])
            counter = self.markov[context]
            for key in counter:
                counter[key] *= self.decay
            counter[move] += 1.0

        self.history.append(move)

    def predict(self) -> Tuple[Hashable, float, Dict[Hashable, float]]:
        """
        Predict the player's next move.
        Returns (predicted_move, confidence 0-1, full_probability_distribution)
        """
        n = len(self.history)
        if n == 0:
            uniform = {s: 1.0 / len(self.alphabet) for s in self.alphabet}
            return random.choice(self.alphabet), 0.0, uniform

        freq_total = sum(self.frequency.values()) or 1.0
        freq_dist = {s: self.frequency[s] / freq_total for s in self.alphabet}

        markov_dist = None
        if n >= self.order:
            context = tuple(self.history[-self.order:])
            counter = self.markov.get(context)
            if counter:
                m_total = sum(counter.values())
                if m_total > 0:
                    markov_dist = {s: counter[s] / m_total for s in self.alphabet}

        if markov_dist is not None:
            # Trust the sequential model more as more history accumulates,
            # capped so frequency data always has *some* influence.
            markov_weight = min(0.75, 0.25 + n * 0.03)
            blended = {
                s: markov_weight * markov_dist[s] + (1 - markov_weight) * freq_dist[s]
                for s in self.alphabet
            }
        else:
            blended = freq_dist

        predicted = max(blended, key=blended.get)
        confidence = blended[predicted]
        return predicted, confidence, blended

    def stats(self) -> dict:
        """Human-readable snapshot of what the model has learned so far."""
        total = sum(self.frequency.values()) or 1.0
        return {
            "total_moves": len(self.history),
            "frequency_pct": {
                str(s): round(self.frequency[s] / total * 100, 1) for s in self.alphabet
            },
        }


# ---------------------------------------------------------------------------
# Rock-Paper-Scissors AI
# ---------------------------------------------------------------------------

# counter_move[predicted] = the move that beats `predicted`
_RPS_COUNTER = {beaten: beater for beater, beaten in RPS_BEATS.items()}


class RPSAI:
    """AI opponent for Rock-Paper-Scissors, driven entirely by pattern modeling."""

    def __init__(self, order: int = 2, exploration_rate: float = 0.12):
        self.analyzer = PatternAnalyzer(RPS_MOVES, order=order)
        # Small chance of playing a random move regardless of prediction,
        # so a player can't reverse-engineer a fully deterministic AI.
        self.exploration_rate = exploration_rate

    def choose_move(self) -> Tuple[str, str, float]:
        """
        Returns (ai_move, predicted_player_move, confidence).
        Called BEFORE the player's move for this round is known.
        """
        predicted, confidence, _dist = self.analyzer.predict()

        if confidence == 0.0 or random.random() < self.exploration_rate:
            ai_move = random.choice(RPS_MOVES)
        else:
            ai_move = _RPS_COUNTER[predicted]

        return ai_move, predicted, confidence

    def learn(self, player_move: str) -> None:
        self.analyzer.record(player_move)

    def stats(self) -> dict:
        return self.analyzer.stats()


# ---------------------------------------------------------------------------
# Tic-Tac-Toe AI
# ---------------------------------------------------------------------------

class TicTacToeAI:
    """
    Optimal (unbeatable) Tic-Tac-Toe AI via minimax + alpha-beta pruning.
    Ties between equally-good moves are broken using the same
    PatternAnalyzer engine, tracking which cell the player tends to play
    next -- so the AI "feels" adaptive even though its win-rate floor is
    guaranteed by minimax, not by the pattern model.
    """

    def __init__(self, order: int = 1):
        self.analyzer = PatternAnalyzer(list(range(9)), order=order)

    def learn(self, cell: int) -> None:
        self.analyzer.record(cell)

    def choose_move(
        self, board: List[str], ai_symbol: str, player_symbol: str
    ) -> Tuple[int, int, float]:
        """
        Returns (chosen_cell, predicted_player_cell, confidence).
        """
        best_score = -math.inf
        best_moves: List[int] = []

        for cell in empty_cells(board):
            trial = list(board)
            trial[cell] = ai_symbol
            score = self._minimax(trial, False, ai_symbol, player_symbol, 1, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_moves = [cell]
            elif score == best_score:
                best_moves.append(cell)

        predicted_cell, confidence, _dist = self.analyzer.predict()

        if len(best_moves) > 1 and predicted_cell in best_moves:
            chosen = predicted_cell
        else:
            chosen = random.choice(best_moves)

        return chosen, predicted_cell, confidence

    def _minimax(
        self,
        board: List[str],
        is_maximizing: bool,
        ai_symbol: str,
        player_symbol: str,
        depth: int,
        alpha: float,
        beta: float,
    ) -> int:
        winner = check_winner(board)
        if winner == ai_symbol:
            return 10 - depth
        if winner == player_symbol:
            return depth - 10
        if is_board_full(board):
            return 0

        if is_maximizing:
            best = -math.inf
            for cell in empty_cells(board):
                trial = list(board)
                trial[cell] = ai_symbol
                best = max(
                    best,
                    self._minimax(trial, False, ai_symbol, player_symbol, depth + 1, alpha, beta),
                )
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = math.inf
            for cell in empty_cells(board):
                trial = list(board)
                trial[cell] = player_symbol
                best = min(
                    best,
                    self._minimax(trial, True, ai_symbol, player_symbol, depth + 1, alpha, beta),
                )
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def stats(self) -> dict:
        return self.analyzer.stats()
