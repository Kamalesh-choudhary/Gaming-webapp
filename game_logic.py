"""
game_logic.py
=============
Pure game-rule logic for both supported games: Rock-Paper-Scissors (RPS)
and Tic-Tac-Toe (TTT).

Deliberately framework-free: no Flask, no sessions, no globals. Every
function takes plain data in and returns plain data out, which makes this
module trivial to unit test and reuse (e.g. in a CLI version of the game).
"""

from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rock / Paper / Scissors
# ---------------------------------------------------------------------------

RPS_MOVES = ("rock", "paper", "scissors")

# key beats value, e.g. "rock" beats "scissors"
RPS_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def rps_result(player_move: str, ai_move: str) -> str:
    """
    Determine the outcome of one RPS round from the *player's* perspective.
    Returns one of: "win", "lose", "draw".
    """
    if player_move not in RPS_MOVES or ai_move not in RPS_MOVES:
        raise ValueError("Invalid move for Rock-Paper-Scissors")

    if player_move == ai_move:
        return "draw"
    if RPS_BEATS[player_move] == ai_move:
        return "win"
    return "lose"


# ---------------------------------------------------------------------------
# Tic-Tac-Toe
# ---------------------------------------------------------------------------

# Board is represented as a flat list of 9 cells, index 0-8, left-to-right
# top-to-bottom. Each cell is "X", "O", or None.
WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def new_board() -> List[Optional[str]]:
    return [None] * 9


def empty_cells(board: List[Optional[str]]) -> List[int]:
    return [i for i, v in enumerate(board) if v is None]


def is_board_full(board: List[Optional[str]]) -> bool:
    return all(v is not None for v in board)


def check_winner(board: List[Optional[str]]) -> Optional[str]:
    """Return "X" or "O" if that symbol has a winning line, else None."""
    for a, b, c in WIN_LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None


def ttt_game_over(board: List[Optional[str]]) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_over, outcome) where outcome is "X", "O", "draw", or None
    (None only when is_over is False).
    """
    winner = check_winner(board)
    if winner:
        return True, winner
    if is_board_full(board):
        return True, "draw"
    return False, None


def apply_move(board: List[Optional[str]], cell: int, symbol: str) -> List[Optional[str]]:
    """Return a NEW board with `symbol` placed at `cell` (does not mutate input)."""
    if cell not in empty_cells(board):
        raise ValueError(f"Cell {cell} is not a legal move")
    new = list(board)
    new[cell] = symbol
    return new
