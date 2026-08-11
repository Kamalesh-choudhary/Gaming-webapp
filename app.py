"""
app.py
======
Flask application exposing a small, stateless-from-the-client's-perspective
REST API for two pattern-modeling games: Rock-Paper-Scissors and
Tic-Tac-Toe.

Session handling
-----------------
Each game session lives in a server-side dict (`SESSIONS`), keyed by a
session ID the server generates in `/new-game` and hands back to the
client. The client stores that ID (see static/js/app.js) and sends it back
on every subsequent request via the `X-Session-Id` header. This keeps
requests themselves stateless/self-describing (no server-side cookies
required) while still letting multiple concurrent players each get their
own isolated AI model, board, and score -- one player's moves never leak
into another player's opponent model.

Note for production use: an in-memory dict is perfect for a demo/portfolio
project but won't survive a process restart or scale across multiple
worker processes. Swapping in Redis (or any keyed store) behind the same
`SESSIONS`-like interface would make this production-ready.
"""

import uuid
from typing import Optional

from flask import Flask, jsonify, request

from ai_engine import RPSAI, TicTacToeAI
from game_logic import (
    apply_move,
    new_board,
    rps_result,
    ttt_game_over,
)

app = Flask(__name__)

# session_id -> session state dict (see _new_session_state for shape)
SESSIONS: dict = {}

VALID_MODES = ("rps", "tictactoe")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _new_session_state(mode: str) -> dict:
    state = {
        "mode": mode,
        "score": {"player": 0, "ai": 0, "draws": 0},
        "history": [],  # list of round summaries, most recent last
        "game_over": False,
    }
    if mode == "rps":
        state["ai"] = RPSAI()
    else:  # tictactoe
        state["ai"] = TicTacToeAI()
        state["board"] = new_board()
        state["player_symbol"] = "X"
        state["ai_symbol"] = "O"
    return state


def _get_session_id() -> Optional[str]:
    body = request.get_json(silent=True) or {}
    return request.headers.get("X-Session-Id") or request.args.get("session_id") or body.get("session_id")


def _require_session():
    """Look up the session for this request, or return an error response."""
    session_id = _get_session_id()
    if not session_id or session_id not in SESSIONS:
        return None, (jsonify({"error": "Unknown or missing session_id. Call /new-game first."}), 404)
    return SESSIONS[session_id], None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    from flask import render_template
    return render_template("index.html")


@app.route("/new-game", methods=["POST"])
def new_game():
    """
    Start a new game.

    Body: { "mode": "rps" | "tictactoe", "session_id": "<optional existing id>" }

    If `session_id` refers to an existing session for the SAME mode, its
    score and AI model are preserved (so switching screens or refreshing
    doesn't wipe out what the AI has learned about you) but the board and
    round-over flag are reset. Otherwise a brand-new session is created
    with a fresh, untrained AI.
    """
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if mode not in VALID_MODES:
        return jsonify({"error": f"mode must be one of {VALID_MODES}"}), 400

    session_id = body.get("session_id")
    existing = SESSIONS.get(session_id) if session_id else None

    if existing and existing["mode"] == mode:
        existing["game_over"] = False
        if mode == "tictactoe":
            existing["board"] = new_board()
        state = existing
    else:
        session_id = str(uuid.uuid4())
        state = _new_session_state(mode)
        SESSIONS[session_id] = state

    response = {
        "session_id": session_id,
        "mode": mode,
        "score": state["score"],
    }
    if mode == "tictactoe":
        response["board"] = state["board"]
        response["player_symbol"] = state["player_symbol"]
        response["ai_symbol"] = state["ai_symbol"]

    return jsonify(response)


@app.route("/move", methods=["POST"])
def move():
    """
    Play one round / one ply.

    Body: { "session_id": "...", "move": <move> }
      - RPS mode:        move is "rock" | "paper" | "scissors"
      - Tic-Tac-Toe mode: move is an integer cell index 0-8
    """
    state, error = _require_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    player_move = body.get("move")

    if state["game_over"]:
        return jsonify({"error": "This game is already over. Call /new-game to play again."}), 400

    if state["mode"] == "rps":
        return _handle_rps_move(state, player_move)
    else:
        return _handle_ttt_move(state, player_move)


@app.route("/stats", methods=["GET"])
def stats():
    """Return score, move history, and a snapshot of what the AI has learned."""
    state, error = _require_session()
    if error:
        return error

    return jsonify(
        {
            "mode": state["mode"],
            "score": state["score"],
            "history": state["history"],
            "model_stats": state["ai"].stats(),
        }
    )


# ---------------------------------------------------------------------------
# Mode-specific move handlers
# ---------------------------------------------------------------------------

def _handle_rps_move(state: dict, player_move):
    from game_logic import RPS_MOVES

    if player_move not in RPS_MOVES:
        return jsonify({"error": f"move must be one of {RPS_MOVES}"}), 400

    ai: RPSAI = state["ai"]

    # 1. AI commits to a move based on everything it has learned SO FAR
    #    (i.e. it cannot see this round's player move before choosing).
    ai_move, predicted_move, confidence = ai.choose_move()

    # 2. Score the round.
    result = rps_result(player_move, ai_move)
    if result == "win":
        state["score"]["player"] += 1
    elif result == "lose":
        state["score"]["ai"] += 1
    else:
        state["score"]["draws"] += 1

    # 3. NOW let the AI learn from the player's actual move, for next time.
    ai.learn(player_move)

    round_summary = {
        "round": len(state["history"]) + 1,
        "player_move": player_move,
        "ai_move": ai_move,
        "result": result,
        "ai_predicted": predicted_move,
        "ai_confidence": round(confidence, 3),
    }
    state["history"].append(round_summary)

    return jsonify(
        {
            "player_move": player_move,
            "ai_move": ai_move,
            "result": result,
            "score": state["score"],
            "ai_prediction": {"move": predicted_move, "confidence": round(confidence, 3)},
            "round": round_summary["round"],
        }
    )


def _handle_ttt_move(state: dict, player_move):
    board = state["board"]
    player_symbol = state["player_symbol"]
    ai_symbol = state["ai_symbol"]

    if not isinstance(player_move, int) or player_move not in range(9):
        return jsonify({"error": "move must be an integer cell index 0-8"}), 400

    if board[player_move] is not None:
        return jsonify({"error": "That cell is already taken"}), 400

    ai: TicTacToeAI = state["ai"]

    # Player moves first this ply.
    board = apply_move(board, player_move, player_symbol)
    ai.learn(player_move)
    state["board"] = board

    over, outcome = ttt_game_over(board)
    ai_move = None
    ai_prediction = None

    if not over:
        # AI picks its move (minimax, ties broken by pattern prediction).
        ai_move, predicted_cell, confidence = ai.choose_move(board, ai_symbol, player_symbol)
        board = apply_move(board, ai_move, ai_symbol)
        state["board"] = board
        ai_prediction = {"move": predicted_cell, "confidence": round(confidence, 3)}
        over, outcome = ttt_game_over(board)

    if over:
        state["game_over"] = True
        if outcome == player_symbol:
            state["score"]["player"] += 1
            result = "win"
        elif outcome == ai_symbol:
            state["score"]["ai"] += 1
            result = "lose"
        else:
            state["score"]["draws"] += 1
            result = "draw"
    else:
        result = "in_progress"

    round_summary = {
        "round": len(state["history"]) + 1,
        "player_move": player_move,
        "ai_move": ai_move,
        "result": result,
        "board": state["board"],
    }
    state["history"].append(round_summary)

    return jsonify(
        {
            "player_move": player_move,
            "ai_move": ai_move,
            "board": state["board"],
            "result": result,
            "game_over": state["game_over"],
            "score": state["score"],
            "ai_prediction": ai_prediction,
            "round": round_summary["round"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
