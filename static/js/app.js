/**
 * app.js
 * ------
 * All client-side logic. No page reloads: every action talks to the Flask
 * API via fetch() and re-renders the affected DOM fragments.
 *
 * Session handling: the server generates a session_id per game mode and
 * hands it back on /new-game. We persist one session_id per mode in
 * localStorage so refreshing the page (or switching modes and back)
 * resumes the same score + trained AI instead of starting over.
 *
 * Motion: the one signature moment is the RPS "duel stage" -- your pick
 * and the AI's counter fly in and clash, synced with the prediction panel
 * flipping from "reading" to "locked" a beat before impact. Tic-Tac-Toe
 * mirrors that same reveal rhythm (stamp your move in, brief AI "think"
 * pause, stamp its move in) so both modes share one motion language.
 */

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const REVEAL_DELAY = REDUCED_MOTION ? 0 : 280; // artificial "AI is reading you" beat
const DUEL_SETTLE_DELAY = REDUCED_MOTION ? 0 : 680; // matches the CSS clash animation length

const RPS_EMOJI = { rock: "🪨", paper: "📄", scissors: "✂️" };
const TTT_WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

const API = {
  async newGame(mode, sessionId) {
    return request("/new-game", { mode, session_id: sessionId || undefined });
  },
  async move(sessionId, move) {
    return request("/move", { session_id: sessionId, move });
  },
  async stats(sessionId) {
    const res = await fetch(`/stats?session_id=${encodeURIComponent(sessionId)}`, {
      headers: { "X-Session-Id": sessionId },
    });
    return res.json();
  },
};

async function request(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(body.session_id ? { "X-Session-Id": body.session_id } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Request failed" }));
    throw new Error(err.error || "Request failed");
  }
  return res.json();
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const STORAGE_KEY = "opponent_modeling_sessions";
let sessionIds = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
let currentMode = "rps";
let tttGameOver = false;
let tttBoard = Array(9).fill(null);

function persistSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessionIds));
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const el = {
  modeRps: document.getElementById("mode-rps"),
  modeTtt: document.getElementById("mode-ttt"),
  toggleIndicator: document.getElementById("toggle-indicator"),
  rpsBoard: document.getElementById("rps-board"),
  tttBoard: document.getElementById("ttt-board"),
  rpsButtons: document.querySelectorAll(".rps-btn"),
  rpsResult: document.getElementById("rps-result"),
  duelStage: document.getElementById("duel-stage"),
  duelPlayer: document.getElementById("duel-player"),
  duelAi: document.getElementById("duel-ai"),
  tttGrid: document.getElementById("ttt-grid"),
  tttResult: document.getElementById("ttt-result"),
  scorePlayer: document.getElementById("score-player"),
  scoreAi: document.getElementById("score-ai"),
  scoreDraws: document.getElementById("score-draws"),
  predictBox: document.getElementById("predict-box"),
  predictMove: document.getElementById("predict-move"),
  predictConfidence: document.getElementById("predict-confidence"),
  confidenceFill: document.getElementById("confidence-fill"),
  predictCaption: document.getElementById("predict-caption"),
  freqBars: document.getElementById("freq-bars"),
  historyLog: document.getElementById("history-log"),
  totalMoves: document.getElementById("total-moves"),
  newGameBtn: document.getElementById("new-game-btn"),
  sessionIdDisplay: document.getElementById("session-id-display"),
};

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

el.modeRps.addEventListener("click", () => switchMode("rps"));
el.modeTtt.addEventListener("click", () => switchMode("tictactoe"));
window.addEventListener("resize", () => positionToggleIndicator(currentMode === "rps" ? el.modeRps : el.modeTtt));

function positionToggleIndicator(activeBtn) {
  const toggle = activeBtn.parentElement;
  const toggleRect = toggle.getBoundingClientRect();
  const btnRect = activeBtn.getBoundingClientRect();
  el.toggleIndicator.style.width = `${btnRect.width}px`;
  el.toggleIndicator.style.transform = `translateX(${btnRect.left - toggleRect.left - 4}px)`;
}

async function switchMode(mode) {
  currentMode = mode;
  el.modeRps.classList.toggle("active", mode === "rps");
  el.modeRps.setAttribute("aria-selected", mode === "rps");
  el.modeTtt.classList.toggle("active", mode === "tictactoe");
  el.modeTtt.setAttribute("aria-selected", mode === "tictactoe");
  positionToggleIndicator(mode === "rps" ? el.modeRps : el.modeTtt);

  el.rpsBoard.classList.toggle("hidden", mode !== "rps");
  el.tttBoard.classList.toggle("hidden", mode !== "tictactoe");
  el.newGameBtn.textContent = mode === "rps" ? "Reset Match" : "New Round";

  await startGame(mode, sessionIds[mode]);
}

// ---------------------------------------------------------------------------
// Starting / resetting a game
// ---------------------------------------------------------------------------

el.newGameBtn.addEventListener("click", async () => {
  // RPS "Reset Match" wipes the session entirely (fresh AI + score 0).
  // TTT "New Round" reuses the session, keeping score + trained model.
  const keepSession = currentMode === "tictactoe";
  await startGame(currentMode, keepSession ? sessionIds[currentMode] : null);
});

async function startGame(mode, sessionId) {
  const data = await API.newGame(mode, sessionId);
  sessionIds[mode] = data.session_id;
  persistSessions();
  el.sessionIdDisplay.textContent = data.session_id.slice(0, 8);

  renderScore(data.score, false);
  resetPrediction();
  el.historyLog.innerHTML = "";
  el.totalMoves.textContent = "0";
  renderFrequency({}, mode);

  if (mode === "rps") {
    el.rpsResult.innerHTML = '<span class="text-muted text-sm">Make the first move — the model has nothing to read yet.</span>';
    setRpsButtonsEnabled(true);
    el.duelStage.classList.remove("fly");
    el.duelPlayer.classList.remove("winner", "loser");
    el.duelAi.classList.remove("winner", "loser");
  } else {
    tttGameOver = false;
    tttBoard = Array(9).fill(null);
    renderTttBoard(tttBoard);
    el.tttResult.innerHTML = "<span class=\"text-muted text-sm\">You're X, first move is yours.</span>";
  }
}

// ---------------------------------------------------------------------------
// Rock-Paper-Scissors
// ---------------------------------------------------------------------------

el.rpsButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    setRpsButtonsEnabled(false);
    lockPrediction(false); // enter "scanning" state while we wait
    try {
      const [data] = await Promise.all([API.move(sessionIds.rps, btn.dataset.move), wait(REVEAL_DELAY)]);

      renderPrediction(data.ai_prediction, true);
      playDuel(data.player_move, data.ai_move, data.result);
      await wait(DUEL_SETTLE_DELAY);

      renderScore(data.score);
      renderRpsResult(data);
      await refreshStats("rps");
    } catch (e) {
      el.rpsResult.innerHTML = `<span class="text-lose">${e.message}</span>`;
    } finally {
      setRpsButtonsEnabled(true);
    }
  });
});

function setRpsButtonsEnabled(enabled) {
  el.rpsButtons.forEach((b) => (b.disabled = !enabled));
}

function playDuel(playerMove, aiMove, result) {
  el.duelPlayer.querySelector(".duel-emoji").textContent = RPS_EMOJI[playerMove];
  el.duelAi.querySelector(".duel-emoji").textContent = RPS_EMOJI[aiMove];

  el.duelPlayer.classList.remove("winner", "loser");
  el.duelAi.classList.remove("winner", "loser");
  el.duelStage.classList.remove("fly");
  void el.duelStage.offsetWidth; // force reflow so the animation replays every round
  el.duelStage.classList.add("fly");

  setTimeout(() => {
    if (result === "win") {
      el.duelPlayer.classList.add("winner");
      el.duelAi.classList.add("loser");
    } else if (result === "lose") {
      el.duelAi.classList.add("winner");
      el.duelPlayer.classList.add("loser");
    }
  }, REDUCED_MOTION ? 0 : 460);
}

function renderRpsResult(data) {
  const label = { win: "You win", lose: "AI wins", draw: "Draw" }[data.result];
  const cls = { win: "result-win", lose: "result-lose", draw: "result-draw" }[data.result];
  el.rpsResult.innerHTML = `
    <span class="${cls}">${label}</span>
    <span class="text-muted"> — you played <strong class="text-cool">${data.player_move}</strong>,
    AI played <strong class="text-signal">${data.ai_move}</strong></span>
  `;
}

// ---------------------------------------------------------------------------
// Tic-Tac-Toe
// ---------------------------------------------------------------------------

function renderTttBoard(board, { stampCells = [], winCells = [] } = {}) {
  el.tttGrid.innerHTML = "";
  board.forEach((val, i) => {
    const cell = document.createElement("div");
    const classes = ["ttt-cell"];
    if (val) classes.push("filled", val === "X" ? "x" : "o");
    if (stampCells.includes(i)) classes.push("stamp");
    if (winCells.includes(i)) classes.push("win-cell");
    cell.className = classes.join(" ");
    cell.textContent = val || "";
    cell.dataset.cell = i;
    cell.tabIndex = !val && !tttGameOver ? 0 : -1;
    if (!val && !tttGameOver) {
      cell.setAttribute("role", "button");
      cell.setAttribute("aria-label", `Play cell ${i}`);
      cell.addEventListener("click", () => playTttCell(i), { once: true });
      cell.addEventListener(
        "keydown",
        (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            playTttCell(i);
          }
        },
        { once: true }
      );
    }
    el.tttGrid.appendChild(cell);
  });
}

function findWinningLine(board) {
  for (const line of TTT_WIN_LINES) {
    const [a, b, c] = line;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return line;
  }
  return null;
}

async function playTttCell(cell) {
  // Phase 1: stamp the player's move in immediately, board locked while the AI "thinks".
  tttBoard[cell] = "X";
  renderTttBoard(tttBoard, { stampCells: [cell] });
  el.tttGrid.classList.add("thinking");
  lockPrediction(false);

  try {
    const [data] = await Promise.all([API.move(sessionIds.tictactoe, cell), wait(REVEAL_DELAY)]);

    tttBoard = data.board;
    el.tttGrid.classList.remove("thinking");

    if (data.ai_prediction) renderPrediction(data.ai_prediction, true);
    else resetPrediction();

    // Phase 2: reveal only the AI's new mark (the player's cell already landed).
    const aiCell = data.ai_move !== null && data.ai_move !== undefined ? [data.ai_move] : [];
    renderTttBoard(tttBoard, { stampCells: aiCell });
    renderScore(data.score);

    if (data.game_over) {
      tttGameOver = true;
      const winningSymbol = data.result === "win" ? "X" : data.result === "lose" ? "O" : null;
      if (winningSymbol) {
        await wait(REDUCED_MOTION ? 0 : 320);
        const line = findWinningLine(tttBoard);
        if (line) renderTttBoard(tttBoard, { winCells: line });
      }
      const label = { win: "You win! 🎉", lose: "AI wins", draw: "Draw" }[data.result];
      const cls = { win: "result-win", lose: "result-lose", draw: "result-draw" }[data.result];
      el.tttResult.innerHTML = `<span class="${cls}">${label}</span> <span class="text-muted">— press New Round to play again</span>`;
    } else {
      el.tttResult.innerHTML = `<span class="text-muted text-sm">AI played cell <strong class="text-signal">${data.ai_move}</strong>. Your move.</span>`;
    }
    await refreshStats("tictactoe");
  } catch (e) {
    el.tttGrid.classList.remove("thinking");
    el.tttResult.innerHTML = `<span class="text-lose">${e.message}</span>`;
  }
}

// ---------------------------------------------------------------------------
// Shared readout rendering
// ---------------------------------------------------------------------------

function renderScore(score, animate = true) {
  updateScoreNum(el.scorePlayer, score.player, animate);
  updateScoreNum(el.scoreAi, score.ai, animate);
  updateScoreNum(el.scoreDraws, score.draws, animate);
}

function updateScoreNum(node, value, animate) {
  const changed = Number(node.textContent) !== value;
  node.textContent = value;
  if (animate && changed && !REDUCED_MOTION) {
    node.classList.remove("pulse");
    void node.offsetWidth;
    node.classList.add("pulse");
  }
}

function lockPrediction(locked) {
  el.predictBox.classList.toggle("scanning", !locked);
  if (!locked) {
    el.predictCaption.textContent = "reading your pattern…";
  }
}

function renderPrediction(prediction, locked = false) {
  if (!prediction) return resetPrediction();
  const pct = Math.round(prediction.confidence * 100);
  el.predictMove.textContent = typeof prediction.move === "number" ? `cell ${prediction.move}` : prediction.move;
  el.predictConfidence.textContent = `${pct}%`;
  el.confidenceFill.style.width = `${pct}%`;
  el.predictCaption.innerHTML = "the AI's read on your <em>next</em> move, before it commits to its own";
  el.predictBox.classList.toggle("scanning", !locked);
  if (locked && !REDUCED_MOTION) {
    el.predictBox.classList.remove("locked");
    void el.predictBox.offsetWidth;
    el.predictBox.classList.add("locked");
  }
}

function resetPrediction() {
  el.predictBox.classList.remove("scanning", "locked");
  el.predictMove.textContent = "—";
  el.predictConfidence.textContent = "0%";
  el.confidenceFill.style.width = "0%";
  el.predictCaption.innerHTML = "the AI's read on your <em>next</em> move, before it commits to its own";
}

function renderFrequency(freqPct, mode) {
  el.freqBars.innerHTML = "";
  const entries = Object.entries(freqPct);
  if (entries.length === 0) {
    el.freqBars.innerHTML = '<p class="text-xs text-muted">No data yet.</p>';
    return;
  }
  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([key, pct], i) => {
      const label = mode === "tictactoe" ? `cell ${key}` : key;
      const row = document.createElement("div");
      row.className = "freq-row";
      row.innerHTML = `
        <span class="freq-name">${label}</span>
        <span class="freq-track"><span class="freq-fill" style="width:0%;transition-delay:${i * 40}ms"></span></span>
        <span class="freq-pct">${pct}%</span>
      `;
      el.freqBars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".freq-fill").style.width = `${pct}%`;
      });
    });
}

function renderHistory(history, mode) {
  el.historyLog.innerHTML = "";
  history
    .slice()
    .reverse()
    .slice(0, 25)
    .forEach((round, i) => {
      const row = document.createElement("div");
      const cls = round.result === "win" ? "win" : round.result === "lose" ? "lose" : round.result === "draw" ? "draw" : "";
      row.className = `history-row ${cls}${i === 0 ? " enter" : ""}`;
      const playerMoveLabel = mode === "tictactoe" ? `#${round.player_move}` : round.player_move;
      row.innerHTML = `<span>r${round.round} · you: ${playerMoveLabel}</span><span>${round.result}</span>`;
      el.historyLog.appendChild(row);
    });
}

async function refreshStats(mode) {
  const data = await API.stats(sessionIds[mode]);
  renderFrequency(data.model_stats.frequency_pct, mode);
  renderHistory(data.history, mode);
  el.totalMoves.textContent = data.model_stats.total_moves;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

positionToggleIndicator(el.modeRps);
startGame("rps", sessionIds.rps);
