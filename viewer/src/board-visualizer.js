// board-visualizer.js
// Prettier settings: printWidth=80

const GRID_SIZE = 9;
const CELL_SIZE = 20; // pixels per grid cell
const BORDER_SIZE = 15; // pixels for border around the grid
const STONE_RADIUS = CELL_SIZE * 0.45;
const LINE_COLOR = "#444"; // Light gray for grid lines
const BOARD_COLOR = "#000"; // Black background
const BOARD_GAP = 10;
const BOARD_PIXELS = (GRID_SIZE - 1) * CELL_SIZE + BORDER_SIZE * 2;
const COLORS = { 1: "#000", 2: "#fff" }; // 1=black, 2=white

class Board {
  constructor(container) {
    this.state = new Uint8Array(GRID_SIZE * GRID_SIZE);
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = BOARD_PIXELS;
    this.ctx = this.canvas.getContext("2d");
    container.appendChild(this.canvas);
    this.needsDraw = true; // Initially draw the board
  }

  applyMove(x, y, color) {
    const idx = y * GRID_SIZE + x;
    if (this.state[idx] === color) return;
    this.state[idx] = color;
    this.needsDraw = true;
  }

  drawGrid() {
    const ctx = this.ctx;

    // Fill background
    ctx.fillStyle = BOARD_COLOR;
    ctx.fillRect(0, 0, BOARD_PIXELS, BOARD_PIXELS);

    // Draw grid lines
    ctx.strokeStyle = LINE_COLOR;
    ctx.lineWidth = 2; // Increased from 1
    ctx.translate(BORDER_SIZE, BORDER_SIZE); // Add border padding

    const gridEnd = (GRID_SIZE - 1) * CELL_SIZE;
    for (let i = 0; i < GRID_SIZE; i++) {
      const p = i * CELL_SIZE;
      ctx.beginPath();
      // vertical lines
      ctx.moveTo(p, 0);
      ctx.lineTo(p, gridEnd);
      // horizontal lines
      ctx.moveTo(0, p);
      ctx.lineTo(gridEnd, p);
      ctx.stroke();
    }
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Draw board outline
    ctx.strokeStyle = LINE_COLOR;
    ctx.lineWidth = 3; // Increased from 2
    ctx.strokeRect(1, 1, BOARD_PIXELS - 2, BOARD_PIXELS - 2);
  }

  drawStones() {
    const ctx = this.ctx;
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const c = this.state[y * GRID_SIZE + x];
        if (c === 0) continue;

        const px = BORDER_SIZE + x * CELL_SIZE;
        const py = BORDER_SIZE + y * CELL_SIZE;

        ctx.beginPath();
        ctx.arc(px, py, STONE_RADIUS, 0, 2 * Math.PI);
        ctx.fillStyle = COLORS[c];
        ctx.fill();

        // Add a border to black stones to make them visible
        if (c === 1) {
          ctx.strokeStyle = "#555"; // Dark gray border for black stones
          ctx.lineWidth = 2; // Increased from 1
          ctx.stroke();
        }
      }
    }
  }
  draw() {
    if (!this.needsDraw) return;

    this.drawGrid();
    this.drawStones();

    this.needsDraw = false;
  }
}

export function initVisualizer({ container, boardCount, wsUrl }) {
  // Create a dedicated grid container
  const gridContainer = document.createElement("div");
  container.appendChild(gridContainer);

  // 1) Setup container style
  const numCols = Math.ceil(Math.sqrt(boardCount));
  const gridWidth = numCols * (BOARD_PIXELS + BOARD_GAP) - BOARD_GAP;
  gridContainer.style.display = "grid";
  gridContainer.style.gridTemplateColumns = `repeat(${numCols}, 1fr)`;
  gridContainer.style.gap = `${BOARD_GAP}px`;
  gridContainer.style.width = `${gridWidth}px`;

  // 2) Instantiate all boards
  const boards = Array.from(
    { length: boardCount },
    () => new Board(gridContainer),
  );

  // 3) Connect to WebSocket for game updates
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket connection established. Requesting initial state.");
    // Request the full game state upon connecting
    ws.send(JSON.stringify({ type: "get_initial_state" }));
  };

  ws.onerror = (err) => console.error("WebSocket error:", err);
  ws.onclose = () => console.log("WebSocket connection closed");

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    const { type, payload } = msg;

    if (type === "initial_state") {
      // The server sent the full state of all boards
      console.log("Received initial state. Applying to all boards.");
      payload.forEach((boardState, board_idx) => {
        if (board_idx < boards.length) {
          // Directly set the board's state
          boards[board_idx].state = new Uint8Array(boardState);
          boards[board_idx].needsDraw = true;
        }
      });
    } else if (type === "move_update") {
      // The server sent a single move update
      const { board_idx, x, y, color } = payload;
      if (board_idx >= 0 && board_idx < boards.length) {
        boards[board_idx].applyMove(x, y, color);
      }
    } else if (type === "board_reset") {
      // The server signaled a board reset
      const { board_idx } = payload;
      if (board_idx >= 0 && board_idx < boards.length) {
        console.log(`Resetting board ${board_idx}`);
        // Reset the board's state to be empty
        boards[board_idx].state = new Uint8Array(GRID_SIZE * GRID_SIZE);
        boards[board_idx].needsDraw = true;
      }
    }
  };

  // 4) Main render loop
  function renderLoop() {
    boards.forEach((b) => b.draw());
    requestAnimationFrame(renderLoop);
  }
  renderLoop();
}
