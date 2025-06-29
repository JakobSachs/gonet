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

export function initVisualizer({ container, boardCount }) {
  // 1) Setup container style
  container.style.display = "grid";
  container.style.gridTemplateColumns = `repeat(auto-fill, minmax(${BOARD_PIXELS}px, 1fr))`;
  container.style.gap = `${BOARD_GAP}px`;
  container.style.padding = `${BOARD_GAP}px`;

  // 2) Instantiate all boards
  const boards = Array.from({ length: boardCount }, () => new Board(container));

  // 3) Simulate periodic updates from an external source
  function simulateRandomUpdates() {
    const board = boards[Math.floor(Math.random() * boardCount)];
    const x = Math.floor(Math.random() * GRID_SIZE);
    const y = Math.floor(Math.random() * GRID_SIZE);
    const color = Math.floor(Math.random() * 3); // 0=empty, 1=black, 2=white
    board.applyMove(x, y, color);
  }

  setInterval(simulateRandomUpdates, 50); // Update 20 times per second

  // 4) Main render loop
  function renderLoop() {
    boards.forEach((b) => b.draw());
    requestAnimationFrame(renderLoop);
  }
  renderLoop();
}
