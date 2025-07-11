import { initVisualizer } from "./board-visualizer.js";

const container = document.getElementById("app");
initVisualizer({
  container,
  boardCount: 16,
  wsUrl: import.meta.env.VITE_WS_URL,
});
