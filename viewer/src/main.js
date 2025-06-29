import { initVisualizer } from "./board-visualizer.js";

const container = document.getElementById("app");
initVisualizer({
  container,
  boardCount: 100,
  wsUrl: import.meta.env.VITE_WS_URL,
});
