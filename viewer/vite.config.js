// vite.config.js
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "../dist", // bundle into /dist
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    open: true,
  },
  define: {
    // expose your websocket URL at build time
    // e.g. .env → VITE_WS_URL=wss://api.example.com/ws
    "import.meta.env.VITE_WS_URL": JSON.stringify(
      process.env.VITE_WS_URL || "ws://localhost:8000/ws",
    ),
  },
});
