import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "/" for local dev; set VITE_BASE=/<repo>/ when building for GitHub Pages.
// Proxy /api to the FastAPI backend so the frontend can use relative URLs locally.
export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        // Use an explicit IPv4 address so the proxy and Uvicorn agree even
        // when `localhost` resolves to IPv6 first.
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
