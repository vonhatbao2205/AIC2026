/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to 0.0.0.0 so other machines on the same LAN can reach the dev server
    // (prints a "Network:" URL). API calls stay relative ("/api") and are proxied
    // to the backend on THIS host, so the backend never needs LAN exposure or CORS.
    host: true,
    port: 5173,
    // Allow access through a tunnel (ngrok/cloudflared) — Vite 5.4+ blocks unknown
    // Host headers; bare LAN IPs are allowed by default but tunnel domains are not.
    allowedHosts: [".ngrok-free.app", ".ngrok.app", ".ngrok.io", ".trycloudflare.com", "unarranged-venus-ovately.ngrok-free.dev"],
    proxy: {
      // Proxy API calls to the backend during dev.
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    // Vite loads .env.local for tests too, so without this the suite would spin
    // up a real Supabase client and point at the team's live submission table.
    // Tests must exercise the local-only path; sync itself is covered by the
    // pure record/merge tests in src/test/submission.test.ts.
    env: {
      VITE_SUPABASE_URL: "",
      VITE_SUPABASE_PUBLISHABLE_KEY: "",
    },
  },
});
