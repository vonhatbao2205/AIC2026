/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Public hostname when the dev server is exposed through a tunnel, e.g.
// `VITE_TUNNEL_HOST=app.baoencoder.site npm run dev`. Only HMR needs it: the
// browser derives the websocket URL from the page, so behind a tunnel it would
// dial `wss://<host>:5173`, a port the tunnel does not publish. Left unset, the
// dev server behaves exactly as before for local work.
const tunnelHost = process.env.VITE_TUNNEL_HOST;

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
    allowedHosts: [".ngrok-free.app", ".ngrok.app", ".ngrok.io", ".trycloudflare.com", "unarranged-venus-ovately.ngrok-free.dev", ".ts.net", ".baoencoder.site"],
    // Tunnels terminate TLS on 443, so the HMR client has to be told that
    // instead of guessing from `server.port`.
    ...(tunnelHost
      ? { hmr: { protocol: "wss" as const, host: tunnelHost, clientPort: 443 } }
      : {}),
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
