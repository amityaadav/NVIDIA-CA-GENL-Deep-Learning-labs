import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// A strict Content-Security-Policy, injected only into the production build so
// it doesn't break the dev server's HMR. The app is fully self-contained (no
// external scripts, fonts, or APIs), so 'self' is all it needs.
function cspMeta() {
  const policy = [
    "default-src 'self'",
    "img-src 'self' data:",
    "style-src 'self' 'unsafe-inline'", // React sets inline style attributes
    "script-src 'self'",
    "connect-src 'self'", // only the same-origin weights file
    "base-uri 'self'",
    "object-src 'none'",
  ].join("; ");
  return {
    name: "csp-meta",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace(
        "</head>",
        `  <meta http-equiv="Content-Security-Policy" content="${policy}">\n  </head>`,
      );
    },
  };
}

export default defineConfig({
  // Set VITE_BASE to "/<repo>/" for GitHub Pages project sites; "/" for dev.
  base: process.env.VITE_BASE ?? "/",
  plugins: [react(), cspMeta()],
  server: { port: 5173 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.js",
  },
});
