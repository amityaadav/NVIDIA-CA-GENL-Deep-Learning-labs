import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    // jsdom + the `canvas` package give the preprocessing tests a real 2D
    // context (getImageData/drawImage), which is the load-bearing step.
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.js",
  },
});
