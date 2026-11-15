// Two pages: the 3D viewer (index.html) and the model results (models.html).
// During development, /api requests are passed on to the upload service (uvicorn on port 8000).
import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  // relative asset paths: the same build works at a site root (local server) and in a sub-folder (GitHub Pages)
  base: "./",
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    rollupOptions: {
      input: {
        viewer: resolve(import.meta.dirname, "index.html"),
        models: resolve(import.meta.dirname, "models.html"),
      },
    },
  },
});
