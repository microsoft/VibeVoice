import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/config': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/stream': {
        target: 'ws://localhost:3000',
        ws: true,
      },
    },
  },
})
