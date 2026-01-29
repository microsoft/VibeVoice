import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  // Ensure vitest resolves paths and includes relative to the frontend package root
  root: path.resolve(__dirname),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: 'tests/setup.ts',
    globals: true,
    include: ['tests/unit/**/*.spec.*'],
    coverage: {
      provider: 'v8'
    }
  }
})
