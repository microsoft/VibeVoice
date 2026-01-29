#!/usr/bin/env node
// Wrapper to run vitest with a file path or arbitrary args reliably across shells
const { spawn } = require('child_process')
const args = process.argv.slice(2)
if (args.length === 0) {
  console.error('Usage: node scripts/run-vitest-file.js <path-or-pattern> [vitest args]')
  process.exit(2)
}
// Include explicit config path so root-level vitest doesn't pick up unrelated tests or configs
const child = spawn(`npx vitest run -c ./frontend/vitest.config.ts -- ${args.map(a => `"${a.replace(/"/g, '\"')}"`).join(' ')}`, { stdio: 'inherit', shell: true })
child.on('exit', (code) => process.exit(code))
