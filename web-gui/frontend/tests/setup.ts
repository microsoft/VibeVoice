import '@testing-library/jest-dom'

// Mock sonner to avoid real toasts during unit tests and expose spies
vi.mock('sonner', () => {
  // Use `any` for toast to allow assigning mock methods easily in tests
  const toast: any = vi.fn()
  // Attach common methods as spies so tests can assert on both toast() calls and toast.success/error/info/etc.
  toast.success = vi.fn()
  toast.error = vi.fn()
  toast.info = vi.fn()
  toast.warning = vi.fn()
  toast.loading = vi.fn()
  toast.dismiss = vi.fn()
  // Ensure promise returns a Promise so awaiting code works in tests
  toast.promise = vi.fn((p: Promise<unknown> | unknown) => {
    if (p && typeof (p as any).then === 'function') {
      return p as Promise<unknown>
    }
    return Promise.resolve(p)
  })
  // Additional toast APIs commonly used by the app
  toast.custom = vi.fn()
  toast.message = vi.fn()

  return { toast }
})

// Suppress known non-actionable React testing warnings about `act(...)` in this test environment
const _origConsoleError = console.error
console.error = (...args: unknown[]) => {
  const first = String(args?.[0] ?? '')
  if (/not wrapped in act|configured to support act|ReactDOMTestUtils.act is deprecated/i.test(first)) {
    return
  }
  _origConsoleError.apply(console, args as any)
}

