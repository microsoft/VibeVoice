import React from 'react'
import { render, screen } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'
import { toast } from 'sonner' // Mock/spy for `toast` is configured in `tests/setup.ts` for reliable tests

// Simulate a voices API to satisfy initial fetch
const mockVoicesResponse = { voices: [] }

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('Session restore sanitization', () => {
  beforeEach(() => {
    // Clear localStorage and mocks
    localStorage.clear()
    vi.resetAllMocks()

    // Ensure `toast.info` is a spy so assertions remain reliable when running tests in isolation.
    // The canonical mock is provided in `tests/setup.ts`; if not present, create a spy (or noop mock)
    try {
      if (!(toast.info as any)?.mock) {
        // Use spyOn to wrap the existing implementation and record calls without changing behavior
        vi.spyOn(toast, 'info')
      }
    } catch (err) {
      // If spyOn fails (non-configurable), fall back to a noop mock to allow assertions
      ;(toast as any).info = vi.fn()
    }

    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }))

    // JSDOM doesn't implement matchMedia; provide a minimal stub used by ThemeProvider
    vi.stubGlobal('matchMedia', vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      onchange: null,
    })))
  })

  afterEach(() => {
    // restore fetch mock
    vi.unstubAllGlobals()
  })

  test('drops malformed audioHistory entries and shows toast', async () => {
    const validItem = {
      id: 'good-1',
      filename: 'good.wav',
      audioUrl: '/static/voices/good.wav',
      audioPath: '/static/voices/good.wav',
      duration: 1.23,
      createdAt: 1600000000000,
    }

    const malformed = [
      null,
      123,
      { id: '', audioUrl: '/x.wav', audioPath: '/x.wav' }, // invalid id
      { id: 'no-url', audioPath: '/y.wav' }, // missing audioUrl
      { id: 'no-path', audioUrl: '/z.wav' }, // missing audioPath
      validItem,
    ]

    const payload = {
      documentContent: '',
      documentFilename: '',
      selectedVoice: null,
      configuration: {},
      activeTab: 'upload',
      audioHistory: malformed,
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    render(<Dashboard />, { wrapper: Wrapper })

    // Audio History should show only the valid item
    const historyHeader = await screen.findByText(/Audio History/i)
    expect(historyHeader).toBeInTheDocument()

    // Should display the valid filename (wait for async render)
    const goodFile = await screen.findByText('good.wav')
    expect(goodFile).toBeInTheDocument()

    // Should not display the invalid entries
    expect(screen.queryByText(/no-url|no-path/)).not.toBeInTheDocument()

    // The toast mock from tests/setup.ts exposes toast.info spy
    // Verify the user-facing message mentions the restored session
    expect(toast.info).toHaveBeenCalledWith(expect.stringContaining('Restored session'))
  })
})
