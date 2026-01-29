import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'

import { toast } from 'sonner'

// Mock the `sonner` toast module so tests can assert on toast calls
vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
  },
}))

import { DEFAULT_MAX_ITERATIONS_PER_REQUEST, computePerDocMax, computeTotalMax } from '../../src/lib/quotas'

const mockVoicesResponse = { voices: [] }
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('Convert quotas enforcement', () => {
  // Helper to get all calls to the TTS convert API
  const getConvertApiCalls = () => {
    // Prefer vitest's mocked helper; fall back to the real global.fetch
    const fetchMock = vi.mocked(global.fetch) ?? (global.fetch as any)
    const calls = (fetchMock && (fetchMock as any).mock && (fetchMock as any).mock.calls) || []
    return calls.filter((args: any[]) => {
      const url = typeof args[0] === 'string' ? args[0] : args[0]?.url
      return typeof url === 'string' && url.includes('/api/tts/convert')
    })
  }

  beforeEach(() => {
    localStorage.clear()
    vi.resetAllMocks()

    // Default fetch stub: voices and config
    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      if (url.endsWith('/config')) return Promise.resolve({ ok: true, json: async () => ({ max_iterations_per_request: 10 }) })
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }))

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
    vi.unstubAllGlobals()
  })

  test('blocks conversion when total quota exceeded', async () => {
    // Build a big history using application quota constants
    const perDocMax = computePerDocMax(DEFAULT_MAX_ITERATIONS_PER_REQUEST)
    const totalMax = computeTotalMax(DEFAULT_MAX_ITERATIONS_PER_REQUEST)
    const big = Array.from({ length: totalMax }).map((_, i) => ({
      id: `a-${i}`,
      filename: `a-${i}.wav`,
      audioUrl: `/x/a-${i}.wav`,
      audioPath: `/x/a-${i}.wav`,
      duration: null,
      createdAt: 1600000000000 + i,
      sourceDocument: `doc-${i % 5}`,
    }))

    const payload = {
      documentContent: 'hello',
      documentFilename: 'doc-new.md',
      selectedVoice: 'en-Emma_woman',
      configuration: {},
      activeTab: 'upload',
      audioHistory: big,
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    render(<Dashboard />, { wrapper: Wrapper })

    // Wait for UI
    const convertButton = await screen.findByRole('button', { name: /Convert to Speech/i })

    fireEvent.click(convertButton)

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('Total quota exceeded'))
    })

    // Ensure convert endpoint wasn't called
    expect(getConvertApiCalls()).toHaveLength(0)
  })

  test('blocks conversion when per-document quota exceeded', async () => {
    // Document1 has perDocMax entries (use app constants)
    const perDocMax = computePerDocMax(DEFAULT_MAX_ITERATIONS_PER_REQUEST)
    const docEntries = Array.from({ length: perDocMax }).map((_, i) => ({
      id: `d1-${i}`,
      filename: `d1-${i}.wav`,
      audioUrl: `/x/d1-${i}.wav`,
      audioPath: `/x/d1-${i}.wav`,
      duration: null,
      createdAt: 1600000000000 + i,
      sourceDocument: 'document1.md',
    }))

    const payload = {
      documentContent: 'hello',
      documentFilename: 'document1.md',
      selectedVoice: 'en-Emma_woman',
      configuration: {},
      activeTab: 'upload',
      audioHistory: docEntries,
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    render(<Dashboard />, { wrapper: Wrapper })

    const convertButton = await screen.findByRole('button', { name: /Convert to Speech/i })

    fireEvent.click(convertButton)

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('Per-document quota exceeded'))
    })

    expect(getConvertApiCalls()).toHaveLength(0)
  })
})
