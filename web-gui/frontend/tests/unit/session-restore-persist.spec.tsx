import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'
import { computeTotalMax } from '../../src/lib/quotas'

const mockVoicesResponse = { voices: [] }
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('Session restore persistence', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetAllMocks()

    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      }
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
      dispatchEvent: vi.fn(),
    })))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('sanitized and truncated audioHistory is written back to localStorage after restore', async () => {
    // Create a payload with malformed entries and more than MAX_AUDIO_HISTORY entries
    const perReq = 10
    // Use canonical computeTotalMax so this test follows the app's quota calculation
    // totalMax here represents the maximum audio history length derived from per-request limits
    const totalMax = computeTotalMax(perReq)
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
      { id: '', audioUrl: '/x.wav', audioPath: '/x.wav' },
      { id: 'no-url', audioPath: '/y.wav' },
      validItem,
    ]

    const big = Array.from({ length: totalMax + 5 }).map((_, i) => ({
      id: `a-${i}`,
      filename: `a-${i}.wav`,
      audioUrl: `/x/a-${i}.wav`,
      audioPath: `/x/a-${i}.wav`,
      duration: null,
      createdAt: 1600000000000 + i,
    }))

    const payload = {
      documentContent: '',
      documentFilename: '',
      selectedVoice: null,
      configuration: {},
      activeTab: 'upload',
      audioHistory: [...malformed, ...big],
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    render(<Dashboard />, { wrapper: Wrapper })

    // Wait for UI to show Audio History area (restore happened)
    const header = await screen.findByText(/Audio History/i)
    expect(header).toBeInTheDocument()

    // Wait until the sanitized session has been written back to localStorage and assert final invariants
    await waitFor(() => {
      const savedCheck = localStorage.getItem('vibevoice-session')
      expect(savedCheck).not.toBeNull()
      const parsedCheck = JSON.parse(savedCheck as string)
      expect(Array.isArray(parsedCheck.audioHistory)).toBe(true)
      expect(parsedCheck.audioHistory.length).toBeLessThanOrEqual(totalMax)
      // Ensure malformed entries (e.g., empty id) were filtered out
      expect(parsedCheck.audioHistory.find((item: any) => item.id === '')).toBeUndefined()
      // Ensure no items have missing or empty audioUrl values
      expect(parsedCheck.audioHistory.some((item: any) => !item.audioUrl || String(item.audioUrl).trim() === '')).toBe(false)

      // All entries should have required properties
      for (const item of parsedCheck.audioHistory) {
        expect(typeof item.id).toBe('string')
        expect(typeof item.audioUrl).toBe('string')
        expect(typeof item.audioPath).toBe('string')
      }
    }, { timeout: 2000 })
  })
})
