import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'

const mockVoicesResponse = { voices: [] }
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('History persistence on quota error', () => {
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
    })))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    // Restore any spies created with vi.spyOn() to ensure clean teardown
    vi.restoreAllMocks()
  })

  test('falls back to trimmed retry when localStorage.setItem throws and sets history count', async () => {
    // Seed a session so the Save Version button is enabled
    const payload = {
      documentContent: 'Hello world',
      documentFilename: 'doc.md',
      selectedVoice: null,
      configuration: {},
      activeTab: 'upload',
      audioHistory: [],
      selectedAudioId: null,
      savedAt: Date.now(),
    }
    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    // Spy on setItem and make it throw for HISTORY_KEY the first time
    const originalSetItem = Storage.prototype.setItem
    let called = 0
    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (key: string, value: string) {
      if (key === 'vibevoice-session-history' && called === 0) {
        called++;
        throw new Error('QuotaExceededError simulated');
      }
      return originalSetItem.call(this, key, value);
    })

    render(<Dashboard />, { wrapper: Wrapper })

    // Click Save Version to trigger pushHistory
    const saveButton = await screen.findByRole('button', { name: /Save Version/i })
    expect(saveButton).toBeInTheDocument()

    fireEvent.click(saveButton)

    // Wait for retries and updates
    await waitFor(() => {
      // Count only calls that tried to write the HISTORY_KEY to assert retry behavior
      const filteredCalls = spy.mock.calls.filter((call: any[]) => call[0] === 'vibevoice-session-history')
      expect(filteredCalls.length).toBeGreaterThanOrEqual(2)
    })

    // Now ensure HISTORY_KEY exists and is an array and its length <= original expected
    const historyRaw = localStorage.getItem('vibevoice-session-history')
    expect(historyRaw).not.toBeNull()
    const parsed = JSON.parse(historyRaw as string)
    expect(Array.isArray(parsed)).toBe(true)

    // setHistoryCount should have been set to at least 1 (the new snapshot)
    const hdr = await screen.findByText(/Versions: \d+/i)
    expect(hdr).toBeInTheDocument()

    // Defer restoration to afterEach handler to ensure it runs even on failure
    // spy.mockRestore()
  })
})
