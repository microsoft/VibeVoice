import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'
import { DEFAULT_MAX_ITERATIONS_PER_REQUEST, computePerDocMax, computeTotalMax } from '../../src/lib/quotas'

const mockVoicesResponse = { voices: [] }
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('Audio history capping', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetAllMocks()

    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      }
      // Default convert response will be overridden per test
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

  test('restored history > total max is truncated', async () => {
    const perReq = DEFAULT_MAX_ITERATIONS_PER_REQUEST
    const perDocMax = computePerDocMax(perReq)
    const totalMax = computeTotalMax(perReq)
    const big = Array.from({ length: totalMax + 10 }).map((_, i) => ({
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
      audioHistory: big,
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    render(<Dashboard />, { wrapper: Wrapper })

    // The UI header should show capped count (totalMax files)
    const header = await screen.findByText(/Audio History/i)
    expect(header).toBeInTheDocument()

    await waitFor(() => {
      const countNode = screen.getByText((content) => /\d+ file/.test(content))
      expect(countNode).toBeInTheDocument()
      const match = countNode.textContent?.match(/(\d+) file/)
      expect(match).not.toBeNull()
      const count = match ? parseInt(match[1], 10) : NaN
      expect(count).toBe(totalMax)
    })
  })

  test('on convert, merged history is capped to total max', async () => {
    const perReq = DEFAULT_MAX_ITERATIONS_PER_REQUEST
    const perDocMax = computePerDocMax(perReq)
    const totalMax = computeTotalMax(perReq)
    // Preload session with totalMax - 2 items
    const pre = Array.from({ length: totalMax - 2 }).map((_, i) => ({
      id: `pre-${i}`,
      filename: `pre-${i}.wav`,
      audioUrl: `/pre/pre-${i}.wav`,
      audioPath: `/pre/pre-${i}.wav`,
      duration: null,
      createdAt: 1600000000000 + i,
      sourceDocument: '',
    }))

    const payload = {
      documentContent: 'hello',
      documentFilename: 'doc.md',
      selectedVoice: 'en-Emma_woman',
      configuration: {},
      activeTab: 'upload',
      audioHistory: pre,
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    // Make convert endpoint return 2 outputs (which would push total to totalMax)
    let seq = 0

    // Factory for a convert response for a given sequence number
    const makeConvertResponse = (s: number) => ({
      ok: true,
      json: async () => ({
        success: true,
        outputs: [
          { audio_url: `/out/out-${s}-a.wav`, filename: `out-${s}-a.wav`, duration: 1.2 },
          { audio_url: `/out/out-${s}-b.wav`, filename: `out-${s}-b.wav`, duration: 0.7 },
        ],
      }),
    })

    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      }
      if (url.includes('/api/tts/convert')) {
        seq++
        return Promise.resolve(makeConvertResponse(seq))
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }))

    render(<Dashboard />, { wrapper: Wrapper })

    // Wait for restore to complete and THEN click Convert button
    const convertButton = await screen.findByRole('button', { name: /Convert to Speech/i })

    // Ensure the restored session has been applied (pre-length) before triggering convert to avoid race conditions
    await waitFor(() => {
      const countNode = screen.getByText((content) => /\d+ file/.test(content))
      expect(countNode).toBeInTheDocument()
      const match = countNode.textContent?.match(/(\d+) file/)
      expect(match).not.toBeNull()
      const count = match ? parseInt(match[1], 10) : NaN
      expect(count).toBe(totalMax - 2)
    }, { timeout: 2000 })

    // Wait for filename and selected voice to be present in the action bar
    const filenames = await screen.findAllByText('doc.md')
    expect(filenames.length).toBeGreaterThan(0)
    const voicesText = await screen.findAllByText('en-Emma_woman')
    expect(voicesText.length).toBeGreaterThan(0)

    // Wait until Convert is enabled, then click; after completion, merged history should be capped
    await waitFor(() => expect(convertButton).toBeEnabled())
    fireEvent.click(convertButton)

    // Ensure the convert endpoint was actually invoked
    await waitFor(() => expect(seq).toBeGreaterThan(0), { timeout: 1000 })

    await waitFor(() => {
      const countNode = screen.getByText((content) => /\d+ file/.test(content))
      expect(countNode).toBeInTheDocument()
      const match = countNode.textContent?.match(/(\d+) file/)
      expect(match).not.toBeNull()
      const count = match ? parseInt(match[1], 10) : NaN
      // After merging 2 new outputs, the resulting count must equal the configured cap (detect off-by-one errors)
      expect(count).toBe(totalMax)
    }, { timeout: 3000 })
  })
})
