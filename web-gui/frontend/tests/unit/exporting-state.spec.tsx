import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import Dashboard from '../../src/app/page'
import { ThemeProvider } from '../../src/providers/theme-provider'
import { vi } from 'vitest'

const mockVoicesResponse = { voices: [] }
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
)

describe('Per-item exporting state', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetAllMocks()

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

  test('download button shows downloading state and disables while exporting', async () => {
    const item = {
      id: 'test-1',
      filename: 'test-1.wav',
      audioUrl: '/static/test-1.wav',
      audioPath: '/static/test-1.wav',
      duration: null,
      createdAt: Date.now(),
    }

    const payload = {
      documentContent: '',
      documentFilename: '',
      selectedVoice: null,
      configuration: {},
      activeTab: 'upload',
      audioHistory: [item],
      selectedAudioId: null,
      savedAt: Date.now(),
    }

    localStorage.setItem('vibevoice-session', JSON.stringify(payload))

    // Stub fetch: first call to /voices, others to export/download
    const downloadDelay = 150

    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? ''
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: async () => mockVoicesResponse })
      }
      if (url.includes('/api/export/download')) {
        return Promise.resolve({ ok: true, json: async () => ({ success: true, download_url: '/download/test-1.wav' }) })
      }
      if (url.includes('/download/test-1.wav')) {
        // simulate delayed blob response
        return new Promise((resolve) => setTimeout(() => resolve({ ok: true, blob: async () => new Blob() }), downloadDelay))
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }))

    render(<Dashboard />, { wrapper: Wrapper })

    const downloadButton = await screen.findByRole('button', { name: /Download/i })
    expect(downloadButton).toBeInTheDocument()

    // Click to start export
    fireEvent.click(downloadButton)

    // Immediately the button should be disabled and show Downloading (wait for UI changes)
    await waitFor(() => {
      expect(downloadButton).toBeDisabled()
      expect(downloadButton).toHaveTextContent('Downloading...')
    })

    // After download finishes the button should re-enable
    await waitFor(() => expect(downloadButton).not.toBeDisabled(), { timeout: 1000 })

    // And its label should have been restored to "Download"
    expect(downloadButton).toHaveTextContent(/^Download$/i)

  })
})
