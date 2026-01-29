import React from 'react'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { VoiceSelection } from '../../src/components/voice-selection'

const voices = [
  { id: 'en-Emma_woman', name: 'Emma (female)', language: 'english', path: '/voices/en-emma.wav' },
  { id: 'en-Carter_man', name: 'Carter (male)', language: 'english', path: '/voices/en-carter.wav' }
]

describe('VoiceSelection - blocked autoplay UI', () => {
  test('shows manual Play button when autoplay is blocked', async () => {
    const mockOnVoicePreview = vi.fn().mockReturnValue(true)
    const mockOnVoicePreviewManual = vi.fn()
    const mockOnVoiceSelect = vi.fn()

    render(
      <VoiceSelection
        voices={voices}
        selectedVoice={null}
        onVoiceSelect={mockOnVoiceSelect}
        onVoicePreview={mockOnVoicePreview}
        onVoiceManualPlay={mockOnVoicePreviewManual}
      />
    )

    // Open the first voice card preview control
    const previewButtons = await screen.findAllByRole('button', { name: /preview/i })
    expect(previewButtons.length).toBeGreaterThan(0)

    await act(async () => {
      fireEvent.click(previewButtons[0])
      // Wait for the component to call the preview callback and settle state updates
      await waitFor(() => expect(mockOnVoicePreview).toHaveBeenCalled())
    })

    // The component should now show manual play affordance

    const playButton = await screen.findByRole('button', { name: /play emma/i })
    expect(playButton).toBeInTheDocument()

    fireEvent.click(playButton)
    expect(mockOnVoicePreviewManual).toHaveBeenCalledTimes(1)
  })
})


describe('VoiceSelection - language grouping', () => {
  test('renders language groups in alphabetical order', async () => {
    const unorderedVoices = [
      { id: 'es-1', name: 'Spanish 1', language: 'spanish', path: '' },
      { id: 'en-1', name: 'English 1', language: 'english', path: '' },
      { id: 'de-1', name: 'German 1', language: 'german', path: '' },
    ]

    render(
      <VoiceSelection
        voices={unorderedVoices}
        selectedVoice={null}
        onVoiceSelect={() => {}}
        onVoicePreview={() => {}}
        onVoiceManualPlay={() => {}}
      />
    )

    // Wait for language headings to appear in DOM and capture them once using testing-library queries
    let headingElems: HTMLElement[] = []
    await waitFor(() => {
      headingElems = screen.getAllByRole('heading', { level: 3 })
      const found = headingElems
        .map((h) => h.textContent?.trim() ?? '')
        .filter((text) => /english|german|spanish/i.test(text))
      expect(found.length).toBe(3)
    })

    // Map captured headings to their text content and assert ordering
    const allHeadings = headingElems
      .map((h) => h.textContent?.trim() ?? '')
      .filter((text) => /english|german|spanish/i.test(text))

    expect(allHeadings).toEqual(['English', 'German', 'Spanish'])
  })
})
