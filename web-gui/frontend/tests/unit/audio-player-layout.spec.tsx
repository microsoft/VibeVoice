import React from 'react'
import { render, screen } from '@testing-library/react'
import { AudioPlayer } from '../../src/components/audio-player'

describe('AudioPlayer layout', () => {
  beforeAll(() => {
    // jsdom doesn't implement media methods or ResizeObserver used by some UI primitives
    // Provide lightweight mocks so components can mount without errors
    ;(global as any).HTMLMediaElement.prototype.pause = () => {}
    ;(global as any).HTMLMediaElement.prototype.play = () => Promise.resolve()
    ;(global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  })

  test('time-labeled control buttons show labels and inner padding', () => {
    render(<AudioPlayer audioUrl="/static/audio/output.wav" title="Test" duration={120} />)

    const rewind10 = screen.getByRole('button', { name: /rewind 10 seconds/i })
    const rewind5 = screen.getByRole('button', { name: /rewind 5 seconds/i })
    const ff5 = screen.getByRole('button', { name: /fast forward 5 seconds/i })
    const skip10 = screen.getByRole('button', { name: /skip forward 10 seconds/i })

    // Buttons should render and include the textual label inside
    expect(rewind10).toBeInTheDocument()
    expect(rewind10).toHaveTextContent('-10s')

    expect(rewind5).toBeInTheDocument()
    expect(rewind5).toHaveTextContent('-5s')

    expect(ff5).toBeInTheDocument()
    expect(ff5).toHaveTextContent('+5s')

    expect(skip10).toBeInTheDocument()
    expect(skip10).toHaveTextContent('+10s')

    // The inner container should include padding class so labels don't overflow
    const inner = rewind10.querySelector('div')
    expect(inner).not.toBeNull()
    expect(inner?.className).toMatch(/px-2/)
  })
})