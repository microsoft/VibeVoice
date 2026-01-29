import React from 'react'
import { render, screen } from '@testing-library/react'
import { Play } from '../../src/components/icons'

describe('Icon accessibility defaults', () => {
  test('icons are aria-hidden by default when no label/title provided', () => {
    render(<Play data-testid="play-icon" />)
    const svg = screen.getByTestId('play-icon') as SVGElement

    expect(svg.getAttribute('aria-hidden')).toBe('true')
    expect(svg.getAttribute('role')).toBeNull()
  })

  test('icons with aria-label have role="img" and are not aria-hidden', () => {
    render(<Play aria-label="Play audio" data-testid="play-icon-labeled" />)
    const svg = screen.getByTestId('play-icon-labeled') as SVGElement

    expect(svg.getAttribute('role')).toBe('img')
    expect(svg.getAttribute('aria-hidden')).toBeNull()
  })
})