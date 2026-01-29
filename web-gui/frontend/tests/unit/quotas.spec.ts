import { computePerDocMax, computeTotalMax } from '../../src/lib/quotas'

describe('quota helpers', () => {
  test('computePerDocMax computes per-doc quota', () => {
    expect(computePerDocMax(10)).toBe(50)
    expect(computePerDocMax(0)).toBe(0)
    expect(computePerDocMax(1)).toBe(5)
  })

  test('computeTotalMax computes total quota', () => {
    expect(computeTotalMax(10)).toBe(150)
    expect(computeTotalMax(0)).toBe(0)
    expect(computeTotalMax(1)).toBe(15)
  })
})