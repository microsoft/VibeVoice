import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAudioHistory } from '../../src/hooks/useAudioHistory';

function makeItem(i: number) {
  return { id: `i${i}`, filename: `f${i}`, audioUrl: `u${i}`, audioPath: `/p/${i}`, duration: null, createdAt: Date.now(), sourceDocument: null };
}

describe('useAudioHistory', () => {
  it('adds items and truncates to server limit', () => {
    const { result } = renderHook(() => useAudioHistory([]));

    act(() => {
      for (let i = 0; i < 200; i++) result.current.addAudioItem(makeItem(i));
    });

    // apply truncation for a small server limit
    const { final, dropped } = result.current.applyTruncationTo(result.current.audioHistory, 1); // perDocMax=5 totalMax=15
    expect(dropped).toBeGreaterThan(0);
    expect(final.length).toBeLessThanOrEqual(15);
  });

  it('sets and pops pending restore', () => {
    const { result } = renderHook(() => useAudioHistory([]));
    const items = [makeItem(1), makeItem(2)];
    act(() => {
      result.current.setPendingRestore(items);
    });
    const popped = result.current.popPendingRestore();
    expect(popped).toEqual(items);
    expect(result.current.popPendingRestore()).toBeUndefined();
  });
});
