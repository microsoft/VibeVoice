import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSessionPersistence } from '../../src/hooks/useSessionPersistence';

describe('useSessionPersistence', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('pushHistory and loadHistory work and enforce MAX_HISTORY', () => {
    const { result } = renderHook(() => useSessionPersistence());
    const snapshot = (i: number) => ({ documentContent: `doc${i}`, documentFilename: `f${i}`, selectedVoice: null, configuration: {}, activeTab: 'upload', audioHistory: [], selectedAudioId: null, savedAt: Date.now() });

    act(() => {
      for (let i = 0; i < 25; i++) {
        result.current.pushHistory(snapshot(i));
      }
    });

    const loaded = result.current.loadHistory();
    expect(loaded.length).toBeLessThanOrEqual(20);
    expect(result.current.historyCount).toBe(loaded.length);
  });

  it('persistSession writes session and pushes history', () => {
    const { result } = renderHook(() => useSessionPersistence());
    const snap = { documentContent: 'd', documentFilename: 'f', selectedVoice: null, configuration: {}, activeTab: 'upload', audioHistory: [], selectedAudioId: null, savedAt: Date.now() } as any;
    act(() => result.current.persistSession(snap));
    const session = JSON.parse(localStorage.getItem('vibevoice-session') || '{}');
    expect(session.documentContent).toBe('d');
    const history = result.current.loadHistory();
    expect(history[0].documentContent).toBe('d');
  });

  it('resetSession clears keys', () => {
    const { result } = renderHook(() => useSessionPersistence());
    localStorage.setItem('vibevoice-session', 'x');
    localStorage.setItem('vibevoice-session-history', 'y');
    act(() => result.current.resetSession());
    expect(localStorage.getItem('vibevoice-session')).toBeNull();
    expect(localStorage.getItem('vibevoice-session-history')).toBeNull();
    expect(result.current.historyCount).toBe(0);
  });
});
