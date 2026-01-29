import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVoices } from '../../src/hooks/useVoices';

const originalFetch = global.fetch;

describe('useVoices', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    localStorage.clear();
  });

  it('loads and normalizes voices and calls onConfig when config present', async () => {
    const voiceResp = { voices: [{ id: 'v1', name: 'V1', language: 'en', path: '/v1' }] };
    const cfgResp = { max_iterations_per_request: 7 };

    const fetchMock = vi.fn((url: string) => {
      if (url.endsWith('/voices')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(voiceResp) } as any);
      }
      if (url.endsWith('/config')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(cfgResp) } as any);
      }
      return Promise.reject(new Error('unexpected'));
    });

    // @ts-ignore - override global
    global.fetch = fetchMock;

    const onConfig = vi.fn();

    const { result, waitForNextUpdate } = renderHook(() => useVoices(onConfig));

    // initially loading
    expect(result.current.voicesLoading).toBeTruthy();

    // wait for effect to settle
    await act(async () => {
      // allow the hook's async load to finish
      await Promise.resolve();
    });

    expect(result.current.voicesLoading).toBeFalsy();
    expect(result.current.voices.length).toBe(1);
    expect(result.current.voices[0].id).toBe('v1');
    expect(onConfig).toHaveBeenCalledWith(7);
    expect(localStorage.getItem('vibevoice-server-max-iterations')).toBe('7');
  });

  it('sets voicesError on network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('network'))));

    const { result } = renderHook(() => useVoices());

    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.voicesError).toBeTruthy();
    expect(result.current.voices.length).toBe(0);
  });
});
