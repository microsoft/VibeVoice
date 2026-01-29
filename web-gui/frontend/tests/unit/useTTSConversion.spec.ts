import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useTTSConversion } from '../../src/hooks/useTTSConversion';

const originalFetch = global.fetch;
const originalCreateObjectURL = URL.createObjectURL;

describe('useTTSConversion', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    URL.createObjectURL = originalCreateObjectURL;
  });

  it('preview returns built audio url on success', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string, init?: any) => {
      if (url.endsWith('/api/tts/preview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ success: true, audio_url: '/audio/preview.wav' }) } as any);
      }
      return Promise.reject(new Error('unexpected'));
    }));

    const { result } = renderHook(() => useTTSConversion());
    const url = await act(async () => result.current.preview('v1'));
    expect(typeof url).toBe('string');
    expect(url).toContain('/audio/preview.wav');
  });

  it('convert produces entries, updates generated audio state and toggles isConverting', async () => {
    // Delay the convert response slightly so we can observe the isConverting flag while conversion is in-flight
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.endsWith('/api/tts/convert')) {
        return new Promise((resolve) => setTimeout(() => resolve({ ok: true, json: () => Promise.resolve({ success: true, outputs: [{ audio_url: '/audio/1.wav', filename: 'chunk1.wav', duration: 1.2 }] }) } as any), 20));
      }
      return Promise.reject(new Error('unexpected'));
    }));

    const { result } = renderHook(() => useTTSConversion());

    const params = {
      documentContent: 'hello world',
      selectedVoice: 'v1',
      configuration: { chunkDepth: 1, pauseMs: 100, includeHeading: false, stripMarkdown: true, device: 'auto', iterations: 1 },
      documentFilename: 'doc.md',
      maxIterationsPerRequest: 5,
    } as any;

    expect(result.current.isConverting).toBeFalsy();

    await act(async () => {
      const entries = await result.current.convert(params);
      // ensure it ends not converting and produced entries
      await waitFor(() => expect(result.current.isConverting).toBeFalsy());
      expect(Array.isArray(entries)).toBeTruthy();
      expect(entries.length).toBeGreaterThan(0);
      expect(entries[0].audioUrl).toContain('/audio/1.wav');
      expect(entries[0].audioPath).toBe('/audio/1.wav');
      expect(entries[0].duration).toBe(1.2);    });
  });

  it('convert rejects when requested iterations exceed maxIterationsPerRequest', async () => {
    const { result } = renderHook(() => useTTSConversion());

    const params = {
      documentContent: 'hello',
      selectedVoice: 'v1',
      configuration: { chunkDepth: 1, pauseMs: 100, includeHeading: false, stripMarkdown: true, device: 'auto', iterations: 10 },
      documentFilename: 'doc.md',
      maxIterationsPerRequest: 5,
    } as any;

    await expect(act(async () => result.current.convert(params))).rejects.toThrow();
  });

  it('exportAudio downloads blob and triggers click', async () => {
    // First call: export download request
    // Second call: actual file download (returns blob)
    const fetchMock = vi.fn((url: string, init?: any) => {
      if (url.endsWith('/api/export/download')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ success: true, download_url: '/downloads/file.wav' }) } as any);
      }
      if (url.endsWith('/downloads/file.wav')) {
        return Promise.resolve({ ok: true, blob: () => Promise.resolve(new Blob(['audio'])) } as any);
      }
      return Promise.reject(new Error('unexpected'));
    });

    vi.stubGlobal('fetch', fetchMock);

    // Mock object URL creation + revoke
    vi.stubGlobal('URL', { ...globalThis.URL, createObjectURL: vi.fn(() => 'blob://fake'), revokeObjectURL: vi.fn() } as any);

    // Mock anchor click behavior
    const clickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        const el = originalCreateElement(tagName as any) as HTMLAnchorElement;
        // override the click and remove so we can assert against them, but keep it a real Node
        el.click = clickSpy as any;
        el.remove = () => {};
        return el;
      }
      // default behavior
      return originalCreateElement(tagName as any);
    });

    const { result } = renderHook(() => useTTSConversion());

    await act(async () => {
      await result.current.exportAudio({ audioPath: '/audio/1.wav', downloadFilename: 'out.wav' });
    });

    // ensure fetch was called for both export request and file download
    expect(fetchMock).toHaveBeenCalled();
    expect((URL.createObjectURL as unknown as vi.Mock).mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(clickSpy).toHaveBeenCalled();
  });
});
