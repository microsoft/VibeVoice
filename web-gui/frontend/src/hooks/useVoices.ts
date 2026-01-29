import { useCallback, useEffect, useState } from 'react';
import { buildApiUrl, API_BASE_URL } from '@/lib/api';

export interface Voice {
  id: string;
  name: string;
  language: string;
  gender?: 'male' | 'female';
  path: string;
}

export function useVoices(onConfig?: (maxIterations: number) => void) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voicesLoading, setVoicesLoading] = useState(true);
  const [voicesError, setVoicesError] = useState<string | null>(null);

  const load = useCallback(async (controllerParam?: AbortController) => {
    setVoicesLoading(true);
    setVoicesError(null);
    const controller = controllerParam ?? new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(buildApiUrl('/voices'), { signal: controller.signal });
      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`Voice list request failed: ${response.status} ${response.statusText} - ${text}`);
      }
      const result = await response.json();

      // Attempt to fetch server configuration to adapt client-side quotas dynamically
      try {
        const cfgResp = await fetch(buildApiUrl('/config'), { signal: controller.signal });
        if (cfgResp.ok) {
          const cfgJson = await cfgResp.json();
          if (typeof cfgJson?.max_iterations_per_request === 'number' && typeof onConfig === 'function') {
            onConfig(cfgJson.max_iterations_per_request);
            try {
              localStorage.setItem('vibevoice-server-max-iterations', String(cfgJson.max_iterations_per_request));
            } catch (err) {
              console.debug('Failed to persist server max iterations to localStorage', err);
            }
          }
        }
      } catch (err) {
        // Ignore failures and continue with frontend defaults
        console.debug('Failed to fetch server config, using defaults', err);
      }

      const voiceItems = Array.isArray(result?.voices) ? result.voices : [];
      const normalized = voiceItems.map((voice: Partial<Voice>, index: number) => ({
        id: voice.id ?? `voice-${index + 1}`,
        name: voice.name ?? voice.id ?? `Voice ${index + 1}`,
        language: voice.language ?? 'Unknown',
        gender: voice.gender,
        path: voice.path ?? '',
      }));

      setVoices(normalized);
    } catch (error) {
      const fallbackMessage = error instanceof Error ? error.message : 'Failed to load voices';
      const isAbort = error instanceof Error && (error as any).name === 'AbortError';
      const isNetworkError = error instanceof TypeError;

      if (!isNetworkError && !isAbort) {
        console.warn('Failed to load voices:', error);
      }

      setVoicesError(
        isAbort
          ? 'Voice request timed out. Please try again.'
          : isNetworkError
          ? `Unable to reach the backend at ${API_BASE_URL}. Make sure the backend is running.`
          : fallbackMessage,
      );
      setVoices([]);
    } finally {
      window.clearTimeout(timeoutId);
      setVoicesLoading(false);
    }
  }, [onConfig]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller);
    return () => controller.abort();
  }, [load]);

  return { voices, voicesLoading, voicesError, refresh: load, setVoices } as const;
}
