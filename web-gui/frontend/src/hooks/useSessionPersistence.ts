import { useCallback, useState } from 'react';

const SESSION_KEY = 'vibevoice-session';
const HISTORY_KEY = 'vibevoice-session-history';
const MAX_HISTORY = 20;

export interface SessionSnapshot {
  documentContent: string;
  documentFilename: string;
  selectedVoice: string | null;
  configuration: any;
  activeTab: string;
  audioHistory: any[];
  selectedAudioId: string | null;
  savedAt: number;
}

export function useSessionPersistence() {
  const [historyCount, setHistoryCount] = useState<number>(0);

  const loadHistory = useCallback((): SessionSnapshot[] => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }, []);

  const pushHistory = useCallback((snapshot: SessionSnapshot) => {
    if (!snapshot.documentContent.trim()) {
      return;
    }
    const existing = loadHistory();
    const last = existing[0];
    if (
      last &&
      last.documentContent === snapshot.documentContent &&
      last.documentFilename === snapshot.documentFilename &&
      last.selectedVoice === snapshot.selectedVoice
    ) {
      return;
    }

    const nextHistory = [snapshot, ...existing].slice(0, MAX_HISTORY);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(nextHistory));
      setHistoryCount(nextHistory.length);
    } catch (err) {
      // Handle localStorage quota exceeded or other storage errors gracefully.
      try {
        // Drop oldest entries and retry once to reduce size
        const trimmed = nextHistory.slice(0, Math.max(1, Math.floor(nextHistory.length / 2)));
        localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
        setHistoryCount(trimmed.length);
      } catch (err2) {
        // If retry fails, skip persistence but still update in-memory count to reflect intended history length
        setHistoryCount(nextHistory.length);
      }
    }
  }, [loadHistory]);

  const resetSession = useCallback(() => {
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(HISTORY_KEY);
    setHistoryCount(0);
    // Trigger backend preview purge (fire-and-forget)
    void fetch('/api/tts/preview/purge', { method: 'POST' }).catch(() => {
      // ignore
    });
  }, []);

  const persistSession = useCallback((payload: SessionSnapshot) => {
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(payload));
      pushHistory(payload);
    } catch (err) {
      console.warn('Failed to persist session to localStorage, aborting', err);
    }
  }, [pushHistory]);

  const getSession = useCallback((): SessionSnapshot | undefined => {
    try {
      const saved = localStorage.getItem(SESSION_KEY);
      if (!saved) return undefined;
      return JSON.parse(saved);
    } catch (err) {
      return undefined;
    }
  }, []);

  return { historyCount, setHistoryCount, loadHistory, pushHistory, resetSession, persistSession, getSession } as const;
}
