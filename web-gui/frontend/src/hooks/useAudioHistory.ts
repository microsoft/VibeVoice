import { useRef, useState, useCallback } from 'react';
import { computePerDocMax, computeTotalMax } from '@/lib/quotas';

export interface AudioHistoryItem {
  id: string;
  filename: string;
  audioUrl: string;
  audioPath: string;
  duration: number | null;
  createdAt: number;
  sourceDocument?: string | null;
}

export function useAudioHistory(initial: AudioHistoryItem[] = []) {
  const [audioHistory, setAudioHistory] = useState<AudioHistoryItem[]>(initial);
  const pendingRestoreRef = useRef<AudioHistoryItem[] | undefined>(undefined);

  const setPendingRestore = useCallback((items: AudioHistoryItem[] | undefined) => {
    pendingRestoreRef.current = items;
  }, []);

  const popPendingRestore = useCallback(() => {
    const val = pendingRestoreRef.current;
    pendingRestoreRef.current = undefined;
    return val;
  }, []);

  const addAudioItem = useCallback((item: AudioHistoryItem) => {
    setAudioHistory((prev) => [item, ...prev]);
  }, []);

  const truncateToLimit = useCallback((limitIterations: number) => {
    const perDocMax = computePerDocMax(limitIterations);
    const totalMax = computeTotalMax(limitIterations);
    if (audioHistory.length > totalMax) {
      const final = audioHistory.slice(0, totalMax);
      setAudioHistory(final);
      return { final, dropped: audioHistory.length - totalMax };
    }
    return { final: audioHistory, dropped: 0 };
  }, [audioHistory]);

  const applyTruncationTo = useCallback((items: AudioHistoryItem[], limitIterations: number) => {
    const perDocMax = computePerDocMax(limitIterations);
    const totalMax = computeTotalMax(limitIterations);
    if (items.length > totalMax) {
      return { final: items.slice(0, totalMax), dropped: items.length - totalMax };
    }
    return { final: items, dropped: 0 };
  }, []);

  return {
    audioHistory,
    setAudioHistory,
    addAudioItem,
    setPendingRestore,
    popPendingRestore,
    truncateToLimit,
    applyTruncationTo,
  } as const;
}
