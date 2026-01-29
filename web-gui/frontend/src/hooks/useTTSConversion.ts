import { useState, useCallback } from 'react';
import { buildApiUrl } from '@/lib/api';
import { toast as toastSuccess } from 'sonner';

export type Configuration = {
  chunkDepth: number;
  pauseMs: number;
  includeHeading: boolean;
  stripMarkdown: boolean;
  device: 'auto' | 'cuda' | 'cpu' | 'mps';
  iterations: number;
};

export interface AudioHistoryItem {
  id: string;
  filename: string;
  audioUrl: string;
  audioPath: string;
  duration: number | null;
  createdAt: number;
  sourceDocument?: string | null;
}

export function useTTSConversion() {
  const [generatedAudioPath, setGeneratedAudioPath] = useState<string | null>(null);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState<string | null>(null);
  const [generatedAudioDuration, setGeneratedAudioDuration] = useState<number | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [exportingId, setExportingId] = useState<string | null>(null);

  const preview = useCallback(async (voiceId: string, device: string = 'auto') => {
    const response = await fetch(buildApiUrl('/api/tts/preview'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice_id: voiceId, text: 'Hello! This is a short voice preview.', device }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`Preview request failed: ${response.status} ${response.statusText} - ${text}`);
    }

    const result = await response.json();
    if (!result.success || !result.audio_url) {
      throw new Error(result.message || 'Preview failed');
    }

    return buildApiUrl(result.audio_url);
  }, []);

  const convert = useCallback(async (params: {
    documentContent: string;
    selectedVoice: string;
    configuration: Configuration;
    documentFilename?: string | null;
    maxIterationsPerRequest: number;
    onNewEntries?: (entries: AudioHistoryItem[]) => void;
    onSelect?: (id: string) => void;
  }) => {
    const { documentContent, selectedVoice, configuration, documentFilename, onNewEntries, onSelect, maxIterationsPerRequest } = params;

    if (!documentContent) throw new Error('Please upload a document first');
    if (!selectedVoice) throw new Error('Please select a voice');

    const requested = configuration.iterations;
    if (requested > maxIterationsPerRequest) throw new Error(`Maximum iterations per request is ${maxIterationsPerRequest}`);

    setIsConverting(true);
    try {
      const response = await fetch(buildApiUrl('/api/tts/convert'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: documentContent,
          voice_id: selectedVoice,
          filename: documentFilename || 'document.md',
          chunk_depth: configuration.chunkDepth,
          pause_ms: configuration.pauseMs,
          include_heading: configuration.includeHeading,
          strip_markdown: configuration.stripMarkdown,
          device: configuration.device,
          iterations: configuration.iterations,
        }),
      });

      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`TTS request failed: ${response.status} ${response.statusText} - ${text}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || 'TTS failed');
      }

      const outputs = Array.isArray(result.outputs) ? result.outputs : [];
      const entries: AudioHistoryItem[] = outputs.map((item: any) => ({
        id: `${item.audio_url}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        filename: item.filename ?? 'output.wav',
        audioUrl: item.audio_url ? buildApiUrl(item.audio_url) : '',
        audioPath: item.audio_url ?? '',
        duration: item.duration ?? null,
        createdAt: Date.now(),
        sourceDocument: documentFilename || '',
      }));

      if (entries.length === 0 && result.audio_url) {
        entries.push({
          id: `${result.audio_url}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          filename: `${documentFilename || 'output'}_001.wav`,
          audioUrl: buildApiUrl(result.audio_url),
          audioPath: result.audio_url,
          duration: result.duration ?? null,
          createdAt: Date.now(),
        });
      }

      if (entries.length > 0) {
        const latest = entries[0];
        setGeneratedAudioPath(latest.audioPath);
        setGeneratedAudioUrl(latest.audioUrl);
        setGeneratedAudioDuration(latest.duration);
        onSelect?.(latest.id);
        onNewEntries?.(entries);
      }

      return entries;
    } finally {
      setIsConverting(false);
    }
  }, []);

  const exportAudio = useCallback(async (params: { audioPath: string; downloadFilename: string }) => {
    const { audioPath, downloadFilename } = params;
    const response = await fetch(buildApiUrl('/api/export/download'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_url: audioPath, filename: downloadFilename, format: 'wav' }),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`Export request failed: ${response.status} ${response.statusText} - ${text}`);
    }
    const result = await response.json();
    if (!result.success) throw new Error(result.message || 'Export failed');

    const downloadUrl = buildApiUrl(result.download_url || audioPath);
    const fileResponse = await fetch(downloadUrl);
    if (!fileResponse.ok) {
      const text = await fileResponse.text().catch(() => fileResponse.statusText);
      throw new Error(`Download failed: ${fileResponse.status} ${fileResponse.statusText} - ${text}`);
    }

    const blob = await fileResponse.blob();
    const blobUrl = URL.createObjectURL(blob);
    try {
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = downloadFilename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      toastSuccess('Download started!');
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  }, []);

  return {
    generatedAudioPath,
    setGeneratedAudioPath,
    generatedAudioUrl,
    setGeneratedAudioUrl,
    generatedAudioDuration,
    setGeneratedAudioDuration,
    isConverting,
    exportingId,
    setExportingId,
    preview,
    convert,
    exportAudio,
  } as const;
}
