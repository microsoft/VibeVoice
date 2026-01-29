'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Mic2, FileText, Settings, Play, Download, Loader2 } from 'lucide-react';
import { ThemeSwitcher } from '@/components/theme-switcher';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DocumentUpload } from '@/components/document-upload';
import { useVoices } from '@/hooks/useVoices';
import { useSessionPersistence } from '@/hooks/useSessionPersistence';
import { useAudioHistory } from '@/hooks/useAudioHistory';
import { VoiceSelection } from '@/components/voice-selection';
import { ConfigurationPanel, DEFAULT_CONFIGURATION } from '@/components/configuration-panel';
import { AudioPlayer } from '@/components/audio-player';

import { useTTSConversion } from '@/hooks/useTTSConversion';
import { toast } from 'sonner';
import { computePerDocMax, computeTotalMax } from '@/lib/quotas';

export interface Configuration {
  chunkDepth: number;
  pauseMs: number;
  includeHeading: boolean;
  stripMarkdown: boolean;
  device: 'auto' | 'cuda' | 'cpu' | 'mps';
  iterations: number;
}

export interface Voice {
  id: string;
  name: string;
  language: string;
  gender?: 'male' | 'female';
  path: string;
}

const PREVIEW_TEXT = 'Hello! This is a short voice preview.';
const SESSION_KEY = 'vibevoice-session';
const HISTORY_KEY = 'vibevoice-session-history';
const MAX_HISTORY = 20;
// Maximum stored audio history items to prevent localStorage/quota growth

interface SessionSnapshot {
  documentContent: string;
  documentFilename: string;
  selectedVoice: string | null;
  configuration: Configuration;
  activeTab: string;
  audioHistory: AudioHistoryItem[];
  selectedAudioId: string | null;
  savedAt: number;
}

interface AudioHistoryItem {
  id: string;
  filename: string;
  audioUrl: string;
  audioPath: string;
  duration: number | null;
  createdAt: number;
  sourceDocument?: string | null;
}

export default function Dashboard() {
  const sessionSaveRef = useRef<number | null>(null);
  const { historyCount, setHistoryCount, loadHistory, pushHistory, resetSession, persistSession, getSession } = useSessionPersistence();
  const [activeTab, setActiveTab] = useState('upload');
  const [documentContent, setDocumentContent] = useState('');
  const [documentFilename, setDocumentFilename] = useState('');
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  // Server-provided limits (dynamically fetched)
  const [maxIterationsPerRequest, setMaxIterationsPerRequest] = useState<number>(10);
  const { voices, voicesLoading, voicesError, refresh: refreshVoices, setVoices } = useVoices(setMaxIterationsPerRequest);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const [configuration, setConfiguration] = useState<Configuration>({ ...DEFAULT_CONFIGURATION });




    // function body removed; useVoices provides voice state and refresh helper.

  // Voice loading is handled by the `useVoices` hook on mount; use `refreshVoices()` to manually refresh if needed.

  const handleDocumentLoad = (content: string, filename: string) => {
    setDocumentContent(content);
    setDocumentFilename(filename);
    setActiveTab('voice');
  };

  const handleDocumentChange = (content: string, filename: string) => {
    setDocumentContent(content);
    setDocumentFilename(filename);
  };

  const handleVoiceSelect = (voiceId: string) => {
    setSelectedVoice(voiceId);
  };

  const { generatedAudioPath, setGeneratedAudioPath, generatedAudioUrl, setGeneratedAudioUrl, generatedAudioDuration, setGeneratedAudioDuration, isConverting, exportingId, setExportingId, preview, convert, exportAudio } = useTTSConversion();

  const handleVoicePreview = async (voice: Voice) => {
    try {
      const audioUrl = await preview(voice.id, configuration.device);

      // Dispose existing preview audio completely before creating a new one
      if (previewAudioRef.current) {
        try { previewAudioRef.current.pause(); } catch (e) { /* ignore */ }
        try { previewAudioRef.current.src = ''; previewAudioRef.current.onended = null; previewAudioRef.current.onerror = null; } catch (e) { /* ignore */ }
        previewAudioRef.current = null;
      }

      const audio = new Audio(audioUrl);
      audio.preload = 'auto';

      // Cleanup when audio ends
      audio.onended = () => {
        try { audio.src = ''; } catch (e) { /* ignore */ }
        if (previewAudioRef.current === audio) previewAudioRef.current = null;
      };

      try {
        const playPromise = audio.play();
        if (playPromise !== undefined) await playPromise;
        // Only set the ref after play succeeded
        previewAudioRef.current = audio;
        return false;
      } catch (err: any) {
        const name = err && err.name ? err.name : null;
        if (name === 'NotAllowedError' || name === 'NotSupportedError') {
          // Keep the audio instance so user can manually start playback
          previewAudioRef.current = audio;
          toast.error('Autoplay blocked — click to play');
          return true;
        }
        toast.error(err instanceof Error ? err.message : 'Failed to preview voice');
        throw err;
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to preview voice');
      throw error;
    }
  };

  const handleConfigurationChange = (newConfig: Configuration) => {
    setConfiguration(newConfig);
  };

  const handleManualPlay = async (voice: Voice) => {
    try {
      if (previewAudioRef.current) {
        try {
          const playPromise = previewAudioRef.current.play();
          if (playPromise !== undefined) await playPromise;
          toast.success(`Playing preview for "${voice.name}"`);
          return;
        } catch (err: any) {
          const name = err && err.name ? err.name : null;
          if (name === 'NotAllowedError' || name === 'NotSupportedError') {
            toast.error('Autoplay blocked — click to play');
            return;
          }
          toast.error(err instanceof Error ? err.message : 'Failed to play preview');
          throw err;
        }
      }

      // As a fallback, request a preview; because this was initiated by a user gesture it may succeed
      await handleVoicePreview(voice);
    } catch (err) {
      console.error('Manual play failed', err);
      toast.error(err instanceof Error ? err.message : 'Failed to start preview');
      throw err;
    }
  };

  const { audioHistory, setAudioHistory, setPendingRestore, popPendingRestore, applyTruncationTo } = useAudioHistory([]);
  const [selectedAudioId, setSelectedAudioId] = useState<string | null>(null);

  // Restore session from localStorage (legacy behavior): the heavy validation/restore logic remains in component for now.
  useEffect(() => {
    const saved = getSession();
    if (!saved) return;

    try {
      const parsed = saved as any;
      if (typeof parsed.documentContent === 'string') {
        setDocumentContent(parsed.documentContent);
      }
      if (typeof parsed.documentFilename === 'string') {
        setDocumentFilename(parsed.documentFilename);
      }
      if (typeof parsed.selectedVoice === 'string') {
        setSelectedVoice(parsed.selectedVoice);
      }
      if (parsed.configuration) {
        setConfiguration((current) => ({
          ...current,
          ...parsed.configuration,
        }));
      }
      if (typeof parsed.activeTab === 'string') {
        setActiveTab(parsed.activeTab);
      }

      // audioHistory restore and sanitization kept here pending audio-history extraction
      if (Array.isArray(parsed.audioHistory)) {
        // ... existing validation logic retained (unchanged) ...

        let droppedCount = 0;
        const droppedSamples: any[] = [];

        const validHistory: AudioHistoryItem[] = parsed.audioHistory.reduce((acc: AudioHistoryItem[], raw: any) => {
          if (!raw || typeof raw !== 'object') {
            droppedCount++;
            if (droppedSamples.length < 3) droppedSamples.push({ reason: 'not-an-object', sample: raw });
            return acc;
          }

          const { id, audioUrl, audioPath, filename, duration, createdAt, sourceDocument } = raw;

          if (typeof id !== 'string' || id.trim() === '') {
            droppedCount++;
            if (droppedSamples.length < 3) droppedSamples.push({ reason: 'missing-id', sample: raw });
            return acc;
          }

          if (typeof audioUrl !== 'string' || audioUrl.trim() === '') {
            droppedCount++;
            if (droppedSamples.length < 3) droppedSamples.push({ reason: 'missing-audioUrl', sample: raw });
            return acc;
          }

          if (typeof audioPath !== 'string' || audioPath.trim() === '') {
            droppedCount++;
            if (droppedSamples.length < 3) droppedSamples.push({ reason: 'missing-audioPath', sample: raw });
            return acc;
          }

          const sanitized: AudioHistoryItem = {
            id: id.trim(),
            filename: typeof filename === 'string' ? filename : (audioPath.split('/').pop() ?? id.trim()),
            audioUrl: audioUrl.trim(),
            audioPath: audioPath.trim(),
            duration: typeof duration === 'number' ? duration : null,
            createdAt: typeof createdAt === 'number' ? createdAt : Date.now(),
            sourceDocument: typeof sourceDocument === 'string' ? sourceDocument : (typeof parsed.documentFilename === 'string' ? parsed.documentFilename : ''),
          };

          acc.push(sanitized);
          return acc;
        }, []);

        if (droppedCount > 0) {
          toast.info(`Restored session: dropped ${droppedCount} invalid audio history item(s)`);
          console.debug('Dropped invalid audioHistory items during restore:', { droppedCount, droppedSamples });
        }

        // Decide whether to truncate restored history now or defer until server/stored limit is available
        const SERVER_MAX_KEY = 'vibevoice-server-max-iterations';
        const storedMaxRaw = localStorage.getItem(SERVER_MAX_KEY);
        const storedMax = storedMaxRaw ? parseInt(storedMaxRaw, 10) : null;

        let finalHistory = validHistory;
        const applyTruncation = (limitIterations: number) => {
          const perDocMaxRestore = limitIterations * 5;
          const totalMaxRestore = perDocMaxRestore * 3;
          if (finalHistory.length > totalMaxRestore) {
            const dropped = finalHistory.length - totalMaxRestore;
            finalHistory = finalHistory.slice(0, totalMaxRestore);
            toast.info(`Restored session: truncated audio history by ${dropped} item(s)`);
            console.debug('Truncated restored audioHistory to max length:', { original: validHistory.length, kept: finalHistory.length });
          }
        };

        if (typeof storedMax === 'number' && !Number.isNaN(storedMax)) {
          const { final } = applyTruncationTo(finalHistory, storedMax);
          setAudioHistory(final);

          try {
            const sanitizedPayload = {
              documentContent: typeof parsed.documentContent === 'string' ? parsed.documentContent : '',
              documentFilename: typeof parsed.documentFilename === 'string' ? parsed.documentFilename : '',
              selectedVoice: typeof parsed.selectedVoice === 'string' ? parsed.selectedVoice : null,
              configuration: parsed.configuration ?? configuration,
              activeTab: typeof parsed.activeTab === 'string' ? parsed.activeTab : 'upload',
              audioHistory: final,
              selectedAudioId: typeof parsed.selectedAudioId === 'string' ? parsed.selectedAudioId : null,
              savedAt: Date.now(),
            };
            persistSession(sanitizedPayload as any);
            console.debug('Persisted sanitized session to localStorage', { kept: final.length });
          } catch (err) {
            console.warn('Failed to persist sanitized session:', err);
          }
        } else {
          const { final } = applyTruncationTo(finalHistory, maxIterationsPerRequest);
          setAudioHistory(final);
          try {
            const sanitizedPayload = {
              documentContent: typeof parsed.documentContent === 'string' ? parsed.documentContent : '',
              documentFilename: typeof parsed.documentFilename === 'string' ? parsed.documentFilename : '',
              selectedVoice: typeof parsed.selectedVoice === 'string' ? parsed.selectedVoice : null,
              configuration: parsed.configuration ?? configuration,
              activeTab: typeof parsed.activeTab === 'string' ? parsed.activeTab : 'upload',
              audioHistory: final,
              selectedAudioId: typeof parsed.selectedAudioId === 'string' ? parsed.selectedAudioId : null,
              savedAt: Date.now(),
            };
            persistSession(sanitizedPayload as any);
            console.debug('Persisted sanitized session to localStorage (pending truncation)', { kept: final.length });
          } catch (err) {
            console.warn('Failed to persist sanitized session:', err);
          }

          setPendingRestore(validHistory);
        }
      }
      if (typeof parsed.selectedAudioId === 'string') {
        setSelectedAudioId(parsed.selectedAudioId);
      }
    } catch (error) {
      console.warn('Failed to restore session:', error);
    }

    // Cleanup preview audio on unmount
    return () => {
      if (previewAudioRef.current) {
        try {
          previewAudioRef.current.pause();
        } catch (e) {
          // ignore
        }
        try {
          previewAudioRef.current.src = '';
          previewAudioRef.current.onended = null;
          previewAudioRef.current.onerror = null;
        } catch (e) {
          // ignore
        }
        previewAudioRef.current = null;
      }
    };
  }, []);

  // After server limit is fetched, if a pending restored history exists, apply truncation
  // Also, if the current in-memory history exceeds the newly fetched limit (e.g. server lowered limits), truncate it.
  useEffect(() => {
    try {
      const pending = popPendingRestore();
      const limit = maxIterationsPerRequest;

      if (pending && typeof limit === 'number' && limit > 0) {
        let finalHistory = pending;
        const perDocMaxRestore = computePerDocMax(limit);
        const totalMaxRestore = computeTotalMax(limit);
        if (finalHistory.length > totalMaxRestore) {
          const dropped = finalHistory.length - totalMaxRestore;
          finalHistory = finalHistory.slice(0, totalMaxRestore);
          toast.info(`Restored session: truncated audio history by ${dropped} item(s)`);
          console.debug('Truncated restored audioHistory to max length (post-config):', { original: pending.length, kept: finalHistory.length });
        }

        setAudioHistory(finalHistory);

        try {
          const sanitizedPayload = {
            documentContent: '',
            documentFilename: '',
            selectedVoice: null,
            configuration: configuration,
            activeTab: 'upload',
            audioHistory: finalHistory,
            selectedAudioId: null,
            savedAt: Date.now(),
          } as any;
          const existing = localStorage.getItem(SESSION_KEY);
          if (existing) {
            try {
              const parsedExisting = JSON.parse(existing);
              sanitizedPayload.documentContent = typeof parsedExisting.documentContent === 'string' ? parsedExisting.documentContent : '';
              sanitizedPayload.documentFilename = typeof parsedExisting.documentFilename === 'string' ? parsedExisting.documentFilename : '';
              sanitizedPayload.selectedVoice = typeof parsedExisting.selectedVoice === 'string' ? parsedExisting.selectedVoice : null;
              sanitizedPayload.configuration = parsedExisting.configuration ?? configuration;
              sanitizedPayload.activeTab = typeof parsedExisting.activeTab === 'string' ? parsedExisting.activeTab : 'upload';
              sanitizedPayload.selectedAudioId = typeof parsedExisting.selectedAudioId === 'string' ? parsedExisting.selectedAudioId : null;
            } catch (err) {
              // ignore, we'll persist defaults
            }
          }
          localStorage.setItem(SESSION_KEY, JSON.stringify(sanitizedPayload));
          console.debug('Persisted sanitized session to localStorage (post-config truncation)', { kept: finalHistory.length });
        } catch (err) {
          console.warn('Failed to persist sanitized session after applying server limit:', err);
        }

        // pending cleared by popPendingRestore() above
        return;
      }

      // No pending restore. Ensure the currently-loaded audioHistory adheres to the newly reported limit.
      if (!pending && typeof limit === 'number' && limit > 0) {
        const perDocMax = computePerDocMax(limit);
        const totalMax = computeTotalMax(limit);
        setAudioHistory((prev) => {
          if (prev.length > totalMax) {
            const dropped = prev.length - totalMax;
            const finalHistory = prev.slice(0, totalMax);

            try {
              const sanitizedPayload = {
                documentContent: '',
                documentFilename: '',
                selectedVoice: null,
                configuration: configuration,
                activeTab: 'upload',
                audioHistory: finalHistory,
                selectedAudioId: null,
                savedAt: Date.now(),
              } as any;
              const existing = localStorage.getItem(SESSION_KEY);
              if (existing) {
                try {
                  const parsedExisting = JSON.parse(existing);
                  sanitizedPayload.documentContent = typeof parsedExisting.documentContent === 'string' ? parsedExisting.documentContent : '';
                  sanitizedPayload.documentFilename = typeof parsedExisting.documentFilename === 'string' ? parsedExisting.documentFilename : '';
                  sanitizedPayload.selectedVoice = typeof parsedExisting.selectedVoice === 'string' ? parsedExisting.selectedVoice : null;
                  sanitizedPayload.configuration = parsedExisting.configuration ?? configuration;
                  sanitizedPayload.activeTab = typeof parsedExisting.activeTab === 'string' ? parsedExisting.activeTab : 'upload';
                  sanitizedPayload.selectedAudioId = typeof parsedExisting.selectedAudioId === 'string' ? parsedExisting.selectedAudioId : null;
                } catch (err) {
                  // ignore, we'll persist defaults
                }
              }
              localStorage.setItem(SESSION_KEY, JSON.stringify(sanitizedPayload));
              toast.info(`Restored session: truncated audio history by ${dropped} item(s)`);
              console.debug('Truncated loaded audioHistory to max length (post-config):', { original: prev.length, kept: finalHistory.length });
            } catch (err) {
              console.warn('Failed to persist sanitized session after applying server limit:', err);
            }

            return finalHistory;
          }
          return prev;
        });
      }
    } catch (err) {
      // swallow
    }
  }, [maxIterationsPerRequest, audioHistory]);

  // History helpers are provided by the `useSessionPersistence` hook (loadHistory, pushHistory, etc.)

  useEffect(() => {
    setHistoryCount(loadHistory().length);
  }, [loadHistory, setHistoryCount]);

  const resetSessionHandler = () => {
    const confirmed = window.confirm('Clear current session and saved versions?');
    if (!confirmed) return;
    resetSession();
    setDocumentContent('');
    setDocumentFilename('');
    setSelectedVoice(null);
    setConfiguration({ ...DEFAULT_CONFIGURATION });
    setActiveTab('upload');
    setAudioHistory([]);
    setSelectedAudioId(null);
    toast.success('Session cleared');
  };

  const saveVersion = () => {
    pushHistory({
      documentContent,
      documentFilename,
      selectedVoice,
      configuration,
      activeTab,
      audioHistory,
      selectedAudioId,
      savedAt: Date.now(),
    });
    toast.success('Version saved');
  };

  useEffect(() => {
    if (sessionSaveRef.current) {
      window.clearTimeout(sessionSaveRef.current);
    }
    sessionSaveRef.current = window.setTimeout(() => {
      const payload: SessionSnapshot = {
        documentContent,
        documentFilename,
        selectedVoice,
        configuration,
        activeTab,
        audioHistory,
        selectedAudioId,
        savedAt: Date.now(),
      };
      // Delegate persistence and history logic to the session hook
      persistSession(payload);
    }, 400);

    return () => {
      if (sessionSaveRef.current) {
        window.clearTimeout(sessionSaveRef.current);
      }
    };
  }, [documentContent, documentFilename, selectedVoice, configuration, activeTab, audioHistory, selectedAudioId, persistSession]);

  const handleConvert = async () => {
    try {
      // Preflight quota checks
      const perDocMax = computePerDocMax(maxIterationsPerRequest);
      const totalMax = computeTotalMax(maxIterationsPerRequest);

      if (audioHistory.length >= totalMax) {
        toast.error('Total quota exceeded');
        return;
      }

      const perDocCount = audioHistory.filter((h) => h.sourceDocument === (documentFilename || '')).length;
      if (perDocCount >= perDocMax) {
        toast.error('Per-document quota exceeded');
        return;
      }

      const entries = await convert({
        documentContent,
        selectedVoice: selectedVoice ?? '',
        configuration,
        documentFilename,
        maxIterationsPerRequest,
        onNewEntries: (entries) => {
          setAudioHistory((prev) => {
            const merged = [...entries, ...prev];
            const totalMaxMerge = computeTotalMax(maxIterationsPerRequest);
            return merged.slice(0, totalMaxMerge);
          });
        },
        onSelect: (id) => setSelectedAudioId(id),
      });

      if (entries && entries.length > 0) {
        toast.success('TTS conversion completed!');
      }
    } catch (err) {
      toast.error(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleExport = async (item?: AudioHistoryItem) => {
    const selectedItem = item ?? audioHistory.find((entry) => entry.id === selectedAudioId) ?? null;
    const audioPath = selectedItem?.audioPath || generatedAudioPath;
    const downloadFilename = selectedItem?.filename || `${documentFilename || 'output'}.wav`;

    if (!audioPath) {
      toast.error('No audio to export');
      return;
    }

    try {
      await exportAudio({ audioPath, downloadFilename });
    } catch (error) {
      toast.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 flex">
            <Mic2 className="h-6 w-6" />
            <span className="ml-2 hidden font-bold sm:inline-block">
              VibeVoice-Narrator
            </span>
          </div>
          <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
            <nav className="flex items-center">
              <ThemeSwitcher />
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4 lg:w-[500px]">
            <TabsTrigger value="upload">
              <FileText className="h-4 w-4 mr-2" />
              Document
            </TabsTrigger>
            <TabsTrigger value="voice">
              <Play className="h-4 w-4 mr-2" />
              Voice
            </TabsTrigger>
            <TabsTrigger value="settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </TabsTrigger>
            <TabsTrigger value="player">
              <Play className="h-4 w-4 mr-2" />
              Player
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="mt-6">
            <DocumentUpload
              onDocumentLoad={handleDocumentLoad}
              onDocumentChange={handleDocumentChange}
              initialContent={documentContent}
              initialFilename={documentFilename}
            />
          </TabsContent>

          <TabsContent value="voice" className="mt-6">
            {voicesLoading ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Play className="h-5 w-5" />
                    Voice Selection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">Loading voices...</p>
                </CardContent>
              </Card>
            ) : voicesError ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Play className="h-5 w-5" />
                    Voice Selection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-destructive">Failed to load voices.</p>
                  <p className="text-xs text-muted-foreground mt-2">{voicesError}</p>
                  <Button variant="outline" size="sm" className="mt-4" onClick={refreshVoices}>
                    Retry
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <VoiceSelection
                voices={voices}
                selectedVoice={selectedVoice}
                onVoiceSelect={handleVoiceSelect}
                onVoicePreview={handleVoicePreview}
                onVoiceManualPlay={handleManualPlay}
                showSelectionToast={false}
              />
            )}
          </TabsContent>

          <TabsContent value="settings" className="mt-6">
            <ConfigurationPanel
              configuration={configuration}
              onConfigurationChange={handleConfigurationChange}
              maxIterations={maxIterationsPerRequest}
            />
          </TabsContent>

          <TabsContent value="player" className="mt-6">
            <AudioPlayer
              audioUrl={generatedAudioUrl}
              duration={generatedAudioDuration ?? undefined}
              title={generatedAudioUrl ? (documentFilename || 'Generated Audio') : 'No Audio Generated'}
            />
          </TabsContent>
        </Tabs>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border/60 bg-muted/30 px-4 py-3 text-sm">
          <div className="text-muted-foreground">
            Session autosaved • Versions: {historyCount}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={saveVersion} disabled={!documentContent.trim()}>
              Save Version
            </Button>
            <Button variant="ghost" size="sm" onClick={resetSessionHandler}>
              Reset Session
            </Button>
          </div>
        </div>

        {audioHistory.length > 0 && (
          <div className="mt-6 rounded-lg border border-border/60 bg-background/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-semibold">Audio History</div>
              <div className="text-xs text-muted-foreground">
                {audioHistory.length} file{audioHistory.length === 1 ? '' : 's'}
              </div>
            </div>
            <div className="space-y-2">
              {audioHistory.map((item) => (
                <div
                  key={item.id}
                  className={`flex flex-wrap items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm ${
                    selectedAudioId === item.id
                      ? 'border-primary/60 bg-primary/5'
                      : 'border-border/60 bg-muted/20'
                  }`}
                >
                  <div className="flex flex-col">
                    <span className="font-medium">{item.filename}</span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(item.createdAt).toLocaleTimeString()} • {item.duration ? `${item.duration.toFixed(1)}s` : 'duration unknown'}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSelectedAudioId(item.id);
                        setGeneratedAudioPath(item.audioPath);
                        setGeneratedAudioUrl(item.audioUrl);
                        setGeneratedAudioDuration(item.duration);
                        setActiveTab('player');
                      }}
                    >
                      Play
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={exportingId === item.id}
                      onClick={async () => {
                        setSelectedAudioId(item.id);
                        setExportingId(item.id);
                        try {
                          await handleExport(item);
                        } finally {
                          setExportingId(null);
                        }
                      }}
                      aria-disabled={exportingId === item.id}
                    >
                      {exportingId === item.id ? (
                        <>
                          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                          Downloading...
                        </>
                      ) : (
                        'Download'
                      )}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Bar */}
        {documentContent && (
          <div className="mt-6 flex items-center justify-between gap-4">
            <div className="text-sm text-muted-foreground">
              {documentFilename && (
                <span className="mr-4">
                  <FileText className="h-4 w-4 inline mr-1" />
                  {documentFilename}
                </span>
              )}
              {selectedVoice && (
                <span className="mr-4">
                  <Play className="h-4 w-4 inline mr-1" />
                  {voices.find(v => v.id === selectedVoice)?.name || selectedVoice}
                </span>
              )}
              {isConverting && (
                <span className="text-primary animate-pulse">
                  Converting...
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <Button
                onClick={handleConvert}
                size="lg"
                className="flex-1"
                disabled={isConverting}
              >
                <Play className="h-4 w-4 mr-2" />
                {isConverting ? 'Converting...' : 'Convert to Speech'}
              </Button>
              <Button
                onClick={async () => {
                  if (!generatedAudioUrl && !selectedAudioId) {
                    toast.error('No audio to export');
                    return;
                  }
                  const id = selectedAudioId ?? 'export-selected';
                  setExportingId(id);
                  try {
                    await handleExport();
                  } finally {
                    setExportingId(null);
                  }
                }}
                variant="outline"
                size="lg"
                disabled={!generatedAudioUrl || isConverting || exportingId !== null}
              >
                {exportingId && exportingId === (selectedAudioId ?? 'export-selected') ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export Audio
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
