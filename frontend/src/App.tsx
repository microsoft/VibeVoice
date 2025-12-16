import { useState, useCallback, useRef } from 'react';
import { Play, Square, Download, Waves } from 'lucide-react';
import {
  TextInput,
  VoiceSelector,
  ControlPanel,
  StreamingPreview,
  MetricsDisplay,
  LogConsole,
  ThemeToggle,
  WaveformVisualizer,
} from '@/components';
import {
  useWebSocket,
  useAudioStream,
  useVoiceConfig,
  useTheme,
} from '@/hooks';
import type { LogEntry, AudioMetrics } from '@/types';

const DEFAULT_TEXT = `Enter your text here and click "Start" to instantly hear the VibeVoice-Realtime TTS output audio.`;

export default function App() {
  const { isDark, toggle: toggleTheme } = useTheme();
  const { voices, selectedVoice, setSelectedVoice, loading: voicesLoading } = useVoiceConfig();
  
  const [text, setText] = useState(DEFAULT_TEXT);
  const [cfg, setCfg] = useState(1.5);
  const [steps, setSteps] = useState(5);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logIdRef = useRef(0);
  const [metrics, setMetrics] = useState<AudioMetrics>({ modelGenerated: 0, playbackElapsed: 0 });

  const audioStream = useAudioStream();

  const addLog = useCallback((entry: Omit<LogEntry, 'id'>) => {
    logIdRef.current += 1;
    const newId = logIdRef.current;
    setLogs((currentLogs) => [...currentLogs.slice(-399), { ...entry, id: newId }]);
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setMetrics({ modelGenerated: 0, playbackElapsed: 0 });
  }, []);

  const handleMetricsUpdate = useCallback((generated: number) => {
    setMetrics((prev) => ({ ...prev, modelGenerated: generated }));
  }, []);

  const handleComplete = useCallback(() => {
    audioStream.markComplete();
  }, [audioStream]);

  const websocket = useWebSocket({
    onAudioChunk: audioStream.appendAudio,
    onLog: addLog,
    onMetricsUpdate: handleMetricsUpdate,
    onComplete: handleComplete,
  });

  const handleStart = useCallback(() => {
    if (!text.trim()) {
      addLog({ timestamp: new Date().toISOString(), message: '[Error] Please enter some text', type: 'error' });
      return;
    }

    clearLogs();
    audioStream.start(() => {
      websocket.disconnect();
    });
    
    websocket.connect({
      text,
      cfg,
      steps,
      voice: selectedVoice,
    });
  }, [text, cfg, steps, selectedVoice, clearLogs, audioStream, websocket, addLog]);

  const handleStop = useCallback(() => {
    websocket.disconnect();
    audioStream.stop();
    addLog({ timestamp: new Date().toISOString(), message: '[Frontend] Playback stopped', type: 'info' });
  }, [websocket, audioStream, addLog]);

  const handleReset = useCallback(() => {
    setCfg(1.5);
    setSteps(5);
    addLog({ timestamp: new Date().toISOString(), message: '[Frontend] Controls reset to defaults', type: 'info' });
  }, [addLog]);

  const handleSave = useCallback(() => {
    audioStream.saveAudio();
    addLog({ timestamp: new Date().toISOString(), message: '[Frontend] Audio download triggered', type: 'success' });
  }, [audioStream, addLog]);

  const isActive = websocket.isStreaming || audioStream.isPlaying;

  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-100 via-surface-50 to-primary-50/30 
                    dark:from-surface-950 dark:via-surface-900 dark:to-primary-950/20
                    py-8 px-4 sm:px-6 transition-colors duration-300">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 shadow-lg shadow-primary-500/30">
              <Waves className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-surface-900 dark:text-white">
                VibeVoice
              </h1>
              <p className="text-sm text-surface-500 dark:text-surface-400">
                Realtime Text-to-Speech
              </p>
            </div>
          </div>
          <ThemeToggle isDark={isDark} onToggle={toggleTheme} />
        </header>

        {/* Main Card */}
        <main className="card p-6 sm:p-8 space-y-6">
          {/* Text Input */}
          <TextInput
            value={text}
            onChange={setText}
            disabled={isActive}
          />

          {/* Streaming Preview */}
          <StreamingPreview
            text={text}
            isStreaming={isActive}
          />

          {/* Voice Selector */}
          <VoiceSelector
            voices={voices}
            value={selectedVoice}
            onChange={setSelectedVoice}
            disabled={isActive}
            loading={voicesLoading}
          />

          {/* Control Panel */}
          <ControlPanel
            cfg={cfg}
            steps={steps}
            onCfgChange={setCfg}
            onStepsChange={setSteps}
            onReset={handleReset}
            disabled={isActive}
          />

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3">
            {!isActive ? (
              <button
                onClick={handleStart}
                disabled={voicesLoading || !text.trim()}
                className="btn-primary"
              >
                <Play className="w-5 h-5" />
                Start
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="btn-primary bg-red-500 hover:bg-red-600 shadow-red-500/25 hover:shadow-red-500/30"
              >
                <Square className="w-5 h-5" />
                Stop
              </button>
            )}

            <button
              onClick={handleSave}
              disabled={!audioStream.canSave}
              className="btn-secondary"
            >
              <Download className="w-4 h-4" />
              Save Audio
            </button>
          </div>

          {/* Waveform Visualizer */}
          <WaveformVisualizer
            isActive={audioStream.isPlaying}
            getWaveformData={audioStream.getWaveformData}
          />

          {/* Metrics */}
          <MetricsDisplay
            metrics={{
              modelGenerated: metrics.modelGenerated,
              playbackElapsed: audioStream.playbackSeconds,
            }}
          />

          {/* Log Console */}
          <LogConsole logs={logs} onClear={clearLogs} />
        </main>

        {/* Footer */}
        <footer className="text-center text-sm text-surface-500 dark:text-surface-400">
          <p>
            Powered by{' '}
            <a
              href="https://github.com/microsoft/VibeVoice"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 dark:text-primary-400 hover:underline"
            >
              VibeVoice-Realtime-0.5B
            </a>
          </p>
        </footer>
      </div>
    </div>
  );
}

