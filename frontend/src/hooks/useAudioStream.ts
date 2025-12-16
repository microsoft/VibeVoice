import { useCallback, useRef, useState } from 'react';

const SAMPLE_RATE = 24000;
const BUFFER_SIZE = 2048;
const PREBUFFER_SEC = 0.1;

export interface AudioStreamState {
  isPlaying: boolean;
  playbackSeconds: number;
  canSave: boolean;
}

export function useAudioStream() {
  const [state, setState] = useState<AudioStreamState>({
    isPlaying: false,
    playbackSeconds: 0,
    canSave: false,
  });

  const audioCtxRef = useRef<AudioContext | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const bufferRef = useRef<Float32Array>(new Float32Array(0));
  const playbackSamplesRef = useRef(0);
  const hasStartedPlaybackRef = useRef(false);
  const silentFrameCountRef = useRef(0);
  const recordedChunksRef = useRef<ArrayBuffer[]>([]);
  const recordedSamplesRef = useRef(0);
  const streamCompleteRef = useRef(false);
  const playbackTimerRef = useRef<number | null>(null);
  const onStopCallbackRef = useRef<(() => void) | null>(null);

  const appendAudio = useCallback((chunk: Float32Array) => {
    const merged = new Float32Array(bufferRef.current.length + chunk.length);
    merged.set(bufferRef.current, 0);
    merged.set(chunk, bufferRef.current.length);
    bufferRef.current = merged;

    const pcmBuffer = new ArrayBuffer(chunk.length * 2);
    const pcmView = new DataView(pcmBuffer);
    for (let i = 0; i < chunk.length; i++) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      pcmView.setInt16(i * 2, sample * 32767, true);
    }
    recordedChunksRef.current.push(pcmBuffer);
    recordedSamplesRef.current += chunk.length;
  }, []);

  const pullAudio = useCallback((frameCount: number): Float32Array => {
    const available = bufferRef.current.length;
    if (available === 0) {
      return new Float32Array(frameCount);
    }
    if (available <= frameCount) {
      const chunk = bufferRef.current;
      bufferRef.current = new Float32Array(0);
      if (chunk.length < frameCount) {
        const padded = new Float32Array(frameCount);
        padded.set(chunk, 0);
        return padded;
      }
      return chunk;
    }
    const chunk = bufferRef.current.subarray(0, frameCount);
    bufferRef.current = bufferRef.current.subarray(frameCount);
    return chunk;
  }, []);

  const updatePlaybackTime = useCallback(() => {
    setState((prev) => ({
      ...prev,
      playbackSeconds: playbackSamplesRef.current / SAMPLE_RATE,
    }));
  }, []);

  const startPlaybackTimer = useCallback(() => {
    if (playbackTimerRef.current) return;
    playbackTimerRef.current = window.setInterval(updatePlaybackTime, 250);
  }, [updatePlaybackTime]);

  const stopPlaybackTimer = useCallback(() => {
    if (playbackTimerRef.current) {
      clearInterval(playbackTimerRef.current);
      playbackTimerRef.current = null;
    }
  }, []);

  const teardown = useCallback(() => {
    stopPlaybackTimer();
    
    if (scriptNodeRef.current) {
      try {
        scriptNodeRef.current.disconnect();
      } catch {}
      scriptNodeRef.current.onaudioprocess = null;
      scriptNodeRef.current = null;
    }
    
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
      audioCtxRef.current = null;
    }
  }, [stopPlaybackTimer]);

  const stop = useCallback(() => {
    teardown();
    hasStartedPlaybackRef.current = false;
    silentFrameCountRef.current = 0;
    
    setState((prev) => ({
      ...prev,
      isPlaying: false,
      canSave: recordedSamplesRef.current > 0 && streamCompleteRef.current,
    }));

    if (onStopCallbackRef.current) {
      onStopCallbackRef.current();
      onStopCallbackRef.current = null;
    }
  }, [teardown]);

  const start = useCallback((onStop?: () => void) => {
    teardown();

    bufferRef.current = new Float32Array(0);
    playbackSamplesRef.current = 0;
    hasStartedPlaybackRef.current = false;
    silentFrameCountRef.current = 0;
    recordedChunksRef.current = [];
    recordedSamplesRef.current = 0;
    streamCompleteRef.current = false;
    onStopCallbackRef.current = onStop || null;

    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    audioCtxRef.current = new AudioContextClass({ sampleRate: SAMPLE_RATE });
    scriptNodeRef.current = audioCtxRef.current.createScriptProcessor(BUFFER_SIZE, 0, 1);

    const minBufferSamples = Math.floor(SAMPLE_RATE * PREBUFFER_SEC);

    scriptNodeRef.current.onaudioprocess = (event) => {
      const output = event.outputBuffer.getChannelData(0);
      const needPrebuffer = !hasStartedPlaybackRef.current;
      const isComplete = streamCompleteRef.current;

      if (needPrebuffer) {
        if (bufferRef.current.length >= minBufferSamples || isComplete) {
          hasStartedPlaybackRef.current = true;
          startPlaybackTimer();
        } else {
          output.fill(0);
          return;
        }
      }

      const chunk = pullAudio(output.length);
      output.set(chunk);

      if (hasStartedPlaybackRef.current) {
        playbackSamplesRef.current += output.length;
      }

      if (isComplete && bufferRef.current.length === 0 && chunk.every((s) => s === 0)) {
        silentFrameCountRef.current += 1;
        if (silentFrameCountRef.current >= 4) {
          stop();
        }
      } else {
        silentFrameCountRef.current = 0;
      }
    };

    scriptNodeRef.current.connect(audioCtxRef.current.destination);
    setState({ isPlaying: true, playbackSeconds: 0, canSave: false });
  }, [teardown, pullAudio, startPlaybackTimer, stop]);

  const markComplete = useCallback(() => {
    streamCompleteRef.current = true;
    setState((prev) => ({
      ...prev,
      canSave: recordedSamplesRef.current > 0,
    }));
  }, []);

  const createWavBlob = useCallback((): Blob | null => {
    if (recordedSamplesRef.current === 0) return null;

    const totalSamples = recordedSamplesRef.current;
    const wavBuffer = new ArrayBuffer(44 + totalSamples * 2);
    const view = new DataView(wavBuffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + totalSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, totalSamples * 2, true);

    const pcmData = new Int16Array(wavBuffer, 44, totalSamples);
    let offset = 0;
    for (const chunk of recordedChunksRef.current) {
      const chunkData = new Int16Array(chunk);
      pcmData.set(chunkData, offset);
      offset += chunkData.length;
    }

    return new Blob([wavBuffer], { type: 'audio/wav' });
  }, []);

  const saveAudio = useCallback(() => {
    const blob = createWavBlob();
    if (!blob) return;

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.href = url;
    link.download = `vibevoice_audio_${timestamp}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [createWavBlob]);

  const getWaveformData = useCallback((): Float32Array => {
    return bufferRef.current.length > 0 ? bufferRef.current.slice(-2048) : new Float32Array(0);
  }, []);

  return {
    ...state,
    start,
    stop,
    appendAudio,
    markComplete,
    saveAudio,
    getWaveformData,
  };
}

