import { useCallback, useRef, useState } from 'react';
import type { ConnectionStatus, TTSParams, WebSocketMessage, LogEntry } from '@/types';

interface UseWebSocketOptions {
  onAudioChunk: (chunk: Float32Array) => void;
  onLog: (entry: Omit<LogEntry, 'id'>) => void;
  onMetricsUpdate: (generated: number) => void;
  onComplete: () => void;
}

export function useWebSocket({
  onAudioChunk,
  onLog,
  onMetricsUpdate,
  onComplete,
}: UseWebSocketOptions) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const socketRef = useRef<WebSocket | null>(null);
  const firstChunkReceivedRef = useRef(false);

  const formatTimestamp = () => {
    const d = new Date();
    const pad = (n: number) => n.toString().padStart(2, '0');
    const pad3 = (n: number) => n.toString().padStart(3, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad3(d.getMilliseconds())}`;
  };

  const log = useCallback(
    (message: string, type: LogEntry['type'] = 'info') => {
      onLog({ timestamp: formatTimestamp(), message, type });
    },
    [onLog]
  );

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const payload: WebSocketMessage = JSON.parse(event.data);
          if (payload.type !== 'log') {
            log(`[Log] ${event.data}`, 'info');
            return;
          }

          const { event: eventType, data } = payload;

          switch (eventType) {
            case 'backend_request_received':
              log('[Backend] Request received', 'info');
              break;
            case 'backend_first_chunk_sent':
              log('[Backend] Sent first audio chunk', 'success');
              break;
            case 'model_progress':
              if (typeof data.generated_sec === 'number') {
                onMetricsUpdate(data.generated_sec as number);
              }
              return;
            case 'generation_error':
              log(`[Error] Generation error: ${data.message || 'Unknown'}`, 'error');
              break;
            case 'backend_error':
              log(`[Error] Backend error: ${data.message || 'Unknown'}`, 'error');
              break;
            case 'client_disconnected':
              log('[Frontend] Client disconnected', 'warning');
              break;
            case 'backend_stream_complete':
              log('[Backend] Stream complete', 'success');
              onComplete();
              break;
            default:
              log(`[Log] Event: ${eventType}`, 'info');
          }
        } catch {
          log(`[Error] Failed to parse: ${event.data}`, 'error');
        }
        return;
      }

      if (event.data instanceof ArrayBuffer) {
        const rawBuffer = event.data.slice(0);
        const view = new DataView(rawBuffer);
        const floatChunk = new Float32Array(view.byteLength / 2);

        for (let i = 0; i < floatChunk.length; i++) {
          floatChunk[i] = view.getInt16(i * 2, true) / 32768;
        }

        if (!firstChunkReceivedRef.current) {
          firstChunkReceivedRef.current = true;
          log('[Frontend] Received first audio chunk', 'success');
        }

        onAudioChunk(floatChunk);
      }
    },
    [log, onAudioChunk, onMetricsUpdate, onComplete]
  );

  const connect = useCallback(
    (params: TTSParams) => {
      if (socketRef.current) {
        socketRef.current.close();
      }

      firstChunkReceivedRef.current = false;
      setStatus('connecting');

      const searchParams = new URLSearchParams();
      searchParams.set('text', params.text);
      searchParams.set('cfg', params.cfg.toFixed(3));
      searchParams.set('steps', params.steps.toString());
      if (params.voice) {
        searchParams.set('voice', params.voice);
      }

      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/stream?${searchParams.toString()}`;

      log(`[Frontend] Connecting with CFG=${params.cfg.toFixed(2)}, Steps=${params.steps}, Voice=${params.voice || 'default'}`, 'info');

      const socket = new WebSocket(wsUrl);
      socket.binaryType = 'arraybuffer';
      socketRef.current = socket;

      socket.onopen = () => {
        setStatus('streaming');
        log('[Frontend] Connected to server', 'success');
      };

      socket.onmessage = handleMessage;

      socket.onerror = (err) => {
        console.error('WebSocket error:', err);
        log('[Error] WebSocket connection error', 'error');
        setStatus('error');
      };

      socket.onclose = () => {
        socketRef.current = null;
        if (status !== 'error') {
          setStatus('disconnected');
        }
      };
    },
    [handleMessage, log, status]
  );

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setStatus('disconnected');
    log('[Frontend] Disconnected', 'info');
  }, [log]);

  return {
    status,
    connect,
    disconnect,
    isConnected: status === 'connected' || status === 'streaming',
    isStreaming: status === 'streaming',
  };
}

