export interface VoiceConfig {
  voices: string[];
  default_voice: string | null;
}

export interface TTSParams {
  text: string;
  cfg: number;
  steps: number;
  voice: string;
}

export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
}

export interface AudioMetrics {
  modelGenerated: number;
  playbackElapsed: number;
}

export interface WebSocketMessage {
  type: 'log';
  event: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'streaming' | 'error';

