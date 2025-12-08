/**
 * StreamingClient manages WebSocket connection for TTS streaming
 */
export class StreamingClient {
  constructor(onAudioChunk, onLogMessage, onError, onClose) {
    this.socket = null;
    this.onAudioChunk = onAudioChunk;
    this.onLogMessage = onLogMessage;
    this.onError = onError;
    this.onClose = onClose;
  }

  /**
   * Connect to WebSocket server
   */
  connect(text, cfg, steps, voice) {
    this.close();

    const params = new URLSearchParams();
    params.set('text', text);
    if (!Number.isNaN(cfg)) {
      params.set('cfg', cfg.toFixed(3));
    }
    if (!Number.isNaN(steps)) {
      params.set('steps', steps.toString());
    }
    if (voice) {
      params.set('voice', voice);
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/stream?${params.toString()}`;

    this.socket = new WebSocket(wsUrl);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        this.handleLogMessage(event.data);
        return;
      }

      if (!(event.data instanceof ArrayBuffer)) {
        return;
      }

      // Convert int16 PCM to float32
      const rawBuffer = event.data.slice(0);
      const view = new DataView(rawBuffer);
      const floatChunk = new Float32Array(view.byteLength / 2);
      for (let i = 0; i < floatChunk.length; i += 1) {
        floatChunk[i] = view.getInt16(i * 2, true) / 32768;
      }

      this.onAudioChunk(floatChunk, rawBuffer);
    };

    this.socket.onerror = (err) => {
      console.error('WebSocket error', err);
      this.onError(err);
    };

    this.socket.onclose = () => {
      this.socket = null;
      this.onClose();
    };
  }

  /**
   * Handle log message from server
   */
  handleLogMessage(raw) {
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch (err) {
      this.onLogMessage(`[Error] Failed to parse log message: ${raw}`);
      return;
    }

    if (!payload || payload.type !== 'log') {
      this.onLogMessage(`[Log] ${raw}`);
      return;
    }

    const { event, data = {}, timestamp } = payload;
    
    switch (event) {
      case 'backend_request_received': {
        this.onLogMessage('[Backend]  Received request', timestamp, data);
        break;
      }
      case 'backend_first_chunk_sent':
        this.onLogMessage('[Backend]  Sent first audio chunk', timestamp);
        break;
      case 'model_progress':
        if (typeof data.generated_sec !== 'undefined') {
          this.onLogMessage('[Model]    Progress update', timestamp, data);
        }
        return;
      case 'generation_error':
        this.onLogMessage(`[Error] Generation error: ${data.message || 'Unknown error'}`, timestamp);
        break;
      case 'backend_error':
        this.onLogMessage(`[Error] Backend error: ${data.message || 'Unknown error'}`, timestamp);
        break;
      case 'client_disconnected':
        this.onLogMessage('[Frontend] Client disconnected', timestamp);
        break;
      case 'backend_stream_complete':
        this.onLogMessage('[Backend]  Backend finished', timestamp);
        break;
      default:
        this.onLogMessage(`[Log] Event ${event}`, timestamp);
        break;
    }
  }

  /**
   * Close WebSocket connection
   */
  close() {
    if (this.socket && 
        (this.socket.readyState === WebSocket.OPEN || 
         this.socket.readyState === WebSocket.CONNECTING)) {
      this.socket.close();
    }
    this.socket = null;
  }

  /**
   * Get socket instance
   */
  getSocket() {
    return this.socket;
  }
}
