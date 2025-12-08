import { SAMPLE_RATE, BUFFER_SIZE, PREBUFFER_SEC } from '../utils/audioUtils';

/**
 * AudioPlayer class manages Web Audio API for real-time audio playback
 */
export class AudioPlayer {
  constructor(onPlaybackUpdate, onLog) {
    this.audioCtx = null;
    this.scriptNode = null;
    this.buffer = new Float32Array(0);
    this.hasStartedPlayback = false;
    this.silentFrameCount = 0;
    this.playbackSamples = 0;
    this.onPlaybackUpdate = onPlaybackUpdate;
    this.onLog = onLog;
    this.playbackStartedLogged = false;
    this.socket = null;
  }

  /**
   * Initialize audio context and script processor
   */
  init(socket) {
    this.teardown();
    this.resetFlags();
    this.socket = socket;

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ 
      sampleRate: SAMPLE_RATE 
    });
    
    this.scriptNode = this.audioCtx.createScriptProcessor(BUFFER_SIZE, 0, 1);
    const minBufferSamples = Math.floor(this.audioCtx.sampleRate * PREBUFFER_SEC);

    this.scriptNode.onaudioprocess = (event) => {
      const output = event.outputBuffer.getChannelData(0);
      const needPrebuffer = !this.hasStartedPlayback;
      const socketClosed = !this.socket || 
                          this.socket.readyState === WebSocket.CLOSED || 
                          this.socket.readyState === WebSocket.CLOSING;

      if (needPrebuffer) {
        if (this.buffer.length >= minBufferSamples || socketClosed) {
          this.hasStartedPlayback = true;
          if (!this.playbackStartedLogged) {
            this.playbackStartedLogged = true;
            this.onLog('[Frontend] Browser started to play audio');
          }
        } else {
          output.fill(0);
          return;
        }
      }

      const chunk = this.pullAudio(output.length);
      output.set(chunk);

      if (this.hasStartedPlayback) {
        this.playbackSamples += output.length;
        this.onPlaybackUpdate(this.playbackSamples / SAMPLE_RATE);
      }

      if (socketClosed && this.buffer.length === 0 && chunk.every(sample => sample === 0)) {
        this.silentFrameCount += 1;
        if (this.silentFrameCount >= 4) {
          // Signal to stop playback
          if (this.onStop) {
            this.onStop();
          }
        }
      } else {
        this.silentFrameCount = 0;
      }
    };

    this.scriptNode.connect(this.audioCtx.destination);
  }

  /**
   * Set stop callback
   */
  setStopCallback(callback) {
    this.onStop = callback;
  }

  /**
   * Append audio data to buffer
   */
  appendAudio(chunk) {
    const merged = new Float32Array(this.buffer.length + chunk.length);
    merged.set(this.buffer, 0);
    merged.set(chunk, this.buffer.length);
    this.buffer = merged;
  }

  /**
   * Pull audio from buffer
   */
  pullAudio(frameCount) {
    const available = this.buffer.length;
    if (available === 0) {
      return new Float32Array(frameCount);
    }
    if (available <= frameCount) {
      const chunk = this.buffer;
      this.buffer = new Float32Array(0);
      if (chunk.length < frameCount) {
        const padded = new Float32Array(frameCount);
        padded.set(chunk, 0);
        return padded;
      }
      return chunk;
    }
    const chunk = this.buffer.subarray(0, frameCount);
    this.buffer = this.buffer.subarray(frameCount);
    return chunk;
  }

  /**
   * Reset playback flags
   */
  resetFlags(resetSamples = true) {
    this.buffer = new Float32Array(0);
    if (resetSamples) {
      this.playbackSamples = 0;
    }
    this.hasStartedPlayback = false;
    this.silentFrameCount = 0;
    this.playbackStartedLogged = false;
  }

  /**
   * Get current playback time in seconds
   */
  getPlaybackTime() {
    return this.playbackSamples / SAMPLE_RATE;
  }

  /**
   * Teardown audio context
   */
  teardown() {
    if (this.scriptNode) {
      try { 
        this.scriptNode.disconnect(); 
      } catch (err) { 
        console.warn('disconnect error', err); 
      }
      this.scriptNode.onaudioprocess = null;
    }
    if (this.audioCtx) {
      try { 
        this.audioCtx.close(); 
      } catch (err) { 
        console.warn('audioCtx.close error', err); 
      }
    }
    this.audioCtx = null;
    this.scriptNode = null;
  }

  /**
   * Reset everything
   */
  reset(resetSamples = true) {
    this.teardown();
    this.resetFlags(resetSamples);
  }
}
