// Audio utilities for WAV encoding and audio processing

export const SAMPLE_RATE = 24_000;
export const BUFFER_SIZE = 2048;
export const PREBUFFER_SEC = 0.1;

/**
 * Create a WAV file blob from recorded PCM chunks
 */
export const createWavBlob = (recordedChunks, recordedSamples) => {
  if (!recordedSamples) {
    return null;
  }

  const wavBuffer = new ArrayBuffer(44 + recordedSamples * 2);
  const view = new DataView(wavBuffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // WAV header
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + recordedSamples * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, recordedSamples * 2, true);

  // Copy PCM data
  const pcmData = new Int16Array(wavBuffer, 44, recordedSamples);
  let offset = 0;
  recordedChunks.forEach(chunk => {
    const chunkData = new Int16Array(chunk);
    pcmData.set(chunkData, offset);
    offset += chunkData.length;
  });

  return new Blob([wavBuffer], { type: 'audio/wav' });
};

/**
 * Format seconds to 2 decimal places
 */
export const formatSeconds = (raw) => {
  const value = Number(raw);
  return Number.isFinite(value) ? value.toFixed(2) : '0.00';
};
