import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const SAMPLE_RATE = 24000;

const App = () => {
  const [text, setText] = useState('Enter your text here and click "Start" to instantly hear the VibeVoice-Realtime TTS output audio.');
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [cfgScale, setCfgScale] = useState(1.5);
  const [inferenceSteps, setInferenceSteps] = useState(5);
  const [isPlaying, setIsPlaying] = useState(false);
  const [modelGenerated, setModelGenerated] = useState(0);
  const [playbackElapsed, setPlaybackElapsed] = useState(0);
  const [logs, setLogs] = useState('');
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [canSave, setCanSave] = useState(false);

  const audioCtxRef = useRef(null);
  const socketRef = useRef(null);
  const audioBufferRef = useRef(new Float32Array(0));

  // Load voices
  useEffect(() => {
    axios.get('/config')
      .then(res => {
        const voiceList = res.data.voices || [];
        setVoices(voiceList);
        setSelectedVoice(res.data.default_voice || voiceList[0] || '');
        addLog(`Loaded ${voiceList.length} voices`);
      })
      .catch(() => addLog('Error loading voices'));
  }, []);

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => prev + `[${timestamp}] ${message}\n`);
  };

  const start = () => {
    if (isPlaying) return;
    
    setIsPlaying(true);
    setModelGenerated(0);
    setPlaybackElapsed(0);
    setRecordedChunks([]);
    setCanSave(false);
    setLogs('');
    addLog(`Starting: CFG=${cfgScale.toFixed(2)}, Steps=${inferenceSteps}, Voice=${selectedVoice}`);

    // Setup audio context
    audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    audioBufferRef.current = new Float32Array(0);
    
    // Setup WebSocket
    const params = new URLSearchParams({
      text,
      cfg: cfgScale.toFixed(3),
      steps: inferenceSteps.toString(),
      voice: selectedVoice
    });
    
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/stream?${params}`;
    socketRef.current = new WebSocket(wsUrl);
    socketRef.current.binaryType = 'arraybuffer';

    socketRef.current.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        if (msg.event === 'model_progress' && msg.data.generated_sec) {
          setModelGenerated(msg.data.generated_sec);
        }
        addLog(msg.event || event.data);
      } else {
        // Audio data
        const view = new DataView(event.data);
        const floatChunk = new Float32Array(view.byteLength / 2);
        for (let i = 0; i < floatChunk.length; i++) {
          floatChunk[i] = view.getInt16(i * 2, true) / 32768;
        }
        
        // Append to buffer
        const newBuffer = new Float32Array(audioBufferRef.current.length + floatChunk.length);
        newBuffer.set(audioBufferRef.current);
        newBuffer.set(floatChunk, audioBufferRef.current.length);
        audioBufferRef.current = newBuffer;
        
        // Save for WAV export
        setRecordedChunks(prev => [...prev, event.data]);
        
        // Play audio
        playAudio(floatChunk);
      }
    };

    socketRef.current.onclose = () => {
      setCanSave(true);
      addLog('Streaming complete');
    };

    socketRef.current.onerror = () => {
      addLog('WebSocket error');
      stop();
    };
  };

  const playAudio = (audioData) => {
    if (!audioCtxRef.current) return;
    
    const buffer = audioCtxRef.current.createBuffer(1, audioData.length, SAMPLE_RATE);
    buffer.getChannelData(0).set(audioData);
    
    const source = audioCtxRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtxRef.current.destination);
    source.start();
    
    setPlaybackElapsed(prev => prev + audioData.length / SAMPLE_RATE);
  };

  const stop = () => {
    setIsPlaying(false);
    
    if (socketRef.current) {
      socketRef.current.close();
    }
    
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    
    addLog('Stopped');
  };

  const handleSave = () => {
    if (recordedChunks.length === 0) return;
    
    const totalSamples = recordedChunks.reduce((sum, chunk) => sum + chunk.byteLength / 2, 0);
    const wavBuffer = new ArrayBuffer(44 + totalSamples * 2);
    const view = new DataView(wavBuffer);
    
    // WAV header
    const writeString = (offset, str) => {
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
    
    // Copy PCM data
    const pcmData = new Int16Array(wavBuffer, 44);
    let offset = 0;
    recordedChunks.forEach(chunk => {
      pcmData.set(new Int16Array(chunk), offset);
      offset += chunk.byteLength / 2;
    });
    
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vibevoice_${Date.now()}.wav`;
    link.click();
    URL.revokeObjectURL(url);
    
    addLog('Audio saved');
  };

  return (
    <div className="min-h-screen bg-[#f5f7fc] flex justify-center p-12 px-5">
      <div className="w-full max-w-[960px] bg-white rounded-[20px] p-9 pb-11 shadow-[0_18px_45px_rgba(31,39,66,0.08)] flex flex-col gap-7">
        <h1 className="m-0 text-center text-[30px] font-bold tracking-[0.01em] text-[#1f2742]">
          VibeVoice-Realtime TTS Demo
        </h1>

        {/* Text Input */}
        <section className="flex flex-col gap-2.5">
          <label className="flex flex-col gap-2">
            <span className="font-semibold text-[15px] text-[#1f2742]">Text</span>
            <textarea
              className="w-full min-h-[140px] max-h-60 border border-[rgba(31,39,66,0.14)] rounded-xl px-4 py-3.5 text-[15px] leading-relaxed bg-[#f9faff] transition-all duration-200 resize-y focus:outline-none focus:border-[#5562ff] focus:shadow-[0_0_0_3px_rgba(85,98,255,0.18)] focus:bg-white"
              rows="4"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
        </section>

        <span className="text-xs text-[#8a93b5]">
          This demo requires the full text to be provided upfront. The model then receives the text via streaming input during synthesis.
        </span>

        {/* Controls */}
        <section className="flex flex-col gap-4.5">
          <div className="flex flex-col gap-1.5">
            <span className="font-semibold text-[15px] text-[#1f2742]">Speaker</span>
            <select 
              className="w-[220px] border border-[rgba(31,39,66,0.14)] rounded-[10px] px-3 py-2 text-sm bg-[#fbfcff] text-[#1f2742] transition-all duration-200 focus:outline-none focus:border-[#5562ff] focus:shadow-[0_0_0_3px_rgba(85,98,255,0.18)] focus:bg-white"
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
            >
              {voices.map(voice => (
                <option key={voice} value={voice}>{voice}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center flex-wrap gap-5 gap-x-7">
            <label className="flex items-center gap-3 text-sm text-[#1f2742]">
              <span>CFG</span>
              <input
                type="range"
                min="1.3"
                max="3"
                step="0.05"
                value={cfgScale}
                onChange={(e) => setCfgScale(Number(e.target.value))}
                className="w-[200px] accent-[#5562ff]"
              />
              <span className="font-semibold min-w-[42px] text-right">{cfgScale.toFixed(2)}</span>
            </label>
            
            <label className="flex items-center gap-3 text-sm text-[#1f2742]">
              <span>Steps</span>
              <input
                type="range"
                min="5"
                max="20"
                step="1"
                value={inferenceSteps}
                onChange={(e) => setInferenceSteps(Number(e.target.value))}
                className="w-[200px] accent-[#5562ff]"
              />
              <span className="font-semibold min-w-[42px] text-right">{inferenceSteps}</span>
            </label>
            
            <button 
              onClick={() => { setCfgScale(1.5); setInferenceSteps(5); }}
              className="border border-[rgba(31,39,66,0.18)] bg-[#f1f3ff] text-[#1f2742] px-4 py-2 rounded-full text-[13px] font-medium hover:bg-[#e6e9ff]"
            >
              Reset
            </button>
          </div>

          <div className="flex items-center flex-wrap gap-5">
            <button 
              onClick={isPlaying ? stop : start}
              className={`px-6 py-2.5 rounded-full font-semibold text-sm text-white shadow-lg transition-all ${
                isPlaying ? 'bg-[#3f4dff]' : 'bg-[#5562ff] hover:-translate-y-px'
              }`}
            >
              {isPlaying ? 'Stop' : 'Start'}
            </button>
            
            <button 
              onClick={handleSave}
              disabled={!canSave}
              className="border border-[rgba(31,39,66,0.18)] bg-[#f1f3ff] text-[#1f2742] px-4 py-2 rounded-full text-[13px] font-medium hover:bg-[#e6e9ff] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
          </div>
        </section>

        {/* Metrics */}
        <section className="flex flex-wrap gap-4 gap-x-8 text-sm text-[#5d6789]">
          <span className="flex items-baseline gap-1.5">
            Model Generated
            <strong className="text-[#1f2742] font-semibold">{modelGenerated.toFixed(2)}</strong>
            <span className="text-[13px]">s</span>
          </span>
          <span className="flex items-baseline gap-1.5">
            Audio Played
            <strong className="text-[#1f2742] font-semibold">{playbackElapsed.toFixed(2)}</strong>
            <span className="text-[13px]">s</span>
          </span>
        </section>

        {/* Logs */}
        <section className="flex flex-col gap-2.5">
          <span className="font-semibold text-[15px] text-[#1f2742]">Runtime Logs</span>
          <pre className="max-h-[260px] overflow-y-auto bg-[#f7f9ff] text-[#1f2742] p-4 border border-[rgba(31,39,66,0.12)] rounded-xl text-[13px] leading-relaxed font-mono whitespace-pre-wrap">
            {logs}
          </pre>
        </section>
      </div>
    </div>
  );
};

export default App;