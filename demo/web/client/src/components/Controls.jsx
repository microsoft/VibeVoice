import React, { useEffect } from 'react';

const Controls = ({
  voices,
  selectedVoice,
  onVoiceChange,
  cfgScale,
  onCfgChange,
  inferenceSteps,
  onStepsChange,
  onReset,
  isPlaying,
  onPlayPause,
  onSave,
  canSave
}) => {
  const updateCfgDisplay = () => Number(cfgScale).toFixed(2);
  const updateStepsDisplay = () => Number(inferenceSteps).toString();

  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-[18px]">
        <div className="flex flex-col gap-1.5">
          <span className="font-semibold text-[15px] text-[#1f2742]">Speaker</span>
          <select 
            className="w-[220px] border border-[rgba(31,39,66,0.14)] rounded-[10px] px-3 py-2 text-sm bg-[#fbfcff] text-[#1f2742] transition-all duration-200 focus:outline-none focus:border-[#5562ff] focus:shadow-[0_0_0_3px_rgba(85,98,255,0.18)] focus:bg-white"
            value={selectedVoice}
            onChange={(e) => onVoiceChange(e.target.value)}
            disabled={voices.length === 0}
          >
            {voices.length === 0 ? (
              <option value="">Loading...</option>
            ) : (
              voices.map(voice => (
                <option key={voice} value={voice}>{voice}</option>
              ))
            )}
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
              onChange={(e) => onCfgChange(Number(e.target.value))}
              className="w-[200px] accent-[#5562ff]"
            />
            <span className="font-semibold text-[#1f2742] min-w-[42px] text-right">{updateCfgDisplay()}</span>
          </label>
          <label className="flex items-center gap-3 text-sm text-[#1f2742]">
            <span>Inference Steps</span>
            <input
              type="range"
              min="5"
              max="20"
              step="1"
              value={inferenceSteps}
              onChange={(e) => onStepsChange(Number(e.target.value))}
              className="w-[200px] accent-[#5562ff]"
            />
            <span className="font-semibold text-[#1f2742] min-w-[42px] text-right">{updateStepsDisplay()}</span>
          </label>
          <button 
            type="button" 
            className="border border-[rgba(31,39,66,0.18)] bg-[#f1f3ff] text-[#1f2742] px-[18px] py-2 rounded-full cursor-pointer text-[13px] font-medium transition-all duration-150 hover:bg-[#e6e9ff] hover:border-[rgba(31,39,66,0.26)]"
            onClick={onReset}
          >
            Reset Controls
          </button>
        </div>

        <div className="flex items-center flex-wrap gap-5 gap-x-7">
          <button 
            className={`bg-[#5562ff] text-white border-none px-6 py-2.5 rounded-full cursor-pointer font-semibold text-sm shadow-[0_8px_16px_rgba(85,98,255,0.25)] transition-all duration-150 hover:-translate-y-px hover:shadow-[0_10px_20px_rgba(85,98,255,0.28)] active:translate-y-0 ${isPlaying ? 'bg-[#3f4dff]' : ''}`}
            onClick={onPlayPause}
          >
            {isPlaying ? 'Stop' : 'Start'}
          </button>
          <button 
            type="button" 
            className="border border-[rgba(31,39,66,0.18)] bg-[#f1f3ff] text-[#1f2742] px-[18px] py-2 rounded-full cursor-pointer text-[13px] font-medium transition-all duration-150 hover:bg-[#e6e9ff] hover:border-[rgba(31,39,66,0.26)] disabled:opacity-55 disabled:cursor-not-allowed"
            onClick={onSave}
            disabled={!canSave}
          >
            Save
          </button>
        </div>
      </div>
    </section>
  );
};

export default Controls;
