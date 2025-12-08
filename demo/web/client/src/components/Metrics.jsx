import React from 'react';
import { formatSeconds } from '../utils/audioUtils';

const Metrics = ({ modelGenerated, playbackElapsed }) => {
  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex flex-wrap gap-4 gap-x-8 text-sm text-[#5d6789]">
        <span className="flex items-baseline gap-1.5">
          Model Generated Audio
          <strong className="text-[#1f2742] font-semibold">{formatSeconds(modelGenerated)}</strong>
          <span className="text-[#5d6789] text-[13px]">s</span>
        </span>
        <span className="flex items-baseline gap-1.5">
          Audio Played
          <strong className="text-[#1f2742] font-semibold">{formatSeconds(playbackElapsed)}</strong>
          <span className="text-[#5d6789] text-[13px]">s</span>
        </span>
      </div>
    </section>
  );
};

export default Metrics;
