import React from 'react';

const TextInput = ({ value, onChange }) => {
  return (
    <section className="flex flex-col gap-2.5">
      <label className="flex flex-col gap-2">
        <span className="font-semibold text-[15px] text-[#1f2742]">Text</span>
        <textarea
          className="w-full min-h-[140px] max-h-[240px] border border-[rgba(31,39,66,0.14)] rounded-xl px-4 py-3.5 text-[15px] leading-relaxed bg-[#f9faff] transition-all duration-200 resize-y focus:outline-none focus:border-[#5562ff] focus:shadow-[0_0_0_3px_rgba(85,98,255,0.18)] focus:bg-white"
          rows="4"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </section>
  );
};

export default TextInput;
