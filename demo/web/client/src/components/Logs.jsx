import React, { useEffect, useRef } from 'react';

const Logs = ({ logs }) => {
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <section className="flex flex-col gap-2.5">
      <span className="font-semibold text-[15px] text-[#1f2742]">Runtime Logs</span>
      <pre className="max-h-[260px] overflow-y-auto bg-[#f7f9ff] text-[#1f2742] p-4 px-[18px] border border-[rgba(31,39,66,0.12)] rounded-xl text-[13px] leading-relaxed shadow-[inset_0_1px_2px_rgba(15,23,42,0.06)] font-mono mt-0 whitespace-pre-wrap" ref={logRef}>
        {logs}
      </pre>
    </section>
  );
};

export default Logs;
