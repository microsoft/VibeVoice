import React, { useEffect, useRef, useState } from 'react';

const STREAMING_WPM = 180;
const STREAMING_INTERVAL_MS = 60000 / STREAMING_WPM;

const StreamingPreview = ({ text, isActive }) => {
  const [displayText, setDisplayText] = useState('This area will display the streaming input text in real time.');
  const [tokens, setTokens] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    if (isActive && text) {
      // Tokenize text into words
      const wordTokens = text.trimEnd().match(/\S+\s*/g) || [];
      setTokens(wordTokens);
      setCurrentIndex(0);
      setDisplayText('');
    } else if (!isActive) {
      setDisplayText('This area will display the streaming input text in real time.');
      setTokens([]);
      setCurrentIndex(0);
    }
  }, [text, isActive]);

  useEffect(() => {
    if (!isActive || currentIndex >= tokens.length) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      return;
    }

    timerRef.current = setTimeout(() => {
      setDisplayText(prev => prev + tokens[currentIndex]);
      setCurrentIndex(prev => prev + 1);
    }, STREAMING_INTERVAL_MS);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [currentIndex, tokens, isActive]);

  const isStreaming = isActive && currentIndex < tokens.length;

  return (
    <div className="rounded-[14px] border border-[rgba(85,98,255,0.18)] bg-gradient-to-br from-[#eef2ff] to-[#f7f9ff] p-5 shadow-[inset_0_1px_2px_rgba(85,98,255,0.12)]">
      <div className="font-semibold text-[#1f2742] flex items-center gap-2.5 text-sm mb-2">
        <span>Streaming Input Text</span>
      </div>
      <div 
        className={`min-h-[70px] px-3 py-2.5 rounded-[10px] bg-[rgba(255,255,255,0.9)] border border-[rgba(85,98,255,0.25)] font-mono text-sm leading-relaxed text-[#1f2742] whitespace-pre-wrap ${isStreaming ? 'after:inline-block after:w-0.5 after:h-[1.1em] after:bg-[#5562ff] after:ml-0.5 after:align-bottom after:animate-[previewCaret_0.9s_steps(1)_infinite]' : ''}`}
        aria-live="polite"
      >
        {displayText}
      </div>
    </div>
  );
};

export default StreamingPreview;
