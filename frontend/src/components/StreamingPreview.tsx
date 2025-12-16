import { useEffect, useState, useRef } from 'react';
import { Radio } from 'lucide-react';

interface StreamingPreviewProps {
  text: string;
  isStreaming: boolean;
  wordsPerMinute?: number;
}

export function StreamingPreview({ text, isStreaming, wordsPerMinute = 180 }: StreamingPreviewProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const tokensRef = useRef<string[]>([]);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (isStreaming && text) {
      tokensRef.current = text.match(/\S+\s*/g) || [];
      setCurrentIndex(0);
      setDisplayedText('');

      const intervalMs = 60000 / wordsPerMinute;
      intervalRef.current = window.setInterval(() => {
        setCurrentIndex((prev) => {
          if (prev < tokensRef.current.length) {
            setDisplayedText((current) => current + tokensRef.current[prev]);
            return prev + 1;
          }
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
          }
          return prev;
        });
      }, intervalMs);
    } else if (!isStreaming) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [text, isStreaming, wordsPerMinute]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Radio className={`w-4 h-4 ${isStreaming ? 'text-green-500 animate-pulse' : 'text-surface-400'}`} />
        <span className="text-sm font-medium text-surface-600 dark:text-surface-400">
          Streaming Input Text
        </span>
        {isStreaming && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
            Live
          </span>
        )}
      </div>
      <div className="relative bg-gradient-to-br from-surface-100 to-surface-50 dark:from-surface-800 dark:to-surface-900 
                      rounded-xl border border-surface-200 dark:border-surface-700 p-4 min-h-[80px]">
        <p className={`font-mono text-sm leading-relaxed text-surface-700 dark:text-surface-300 whitespace-pre-wrap ${
          isStreaming && currentIndex < tokensRef.current.length ? 'streaming-cursor' : ''
        }`}>
          {isStreaming ? displayedText : (text || 'Text will appear here during synthesis...')}
        </p>
      </div>
    </div>
  );
}

