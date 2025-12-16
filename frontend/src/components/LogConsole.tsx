import { useRef, useEffect } from 'react';
import { Terminal, Trash2 } from 'lucide-react';
import type { LogEntry } from '@/types';

interface LogConsoleProps {
  logs: LogEntry[];
  onClear: () => void;
}

export function LogConsole({ logs, onClear }: LogConsoleProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const getLogColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'success':
        return 'text-emerald-600 dark:text-emerald-400';
      case 'error':
        return 'text-red-500 dark:text-red-400';
      case 'warning':
        return 'text-amber-500 dark:text-amber-400';
      default:
        return 'text-surface-600 dark:text-surface-400';
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="label flex items-center gap-2 mb-0">
          <Terminal className="w-4 h-4" />
          Runtime Logs
        </label>
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-surface-500 hover:text-red-500 flex items-center gap-1 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Clear
        </button>
      </div>
      <div
        ref={containerRef}
        className="h-[200px] overflow-y-auto bg-surface-900 dark:bg-black rounded-xl p-4 
                   font-mono text-xs leading-relaxed"
      >
        {logs.length === 0 ? (
          <p className="text-surface-500">Logs will appear here...</p>
        ) : (
          logs.map((log) => (
            <div key={log.id} className="animate-fade-in">
              <span className="text-surface-500">[{log.timestamp}]</span>{' '}
              <span className={getLogColor(log.type)}>{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

