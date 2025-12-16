import { Clock, AudioLines } from 'lucide-react';
import type { AudioMetrics } from '@/types';

interface MetricsDisplayProps {
  metrics: AudioMetrics;
}

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  const formatTime = (seconds: number) => seconds.toFixed(2);

  return (
    <div className="flex flex-wrap gap-4 sm:gap-8">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30">
          <AudioLines className="w-4 h-4 text-primary-600 dark:text-primary-400" />
        </div>
        <div>
          <p className="text-xs text-surface-500 dark:text-surface-400">Model Generated</p>
          <p className="font-mono font-semibold text-surface-900 dark:text-surface-100">
            {formatTime(metrics.modelGenerated)}
            <span className="text-surface-400 dark:text-surface-500 text-sm ml-1">s</span>
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
          <Clock className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
        </div>
        <div>
          <p className="text-xs text-surface-500 dark:text-surface-400">Audio Played</p>
          <p className="font-mono font-semibold text-surface-900 dark:text-surface-100">
            {formatTime(metrics.playbackElapsed)}
            <span className="text-surface-400 dark:text-surface-500 text-sm ml-1">s</span>
          </p>
        </div>
      </div>
    </div>
  );
}

