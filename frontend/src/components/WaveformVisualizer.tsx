import { useRef, useEffect, useCallback } from 'react';
import { Activity } from 'lucide-react';

interface WaveformVisualizerProps {
  isActive: boolean;
  getWaveformData: () => Float32Array;
}

export function WaveformVisualizer({ isActive, getWaveformData }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const isDarkRef = useRef(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }

    const width = rect.width;
    const height = rect.height;

    isDarkRef.current = document.documentElement.classList.contains('dark');
    const bgColor = isDarkRef.current ? '#18181b' : '#fafafa';
    const waveColor = isDarkRef.current ? '#818cf8' : '#6366f1';
    const gridColor = isDarkRef.current ? '#27272a' : '#e4e4e7';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    const data = getWaveformData();
    
    if (!isActive || data.length === 0) {
      const barCount = 32;
      const barWidth = width / barCount - 2;
      const maxBarHeight = height * 0.6;

      ctx.fillStyle = isDarkRef.current ? '#3f3f46' : '#d4d4d8';
      
      for (let i = 0; i < barCount; i++) {
        const barHeight = maxBarHeight * 0.2;
        const x = i * (barWidth + 2) + 1;
        const y = (height - barHeight) / 2;
        ctx.fillRect(x, y, barWidth, barHeight);
      }
      
      if (isActive) {
        animationRef.current = requestAnimationFrame(draw);
      }
      return;
    }

    const barCount = Math.min(64, Math.floor(width / 8));
    const samplesPerBar = Math.floor(data.length / barCount);
    const barWidth = width / barCount - 2;
    const maxBarHeight = height * 0.8;

    ctx.fillStyle = waveColor;

    for (let i = 0; i < barCount; i++) {
      let sum = 0;
      const startIdx = i * samplesPerBar;
      for (let j = 0; j < samplesPerBar && startIdx + j < data.length; j++) {
        sum += Math.abs(data[startIdx + j]);
      }
      const avg = sum / samplesPerBar;
      const normalizedHeight = Math.min(avg * 2, 1);
      const barHeight = Math.max(normalizedHeight * maxBarHeight, 4);
      
      const x = i * (barWidth + 2) + 1;
      const y = (height - barHeight) / 2;

      const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
      gradient.addColorStop(0, waveColor);
      gradient.addColorStop(0.5, isDarkRef.current ? '#a5b4fc' : '#818cf8');
      gradient.addColorStop(1, waveColor);
      ctx.fillStyle = gradient;

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }

    if (isActive) {
      animationRef.current = requestAnimationFrame(draw);
    }
  }, [isActive, getWaveformData]);

  useEffect(() => {
    if (isActive) {
      draw();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      draw();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, draw]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Activity className={`w-4 h-4 ${isActive ? 'text-primary-500 animate-pulse' : 'text-surface-400'}`} />
        <span className="text-sm font-medium text-surface-600 dark:text-surface-400">
          Audio Waveform
        </span>
        {isActive && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400">
            Playing
          </span>
        )}
      </div>
      <div className="relative bg-surface-50 dark:bg-surface-900 rounded-xl border border-surface-200 dark:border-surface-700 overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-[100px]"
          style={{ display: 'block' }}
        />
      </div>
    </div>
  );
}

