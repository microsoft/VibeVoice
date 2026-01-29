'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, SkipForward, SkipBack, Volume2, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

interface AudioPlayerProps {
  audioUrl: string | null;
  title?: string;
  duration?: number;
  onDownload?: () => Promise<void> | void;
}

export function AudioPlayer({ audioUrl, title, duration, onDownload }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [playbackRate, setPlaybackRate] = useState(1);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    // Reset player state when audio changes
    setCurrentTime(0);
    setIsPlaying(false);
    try {
      audio.pause();
      audio.currentTime = 0;
    } catch (e) {
      // ignore if audio element not ready
    }

    const handleTimeUpdate = () => {
      if (audio) {
        setCurrentTime(audio.currentTime);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  const togglePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
      return;
    }

    try {
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        await playPromise;
      }
      setIsPlaying(true);
    } catch (err) {
      setIsPlaying(false);
      console.error('Failed to play audio:', err);
      toast.error('Failed to start playback');
    }
  };

  const handleSeek = (time: number) => {
    const audio = audioRef.current;
    if (audio) {
      const target = typeof duration === 'number' ? Math.min(duration, time) : time;
      audio.currentTime = target;
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const audio = audioRef.current;
    if (audio) {
      audio.volume = value[0];
      setVolume(value[0]);
    }
  };

  const handlePlaybackRateChange = (rate: number) => {
    const audio = audioRef.current;
    if (audio) {
      audio.playbackRate = rate;
    }
    setPlaybackRate(rate);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Sync volume and playbackRate to the audio element when a new source is loaded
  // (per-control handlers already update the element on user interaction)
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.volume = volume;
      audio.playbackRate = playbackRate;
    }
  }, [audioUrl, volume, playbackRate]);

  const handleDownload = async () => {
    if (!onDownload) {
      // No handler provided; treat as noop
      return;
    }

    try {
      await onDownload();
      toast.success('Audio downloaded');
    } catch (err) {
      console.error('Download failed', err);
      toast.error('Download failed');
      // Do not rethrow to avoid unhandled promise rejections from onClick handlers
      return;
    }
  };

  if (!audioUrl) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Audio Player</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <p className="text-sm">No audio loaded</p>
            <p className="text-xs mt-2">Generate audio to use the player</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            {title || 'Audio Player'}
          </div>
          {duration && (
            <Badge variant="secondary">
              {formatTime(duration)}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <audio
          ref={audioRef}
          src={audioUrl ?? undefined}
          preload="metadata"
          style={{ display: 'none' }}
        />
        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{formatTime(currentTime)}</span>
            <span>{duration ? formatTime(duration) : '0:00'}</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center gap-4">
          {/* Skip Back (10s) */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSeek(Math.max(0, currentTime - 10))}
            disabled={currentTime <= 0}
            aria-label="Rewind 10 seconds"
            title="Rewind 10 seconds"
          >
            <div className="flex items-center gap-1 px-2">
              <SkipBack className="h-4 w-4" />
              <span className="text-xs">-10s</span>
            </div>
          </Button>

          {/* Rewind (5s) */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSeek(Math.max(0, currentTime - 5))}
            disabled={currentTime <= 0}
            aria-label="Rewind 5 seconds"
            title="Rewind 5 seconds"
          >
            <div className="flex items-center gap-1 px-2">
              <SkipBack className="h-4 w-4" />
              <span className="text-xs">-5s</span>
            </div>
          </Button>

          {/* Play/Pause */}
          <Button size="icon" onClick={togglePlayPause} aria-label={isPlaying ? 'Pause' : 'Play'} title={isPlaying ? 'Pause' : 'Play'}>
            {isPlaying ? (
              <Pause className="h-6 w-6" />
            ) : (
              <Play className="h-6 w-6" />
            )}
          </Button>

          {/* Fast Forward (5s) */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSeek(currentTime + 5)}
            disabled={duration ? currentTime >= duration : true}
            aria-label="Fast forward 5 seconds"
            title="Fast forward 5 seconds"
          >
            <div className="flex items-center gap-1 px-2">
              <SkipForward className="h-4 w-4" />
              <span className="text-xs">+5s</span>
            </div>
          </Button> 

          {/* Skip Forward (10s) */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSeek(currentTime + 10)}
            disabled={duration ? currentTime >= duration : true}
            aria-label="Skip forward 10 seconds"
            title="Skip forward 10 seconds"
          >
            <div className="flex items-center gap-1 px-2">
              <SkipForward className="h-4 w-4" />
              <span className="text-xs">+10s</span>
            </div>
          </Button>
        </div>

        {/* Volume Control */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Volume2 className="h-4 w-4" />
            Volume
          </div>
          <Slider
            min={0}
            max={1}
            step={0.1}
            value={[volume]}
            onValueChange={handleVolumeChange}
            className="w-full"
          />
        </div>

        {/* Playback Speed */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Playback Speed</span>
            <Badge variant="outline">{playbackRate}x</Badge>
          </div>
          <div className="flex gap-2">
            {[0.5, 0.75, 1, 1.25, 1.5, 2].map((rate) => (
              <Button
                key={rate}
                variant={playbackRate === rate ? 'default' : 'outline'}
                size="sm"
                onClick={() => handlePlaybackRateChange(rate)}
              >
                {rate}x
              </Button>
            ))}
          </div>
        </div>

        {/* Download Button */}
        {onDownload && (
          <Button onClick={handleDownload} className="w-full">
            <Download className="h-4 w-4 mr-2" />
            Download Audio
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
