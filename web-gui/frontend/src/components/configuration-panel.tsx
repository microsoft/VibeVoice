'use client';

import { useCallback, useEffect } from 'react';
import { Settings, SlidersHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';

export interface Configuration {
  chunkDepth: number;
  pauseMs: number;
  includeHeading: boolean;
  stripMarkdown: boolean;
  device: 'auto' | 'cuda' | 'cpu' | 'mps';
  iterations: number;
}

/**
 * A shared default configuration used by the UI and exported for consumers.
 * Keep this in sync with any server-side defaults where applicable.
 */
export const DEFAULT_CONFIGURATION: Configuration = {
  chunkDepth: 1,
  pauseMs: 500,
  includeHeading: false,
  stripMarkdown: true,
  device: 'auto',
  iterations: 1,
};

interface ConfigurationPanelProps {
  configuration: Configuration;
  onConfigurationChange: (config: Configuration) => void;
  maxIterations?: number; // server-provided max per request (dynamic)
}

export function ConfigurationPanel({ configuration, onConfigurationChange, maxIterations }: ConfigurationPanelProps) {
  // Select components provide string values; parse and forward to the common updater
  const handleChunkDepthSelectChange = useCallback((v: string) => {
    const parsed = parseInt(v, 10);
    if (!Number.isNaN(parsed)) {
      onConfigurationChange({ ...configuration, chunkDepth: parsed });
    }
  }, [configuration, onConfigurationChange]);

  const handlePauseMsChange = useCallback((value: number[]) => {
    onConfigurationChange({ ...configuration, pauseMs: value[0] });
  }, [configuration, onConfigurationChange]);

  const handleIncludeHeadingChange = useCallback((checked: boolean) => {
    onConfigurationChange({ ...configuration, includeHeading: checked });
  }, [configuration, onConfigurationChange]);

  const handleStripMarkdownChange = useCallback((checked: boolean) => {
    onConfigurationChange({ ...configuration, stripMarkdown: checked });
  }, [configuration, onConfigurationChange]);

  const handleDeviceChange = useCallback((device: string) => {
    onConfigurationChange({ ...configuration, device: device as Configuration['device'] });
    toast.info(`Device changed to ${device}`);
  }, [configuration, onConfigurationChange]);

  const handleIterationsChange = useCallback((value: number[]) => {
    onConfigurationChange({ ...configuration, iterations: value[0] });
  }, [configuration, onConfigurationChange]);

  // Ensure `configuration.iterations` is always within [1, maxIterations]
  useEffect(() => {
    const max = typeof maxIterations === 'number' && maxIterations > 0 ? maxIterations : 5;
    const clamped = Math.max(1, Math.min(configuration.iterations, max));
    if (configuration.iterations !== clamped) {
      onConfigurationChange({ ...configuration, iterations: clamped });
    }
  }, [maxIterations, configuration.iterations, onConfigurationChange, configuration]);

  const handleReset = useCallback(() => {
    onConfigurationChange({ ...DEFAULT_CONFIGURATION });
    toast.success('Configuration reset to defaults');
  }, [onConfigurationChange]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Chunk Depth */}
        <div className="space-y-2">
          <Label htmlFor="chunk-depth">
            Chunk Depth (Heading Level)
          </Label>
          <Select value={configuration.chunkDepth.toString()} onValueChange={handleChunkDepthSelectChange}>
            <SelectTrigger id="chunk-depth">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1">H1 (#)</SelectItem>
              <SelectItem value="2">H2 (##)</SelectItem>
              <SelectItem value="3">H3 (###)</SelectItem>
              <SelectItem value="4">H4 (####)</SelectItem>
              <SelectItem value="5">H5 (#####)</SelectItem>
              <SelectItem value="6">H6 (######)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Splits document at selected heading level
          </p>
        </div>

        {/* Pause Duration */}
        <div className="space-y-2">
          <Label htmlFor="pause-duration">
            Pause Between Chunks: {configuration.pauseMs}ms
          </Label>
          <Slider
            id="pause-duration"
            min={0}
            max={5000}
            step={100}
            value={[configuration.pauseMs]}
            onValueChange={handlePauseMsChange}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0ms</span>
            <span>5000ms</span>
          </div>
        </div>

        {/* Include Heading */}
        <div className="flex flex-col space-y-1">
          <div className="flex items-center justify-between">
            <Label htmlFor="include-heading" className="flex-1">
              Include Headings in Speech
            </Label>
            <Switch
              id="include-heading"
              checked={configuration.includeHeading}
              onCheckedChange={handleIncludeHeadingChange}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Speak headings in each chunk
          </p>
        </div>

        {/* Strip Markdown */}
        <div className="flex flex-col space-y-1">
          <div className="flex items-center justify-between">
            <Label htmlFor="strip-markdown" className="flex-1">
              Strip Markdown Formatting
            </Label>
            <Switch
              id="strip-markdown"
              checked={configuration.stripMarkdown}
              onCheckedChange={handleStripMarkdownChange}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Remove markdown syntax from text
          </p>
        </div>

        {/* Device Selection */}
        <div className="space-y-2">
          <Label htmlFor="device">
            Processing Device
          </Label>
          <Select value={configuration.device} onValueChange={handleDeviceChange}>
            <SelectTrigger id="device">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (Recommended)</SelectItem>
              <SelectItem value="cuda">CUDA (GPU)</SelectItem>
              <SelectItem value="cpu">CPU</SelectItem>
              <SelectItem value="mps">MPS (Apple Silicon)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Auto-detects best available option
          </p>
        </div>

        {/* Iterations */}
        <div className="space-y-2">
          <Label htmlFor="iterations">
            Iterations per Conversion: {configuration.iterations} (max {maxIterations ?? 5})
          </Label>
          <Slider
            id="iterations"
            min={1}
            max={maxIterations ?? 5}
            step={1}
            value={[configuration.iterations]}
            onValueChange={handleIterationsChange}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>1</span>
            <span>{maxIterations ?? 5}</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Generate multiple variants in one run
          </p>
        </div>

        {/* Reset Button */}
        <div className="pt-4 border-t">
          <Button variant="outline" className="w-full" onClick={handleReset}>
            <SlidersHorizontal className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
