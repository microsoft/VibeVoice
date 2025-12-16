import { useId } from 'react';
import { Sliders, Zap, RotateCcw } from 'lucide-react';

interface ControlPanelProps {
  cfg: number;
  steps: number;
  onCfgChange: (value: number) => void;
  onStepsChange: (value: number) => void;
  onReset: () => void;
  disabled?: boolean;
}

export function ControlPanel({
  cfg,
  steps,
  onCfgChange,
  onStepsChange,
  onReset,
  disabled,
}: ControlPanelProps) {
  const cfgId = useId();
  const stepsId = useId();

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-surface-700 dark:text-surface-300 flex items-center gap-2">
          <Sliders className="w-4 h-4" />
          Generation Parameters
        </h3>
        <button
          type="button"
          onClick={onReset}
          disabled={disabled}
          className="text-xs text-surface-500 hover:text-primary-600 dark:hover:text-primary-400 
                     flex items-center gap-1 transition-colors disabled:opacity-50"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
      </div>

      <div className="grid sm:grid-cols-2 gap-6">
        {/* CFG Scale */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor={cfgId} className="text-sm font-medium text-surface-600 dark:text-surface-400">
              CFG Scale
            </label>
            <span className="text-sm font-mono font-semibold text-primary-600 dark:text-primary-400 
                           bg-primary-50 dark:bg-primary-950 px-2 py-0.5 rounded-md">
              {cfg.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            id={cfgId}
            min="1.3"
            max="3"
            step="0.05"
            value={cfg}
            onChange={(e) => onCfgChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-surface-400">
            <span>1.3</span>
            <span>3.0</span>
          </div>
        </div>

        {/* Inference Steps */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor={stepsId} className="text-sm font-medium text-surface-600 dark:text-surface-400 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5" />
              Inference Steps
            </label>
            <span className="text-sm font-mono font-semibold text-primary-600 dark:text-primary-400 
                           bg-primary-50 dark:bg-primary-950 px-2 py-0.5 rounded-md">
              {steps}
            </span>
          </div>
          <input
            type="range"
            id={stepsId}
            min="5"
            max="20"
            step="1"
            value={steps}
            onChange={(e) => onStepsChange(parseInt(e.target.value))}
            disabled={disabled}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-surface-400">
            <span>5 (Fast)</span>
            <span>20 (Quality)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

