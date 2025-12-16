import { useId } from 'react';
import { Type } from 'lucide-react';

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  maxLength?: number;
}

export function TextInput({ value, onChange, disabled, maxLength = 5000 }: TextInputProps) {
  const id = useId();
  const charCount = value.length;
  const isNearLimit = charCount > maxLength * 0.9;

  return (
    <div className="space-y-2">
      <label htmlFor={id} className="label flex items-center gap-2">
        <Type className="w-4 h-4" />
        Text Input
      </label>
      <div className="relative">
        <textarea
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          maxLength={maxLength}
          rows={5}
          placeholder="Enter the text you want to convert to speech..."
          className="input-field min-h-[140px] max-h-[300px] resize-y font-normal leading-relaxed"
        />
        <div className={`absolute bottom-3 right-3 text-xs font-mono transition-colors ${
          isNearLimit ? 'text-amber-500' : 'text-surface-400 dark:text-surface-500'
        }`}>
          {charCount.toLocaleString()} / {maxLength.toLocaleString()}
        </div>
      </div>
      <p className="text-xs text-surface-500 dark:text-surface-400">
        The model receives text via streaming input during synthesis. For best results, apply text normalization for special characters.
      </p>
    </div>
  );
}

