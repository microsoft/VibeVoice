import { useId } from 'react';
import { Mic } from 'lucide-react';

interface VoiceSelectorProps {
  voices: string[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  loading?: boolean;
}

export function VoiceSelector({ voices = [], value, onChange, disabled, loading }: VoiceSelectorProps) {
  const id = useId();
  const safeVoices = Array.isArray(voices) ? voices : [];

  const getVoiceLabel = (voice: string) => {
    const parts = voice.split('-');
    if (parts.length >= 2) {
      const lang = parts[0].toUpperCase();
      const name = parts.slice(1).join('-').replace('_', ' ');
      return `${name} (${lang})`;
    }
    return voice;
  };

  const groupedVoices = safeVoices.reduce((acc, voice) => {
    const lang = voice.split('-')[0].toUpperCase();
    if (!acc[lang]) acc[lang] = [];
    acc[lang].push(voice);
    return acc;
  }, {} as Record<string, string[]>);

  return (
    <div className="space-y-2">
      <label htmlFor={id} className="label flex items-center gap-2">
        <Mic className="w-4 h-4" />
        Speaker
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || loading}
        className="select-field"
      >
        {loading ? (
          <option value="">Loading voices...</option>
        ) : safeVoices.length === 0 ? (
          <option value="">No voices available</option>
        ) : (
          Object.entries(groupedVoices).map(([lang, langVoices]) => (
            <optgroup key={lang} label={lang}>
              {langVoices.map((voice) => (
                <option key={voice} value={voice}>
                  {getVoiceLabel(voice)}
                </option>
              ))}
            </optgroup>
          ))
        )}
      </select>
    </div>
  );
}

