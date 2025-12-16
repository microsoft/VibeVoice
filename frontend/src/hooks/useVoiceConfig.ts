import { useState, useEffect, useCallback } from 'react';
import { fetchVoiceConfig } from '@/services/api';

export function useVoiceConfig() {
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadVoices = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const config = await fetchVoiceConfig();
      const voiceList = Array.isArray(config.voices) ? config.voices : [];
      setVoices(voiceList);
      if (config.default_voice && voiceList.includes(config.default_voice)) {
        setSelectedVoice(config.default_voice);
      } else if (voiceList.length > 0) {
        setSelectedVoice(voiceList[0]);
      }
    } catch (err) {
      console.error('Failed to load voices:', err);
      setError('Failed to load voice presets');
      setVoices([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVoices();
  }, [loadVoices]);

  return {
    voices,
    selectedVoice,
    setSelectedVoice,
    loading,
    error,
    reload: loadVoices,
  };
}

