'use client';

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { Play, Check, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

// ============================================================================
// CONSTANTS & TYPES
// ============================================================================

/**
 * Language display mapping - moved outside component to prevent recreation on each render
 */
const LANGUAGE_MAP: Readonly<Record<string, string>> = {
  'english': 'English',
  'german': 'German',
  'french': 'French',
  'hindi': 'Hindi',
  'italian': 'Italian',
  'japanese': 'Japanese',
  'korean': 'Korean',
  'dutch': 'Dutch',
  'polish': 'Polish',
  'portuguese': 'Portuguese',
  'spanish': 'Spanish',
} as const;

/**
 * Gender display mapping
 */
const GENDER_MAP: Readonly<Record<'male' | 'female', string>> = {
  'male': 'Male',
  'female': 'Female',
} as const;

/**
 * Preview timeout in milliseconds - prevents stuck loading state
 */
const PREVIEW_TIMEOUT_MS = 10000;

/**
 * Voice data structure
 */
export interface Voice {
  id: string;
  name: string;
  language: string;
  gender?: 'male' | 'female';
  /**
   * Path or resource identifier for the voice file (server-side path or URL).
   *
   * This value is not used directly by the visual VoiceSelection component,
   * but is forwarded to preview handlers (`onVoicePreview` / `onVoiceManualPlay`)
   * so consumers can locate or request the corresponding voice audio resource.
   */
  path?: string;
}

/**
 * Component props
 */
export interface VoiceSelectionProps {
  voices: Voice[];
  selectedVoice: string | null;
  onVoiceSelect: (voiceId: string) => void;
  onVoicePreview?: (voice: Voice) => Promise<boolean | void> | boolean | void; // return true if autoplay was blocked
  onVoiceManualPlay?: (voice: Voice) => Promise<void> | void;
  /** When true, show a success toast after selecting a voice. Defaults to true. */
  showSelectionToast?: boolean;
}

/**
 * Grouped voices structure
 */
interface GroupedVoices {
  [language: string]: Voice[];
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Formats language name for display
 * @param language - Raw language string
 * @returns Formatted language name
 */
export const formatLanguage = (language: string): string => {
  const normalized = language.toLowerCase().trim();
  return LANGUAGE_MAP[normalized] || capitalizeFirstLetter(language);
};

/**
 * Formats gender for display
 * @param gender - Gender value
 * @returns Formatted gender string or null
 */
export const formatGender = (gender?: 'male' | 'female'): string | null => {
  if (!gender) return null;
  return GENDER_MAP[gender] ?? null;
};

/**
 * Capitalizes first letter of a string
 * @param str - Input string
 * @returns Capitalized string
 */
const capitalizeFirstLetter = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

/**
 * Groups voices by language
 * @param voices - Array of voices
 * @returns Object with language keys and voice arrays as values
 */
const groupVoicesByLanguage = (voices: Voice[]): GroupedVoices => {
  return voices.reduce<GroupedVoices>((acc, voice) => {
    const lang = voice.language.toLowerCase().trim();
    if (!acc[lang]) {
      acc[lang] = [];
    }
    acc[lang].push(voice);
    return acc;
  }, {});
};

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

interface VoiceCardProps {
  voice: Voice;
  isSelected: boolean;
  isPreviewing: boolean;
  hasError: boolean;
  isBlocked?: boolean;
  onSelect: (voiceId: string) => void;
  onPreview: (voice: Voice) => void;
  onManualPlay?: (voice: Voice) => void;
}

/**
 * Individual voice card component for better separation of concerns
 */
const VoiceCard = ({ voice, isSelected, isPreviewing, hasError, isBlocked, onSelect, onPreview, onManualPlay }: VoiceCardProps) => {
  const genderLabel = formatGender(voice.gender);
  const languageLabel = formatLanguage(voice.language);

  return (
    <div
      className={`relative rounded-lg border-2 p-4 transition-all cursor-pointer hover:shadow-md ${
        isSelected
          ? 'border-primary bg-primary/5'
          : hasError
          ? 'border-destructive bg-destructive/5'
          : 'border-border hover:border-primary/50'
      }`}
      onClick={() => onSelect(voice.id)}
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
      aria-label={`Select voice ${voice.name}, ${languageLabel}${genderLabel ? `, ${genderLabel}` : ''}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(voice.id);
        }
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-foreground">
              {voice.name}
            </h4>
            {isSelected && (
              <Check className="h-4 w-4 text-primary" aria-hidden="true" />
            )}
            {hasError && (
              <Badge variant="destructive" className="text-xs">
                Preview failed
              </Badge>
            )}
          </div>
          <div className="flex gap-2 flex-wrap">
            {genderLabel && (
              <Badge variant="secondary" className="text-xs">
                {genderLabel}
              </Badge>
            )}
            <Badge variant="outline" className="text-xs">
              {languageLabel}
            </Badge>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant={isSelected ? 'default' : 'outline'}
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onPreview(voice);
          }}
          disabled={isPreviewing}
          aria-label={`Preview ${voice.name}`}
        >
          {isPreviewing ? (
            <>
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              Loading...
            </>
          ) : (
            <>
              <Play className="h-3 w-3 mr-1" />
              Preview
            </>
          )}
        </Button>
        {isBlocked && onManualPlay && (
          <Button
            variant="default"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onManualPlay(voice);
            }}
            aria-label={`Play ${voice.name}`}
            title="Click to Play (autoplay was blocked)"
          >
            Click to Play
          </Button>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

/**
 * VoiceSelection component - displays available voices grouped by language
 * 
 * @param voices - Array of available voices
 * @param selectedVoice - Currently selected voice ID
 * @param onVoiceSelect - Callback when a voice is selected
 * @param onVoicePreview - Optional callback to preview a voice
 */
export function VoiceSelection({
  voices,
  selectedVoice,
  onVoiceSelect,
  onVoicePreview,
  onVoiceManualPlay,
  showSelectionToast = true,
}: VoiceSelectionProps) {
  const [previewingVoice, setPreviewingVoice] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [blockedAutoplayVoice, setBlockedAutoplayVoice] = useState<string | null>(null);

  // Ref to hold timeout id for preview timeout so we can clear it on unmount or when starting a new preview
  const previewTimeoutRef = useRef<number | null>(null);
  // Abort controller to cancel in-flight preview async work on new preview or unmount
  const previewAbortCtrlRef = useRef<AbortController | null>(null);
  // Mounted flag to avoid updating state after unmount (used in a few handlers)
  const isMountedRef = useRef<boolean>(true);

  // Memoize grouped voices to prevent recalculation on every render
  const groupedVoices = useMemo<GroupedVoices>(() => {
    return groupVoicesByLanguage(voices);
  }, [voices]);

  // Memoize voice lookup for better performance
  const voiceMap = useMemo<Record<string, Voice>>(() => {
    return voices.reduce<Record<string, Voice>>((acc, voice) => {
      acc[voice.id] = voice;
      return acc;
    }, {});
  }, [voices]);

  /**
   * Reset preview state when the list of voices changes (explicit, not cleanup-only)
   */
  useEffect(() => {
    // Abort any in-flight preview and reset transient preview state when available voices change
    if (previewAbortCtrlRef.current) {
      previewAbortCtrlRef.current.abort();
      previewAbortCtrlRef.current = null;
    }
    // Clear any pending preview timeout so a stale callback cannot fire later
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
      previewTimeoutRef.current = null;
    }
    setPreviewingVoice(null);
    setPreviewError(null);
    setBlockedAutoplayVoice(null);
  }, [voices]);

  /**
   * Ensure we clear pending preview timeout and avoid state updates after unmount
   */
  useEffect(() => {
    return () => {
      // Mark as unmounted
      isMountedRef.current = false;

      // Abort any in-flight preview and cleanup timeout on unmount
      if (previewAbortCtrlRef.current) {
        previewAbortCtrlRef.current.abort();
        previewAbortCtrlRef.current = null;
      }
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
        previewTimeoutRef.current = null;
      }
    };
  }, []);

  /**
   * Handles voice selection with error handling
   */
  const handleVoiceSelect = useCallback((voiceId: string) => {
    try {
      const voice = voiceMap[voiceId];
      if (!voice) {
        console.error(`Voice with ID "${voiceId}" not found`);
        toast.error('Voice not found');
        return;
      }
      onVoiceSelect(voiceId);
      if (showSelectionToast) toast.success(`Voice "${voice.name}" selected`);
    } catch (error) {
      console.error('Error selecting voice:', error);
      toast.error('Failed to select voice');
    }
  }, [voiceMap, onVoiceSelect, showSelectionToast]);

  /**
   * Handles voice preview with timeout and error handling
   */
  const handlePreview = useCallback(async (voice: Voice) => {
    if (previewingVoice) {
      // Prevent multiple simultaneous previews
      toast.info('Please wait for the current preview to finish');
      return;
    }

    // Clear any previous preview timeout before starting a new preview
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
      previewTimeoutRef.current = null;
    }

    setPreviewingVoice(voice.id);
    setPreviewError(null);

    // Abort previous preview controller (if any) and create a new one for this preview
    if (previewAbortCtrlRef.current) {
      previewAbortCtrlRef.current.abort();
      previewAbortCtrlRef.current = null;
    }
    const ctrl = new AbortController();
    previewAbortCtrlRef.current = ctrl;
    const signal = ctrl.signal;

    // Set timeout to prevent stuck loading state
    previewTimeoutRef.current = window.setTimeout(() => {
      if (signal.aborted) return;
      setPreviewingVoice(null);
      toast.error('Preview timed out. Please try again.');
      previewTimeoutRef.current = null;
    }, PREVIEW_TIMEOUT_MS);

    try {
      // If the voice has no associated path/resource, gracefully inform the user
      if (!voice.path) {
        if (!signal.aborted) {
          toast.error('Preview not available for this voice');
        }
        return;
      }

      if (onVoicePreview) {
        const blocked = await onVoicePreview(voice);
        if (blocked === true) {
          if (!signal.aborted) {
            setBlockedAutoplayVoice(voice.id);
            toast.info('Autoplay blocked — click Play to start audio');
          }
        } else {
          if (!signal.aborted) {
            toast.success(`Previewed "${voice.name}"`);
            setBlockedAutoplayVoice(null);
          }
        }
      } else {
        // Fallback if no preview handler provided
        if (!signal.aborted) toast.info(`Previewing "${voice.name}"...`);
      }
    } catch (error) {
      console.error('Error previewing voice:', error);
      if (!signal.aborted) {
        setPreviewError(voice.id);
        toast.error('Failed to preview voice');
      }
    } finally {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
        previewTimeoutRef.current = null;
      }
      // Mark this preview as completed and clean up controller if still the active one
      if (previewAbortCtrlRef.current === ctrl) {
        previewAbortCtrlRef.current = null;
      }
      if (!signal.aborted) setPreviewingVoice(null);
    }
  }, [previewingVoice, onVoicePreview]);

  // Render empty state
  if (voices.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Voice Selection
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-sm">No voices available</p>
            <p className="text-xs mt-2">
              Please add voice files to the voices directory
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render voice selection
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Voice Selection
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Object.keys(groupedVoices).sort((a, b) => a.localeCompare(b)).map((language) => {
            const langVoices = groupedVoices[language];
            return (
              <div key={language} className="space-y-3">
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                  {formatLanguage(language)}
                </h3>
                <div className="space-y-2">
                  {langVoices.map((voice) => (
                    <VoiceCard
                      key={voice.id}
                      voice={voice}
                      isSelected={selectedVoice === voice.id}
                      isPreviewing={previewingVoice === voice.id}
                      hasError={previewError === voice.id}
                      isBlocked={blockedAutoplayVoice === voice.id}
                      onSelect={handleVoiceSelect}
                      onPreview={handlePreview}
                      onManualPlay={async (v) => {
                        try {
                          if (!v.path) {
                            toast.error('Preview not available for this voice');
                            return;
                          }
                          if (onVoiceManualPlay) await onVoiceManualPlay(v);
                          // Avoid updating state if component has unmounted during the await
                          if (!isMountedRef.current) return;
                          setBlockedAutoplayVoice(null);
                        } catch (err) {
                          console.error('Manual play failed', err);
                          if (isMountedRef.current) toast.error('Failed to play preview');
                        }
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
