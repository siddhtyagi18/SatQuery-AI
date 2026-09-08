// components/QueryInput.tsx
// Prominent, modern ChatGPT-style natural-language inquiry console.
'use client';

import { cn } from '@/lib/utils';
import type { AnalysisMode } from '@/lib/types/analysis';
import {
  Sparkles,
  ArrowUp,
  MessageSquare,
  CornerDownLeft,
  Loader2,
  Mic,
  Globe,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

const SUGGESTIONS_BY_MODE: Record<AnalysisMode, string[]> = {
  single_image: [
    'What land cover types are visible in this image?',
    'Locate and count all buildings and built structures.',
    'Identify any water bodies or reservoirs in the scene.',
    'Provide a comprehensive scene caption and description.',
    'Detect agricultural parcel boundaries and crop health indicators.',
  ],
  bi_temporal: [
    'What land-use changes occurred between these two dates?',
    'Has urban expansion decreased the vegetation cover?',
    'Identify new infrastructure, roads, or construction projects.',
    'Quantify the flood inundation or water level change extent.',
    'Provide a comparative change summary with class-wise breakdown.',
  ],
  optical_sar: [
    'Does the SAR data confirm the optical change detection?',
    'Identify features visible in SAR backscatter but invisible in optical.',
    'Detect flooded or standing water areas using SAR specular reflection.',
    'Assess building density using SAR double-bounce scattering.',
    'Highlight cross-modal discrepancies between optical and radar signals.',
  ],
};

const MODE_PROMPT_INFO: Record<
  AnalysisMode,
  {
    title: string;
    helper: string;
    placeholder: string;
    domain: 'cyan' | 'magenta' | 'amber';
  }
> = {
  single_image: {
    title: 'Single Image Natural-Language Inquiry',
    helper: 'Ask your analysis question in natural language about this scene.',
    placeholder:
      "Describe what you want to analyze... (e.g., 'What land cover types are visible? Locate all buildings and built structures')",
    domain: 'cyan',
  },
  bi_temporal: {
    title: 'Bi-Temporal Change Inquiry',
    helper: 'Describe what you want to analyze between these two temporal images.',
    placeholder:
      "Describe what you want to analyze between these two images... (e.g., 'What urban or vegetation changes occurred between T1 and T2?')",
    domain: 'magenta',
  },
  optical_sar: {
    title: 'Optical + SAR Fusion Inquiry',
    helper: 'Describe what you want to analyze from the Optical and SAR data.',
    placeholder:
      "Describe what you want to analyze from the Optical and SAR data... (e.g., 'Does the radar backscatter verify optical flood water boundaries?')",
    domain: 'amber',
  },
};

export type QueryLanguage = 'en' | 'hi';

const LANGUAGE_OPTIONS: { value: QueryLanguage; label: string; flag: string; badge: string }[] = [
  { value: 'en', label: 'English', flag: '🇬🇧', badge: 'EN' },
  { value: 'hi', label: 'हिन्दी', flag: '🇮🇳', badge: 'हिं' },
];

/**
 * Lightweight, zero-latency client-side transcript normalization.
 * Runs strictly once after the user stops speaking.
 *
 * Rules:
 *  - Normalize spoken years/numbers in Hindi & English
 *  - Collapse accidental speech-recognition word stutters (e.g. "क्या क्या" -> "क्या", "में में" -> "में")
 *  - Normalize common colloquial speech phrases (e.g. "क्या चेंज हुआ" -> "क्या बदलाव हुआ")
 *  - Preserve technical & domain words (e.g. "agriculture", "area", "sar", "radar", "optical")
 *  - Clean up punctuation & trailing clutter, ensure question mark if interrogative
 *  - NEVER invent info, NEVER add locations, NEVER change dates or user intent.
 */
export function cleanVoiceTranscript(rawText: string, _lang: QueryLanguage = 'en'): string {
  if (!rawText || !rawText.trim()) return '';

  let text = rawText.trim();

  // 1. Spoken years & numbers normalization (Hindi & English)
  const NUMBER_MAP_HI: [RegExp, string][] = [
    [/\bदो\s+(?:हजार|हज़ार)\s+पच्चीस\b/gi, '2025'],
    [/\bदो\s+(?:हजार|हज़ार)\s+चौबीस\b/gi, '2024'],
    [/\bदो\s+(?:हजार|हज़ार)\s+तेईस\b/gi, '2023'],
    [/\bदो\s+(?:हजार|हज़ार)\s+बाइस\b/gi, '2022'],
    [/\bदो\s+(?:हजार|हज़ार)\s+इक्कीस\b/gi, '2021'],
    [/\bदो\s+(?:हजार|हज़ार)\s+बीस\b/gi, '2020'],
    [/\bदो\s+(?:हजार|हज़ार)\s+उन्नीस\b/gi, '2019'],
    [/\bदो\s+(?:हजार|हज़ार)\s+अठारह\b/gi, '2018'],
    [/\bदो\s+(?:हजार|हज़ार)\s+सत्रह\b/gi, '2017'],
    [/\bदो\s+(?:हजार|हज़ार)\s+सोलह\b/gi, '2016'],
    [/\bदो\s+(?:हजार|हज़ार)\s+पंद्रह\b/gi, '2015'],
    [/\bदो\s+(?:हजार|हज़ार)\s+चौदह\b/gi, '2014'],
    [/\bदो\s+(?:हजार|हज़ार)\s+तेरह\b/gi, '2013'],
    [/\bदो\s+(?:हजार|हज़ार)\s+बारह\b/gi, '2012'],
    [/\bदो\s+(?:हजार|हज़ार)\s+ग्यारह\b/gi, '2011'],
    [/\bदो\s+(?:हजार|हज़ार)\s+दस\b/gi, '2010'],
    [/\bदो\s+(?:हजार|हज़ार)\s+नौ\b/gi, '2009'],
    [/\bदो\s+(?:हजार|हज़ार)\s+आठ\b/gi, '2008'],
    [/\bदो\s+(?:हजार|हज़ार)\s+सात\b/gi, '2007'],
    [/\bदो\s+(?:हजार|हज़ार)\s+छह\b/gi, '2006'],
    [/\bदो\s+(?:हजार|हज़ार)\s+पांच\b/gi, '2005'],
    [/\bदो\s+(?:हजार|हज़ार)\s+चार\b/gi, '2004'],
    [/\bदो\s+(?:हजार|हज़ार)\s+तीन\b/gi, '2003'],
    [/\bदो\s+(?:हजार|हज़ार)\s+दो\b/gi, '2002'],
    [/\bदो\s+(?:हजार|हज़ार)\s+एक\b/gi, '2001'],
    [/\bदो\s+(?:हजार|हज़ार)\b/gi, '2000'],
  ];

  const NUMBER_MAP_EN: [RegExp, string][] = [
    [/\btwenty\s+twenty[- ]five\b/gi, '2025'],
    [/\btwenty\s+twenty[- ]four\b/gi, '2024'],
    [/\btwenty\s+twenty[- ]three\b/gi, '2023'],
    [/\btwenty\s+twenty[- ]two\b/gi, '2022'],
    [/\btwenty\s+twenty[- ]one\b/gi, '2021'],
    [/\btwenty\s+twenty\b/gi, '2020'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty[- ]five\b/gi, '2025'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty[- ]four\b/gi, '2024'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty[- ]three\b/gi, '2023'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty[- ]two\b/gi, '2022'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty[- ]one\b/gi, '2021'],
    [/\btwo\s+thousand\s+(?:and\s+)?twenty\b/gi, '2020'],
    [/\btwo\s+thousand\b/gi, '2000'],
  ];

  for (const [pattern, rep] of NUMBER_MAP_HI) {
    text = text.replace(pattern, rep);
  }
  for (const [pattern, rep] of NUMBER_MAP_EN) {
    text = text.replace(pattern, rep);
  }

  // 2. Normalize common Hindi colloquial speech phrases
  text = text.replace(/\bक्या\s+चेंज\s+हुआ\b/gi, 'क्या बदलाव हुआ');
  text = text.replace(/\bक्या\s+चेंज\s+हुए\b/gi, 'क्या बदलाव हुए');

  // 3. Remove accidental speech stutter (repeated identical words of length >= 2)
  text = text.replace(/\b([\p{L}\d]{2,})\s+\1\b/giu, '$1');

  // 4. Whitespace cleanup
  text = text.replace(/\s+/g, ' ').trim();

  // 5. Punctuation cleanup
  text = text.replace(/^[,\-–—.:;]+\s*/, '');
  text = text.replace(/[,\-–—:;]+$/, '');

  // If query is interrogative, ensure it ends with ?
  const isQuestion =
    /^(what|where|how|which|why|did|is|are|has|have|can|could|do|does)\b/i.test(text) ||
    /(\b(क्या|कहाँ|कितना|कितने|कितनी|कैसे|क्यों|कौन|बदलाव|अंतर|परिवर्तन)\b)/i.test(text) ||
    /\b(what|where|how|which|why)\b/i.test(text);

  if (isQuestion && !text.endsWith('?')) {
    text = text.replace(/[।.]*$/, '') + '?';
  }

  return text;
}

interface QueryInputProps {
  value: string;
  onChange: (query: string) => void;
  mode: AnalysisMode;
  onSubmit?: () => void;
  disabled?: boolean;
  canSubmit?: boolean;
  isSubmitting?: boolean;
  className?: string;
  language?: QueryLanguage;
  onLanguageChange?: (lang: QueryLanguage) => void;
}

type SpeechRecognitionResultItem = { transcript?: string };
type SpeechRecognitionResult = {
  isFinal?: boolean;
  length: number;
  [index: number]: SpeechRecognitionResultItem;
};

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResult;
  };
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort?: () => void;
};

type SpeechRecognitionCtorLike = new () => SpeechRecognitionLike;

function detectSpeechRecognition(): SpeechRecognitionCtorLike | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtorLike;
    webkitSpeechRecognition?: SpeechRecognitionCtorLike;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function QueryInput({
  value,
  onChange,
  mode,
  onSubmit,
  disabled,
  canSubmit = true,
  isSubmitting = false,
  className,
  language = 'en',
  onLanguageChange,
}: QueryInputProps) {
  const suggestions = SUGGESTIONS_BY_MODE[mode] ?? SUGGESTIONS_BY_MODE.single_image;
  const promptInfo = MODE_PROMPT_INFO[mode] ?? MODE_PROMPT_INFO.single_image;
  const charCount = value.length;
  const maxChars = 1000;
  const domain = promptInfo.domain;
  const domainVar = `var(--${domain})`;

  const [isListening, setIsListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);

  // Recognition session refs
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const isListeningRef = useRef(false);
  const isExplicitStopRef = useRef(false);
  const baseTextRef = useRef('');
  const finalVoiceTranscriptRef = useRef('');
  const interimVoiceTranscriptRef = useRef('');
  const restartTimerRef = useRef<NodeJS.Timeout | null>(null);

  const languageDropdownRef = useRef<HTMLDivElement | null>(null);
  const [langDropdownOpen, setLangDropdownOpen] = useState(false);
  const valueRef = useRef(value);
  const onChangeRef = useRef(onChange);
  const languageRef = useRef(language);

  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(() => {
    languageRef.current = language;
  }, [language]);

  useEffect(() => {
    setVoiceSupported(detectSpeechRecognition() !== null);

    function handleClickOutside(e: MouseEvent) {
      const el = languageDropdownRef.current;
      if (el && !el.contains(e.target as Node)) {
        setLangDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      if (restartTimerRef.current) {
        clearTimeout(restartTimerRef.current);
      }
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort?.();
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  const stopListening = useCallback(() => {
    isExplicitStopRef.current = true;
    isListeningRef.current = false;
    setIsListening(false);

    if (restartTimerRef.current) {
      clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        /* ignore */
      }
    }

    // Flush any pending interim speech into final
    const pendingInterim = interimVoiceTranscriptRef.current.trim();
    if (pendingInterim) {
      finalVoiceTranscriptRef.current = finalVoiceTranscriptRef.current
        ? `${finalVoiceTranscriptRef.current} ${pendingInterim}`
        : pendingInterim;
      interimVoiceTranscriptRef.current = '';
    }

    // Compose complete raw transcript
    const base = baseTextRef.current.trim();
    const voice = finalVoiceTranscriptRef.current.trim();
    const complete = base ? `${base} ${voice}` : voice;

    // Apply fast lightweight transcript cleanup
    if (complete) {
      const cleaned = cleanVoiceTranscript(complete, languageRef.current);
      const limited = cleaned.slice(0, maxChars);
      onChangeRef.current(limited);
    }
  }, [maxChars]);

  const startListening = useCallback(() => {
    const SpeechRecognitionCtor = detectSpeechRecognition();
    if (!SpeechRecognitionCtor || disabled || isSubmitting) return;

    if (restartTimerRef.current) {
      clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort?.();
      } catch {
        /* ignore */
      }
      recognitionRef.current = null;
    }

    // Preserve whatever text is currently in the input as the baseline
    baseTextRef.current = valueRef.current;
    finalVoiceTranscriptRef.current = '';
    interimVoiceTranscriptRef.current = '';
    isExplicitStopRef.current = false;
    isListeningRef.current = true;
    setIsListening(true);

    try {
      const recognition = new SpeechRecognitionCtor();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.lang = languageRef.current === 'hi' ? 'hi-IN' : 'en-IN';

      recognition.onresult = (ev: SpeechRecognitionEventLike) => {
        if (!ev.results) return;

        let currentInterim = '';
        for (let i = ev.resultIndex; i < ev.results.length; ++i) {
          const res = ev.results[i];
          const chunk = res?.[0]?.transcript || '';
          if (res.isFinal) {
            const trimmed = chunk.trim();
            if (trimmed) {
              finalVoiceTranscriptRef.current = finalVoiceTranscriptRef.current
                ? `${finalVoiceTranscriptRef.current} ${trimmed}`
                : trimmed;
            }
          } else {
            currentInterim += chunk;
          }
        }
        interimVoiceTranscriptRef.current = currentInterim;

        // VISIBLE TRANSCRIPT = baseText + finalTranscript + currentInterim
        const base = baseTextRef.current.trim();
        const voiceParts = [finalVoiceTranscriptRef.current, currentInterim.trim()]
          .filter(Boolean)
          .join(' ');
        const combined = base ? `${base} ${voiceParts}` : voiceParts;
        const limited = combined.slice(0, maxChars);
        onChangeRef.current(limited);
      };

      recognition.onerror = (ev: { error?: string }) => {
        const err = ev?.error;
        if (err === 'not-allowed' || err === 'service-not-allowed') {
          isExplicitStopRef.current = true;
          isListeningRef.current = false;
          setIsListening(false);
          toast.error('Microphone access denied. You can continue typing your question.');
        }
      };

      recognition.onend = () => {
        // If user did not click stop, browser paused/timed out: gracefully restart session!
        if (!isExplicitStopRef.current && isListeningRef.current) {
          restartTimerRef.current = setTimeout(() => {
            if (!isExplicitStopRef.current && isListeningRef.current) {
              try {
                const freshRecognition = new SpeechRecognitionCtor();
                freshRecognition.continuous = true;
                freshRecognition.interimResults = true;
                freshRecognition.maxAlternatives = 1;
                freshRecognition.lang = languageRef.current === 'hi' ? 'hi-IN' : 'en-IN';
                freshRecognition.onresult = recognition.onresult;
                freshRecognition.onerror = recognition.onerror;
                freshRecognition.onend = recognition.onend;
                recognitionRef.current = freshRecognition;
                freshRecognition.start();
              } catch {
                /* transient restart error ignored */
              }
            }
          }, 150);
        } else {
          setIsListening(false);
          isListeningRef.current = false;
        }
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setIsListening(false);
      isListeningRef.current = false;
    }
  }, [disabled, isSubmitting, maxChars]);

  const toggleVoiceInput = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  const handleSelectLanguage = (newLang: QueryLanguage) => {
    if (isListening) {
      stopListening();
    }
    setLangDropdownOpen(false);
    onLanguageChange?.(newLang);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && canSubmit && onSubmit && !disabled) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-7 h-7 rounded-md flex items-center justify-center"
              style={{
                background: `color-mix(in srgb, ${domainVar} 14%, transparent)`,
                border: `1px solid color-mix(in srgb, ${domainVar} 30%, transparent)`,
              }}
            >
              <MessageSquare className="w-4 h-4" style={{ color: domainVar }} />
            </div>
            <label
              htmlFor="query-textarea"
              className="font-heading font-bold text-base text-[var(--text-primary)]"
            >
              {promptInfo.title}
            </label>
          </div>

          <span
            className="font-mono text-[0.65rem] tracking-wider"
            style={{
              color: charCount > maxChars ? 'var(--red)' : 'var(--text-faint)',
            }}
          >
            {charCount} / {maxChars}
          </span>
        </div>

        <p className="text-xs text-[var(--text-muted)] font-medium pl-9">
          {promptInfo.helper}
        </p>
      </div>

      <div
        className="relative rounded-2xl border transition-all duration-300 overflow-hidden shadow-lg"
        style={{
          background: 'var(--surface-1)',
          borderColor: value.trim()
            ? `color-mix(in srgb, ${domainVar} 45%, var(--border-hairline))`
            : 'var(--border-hairline)',
          boxShadow: value.trim()
            ? `0 0 0 1px color-mix(in srgb, ${domainVar} 25%, transparent), 0 8px 30px -4px rgba(0,0,0,0.4)`
            : '0 4px 20px -2px rgba(0,0,0,0.3)',
        }}
      >
        <textarea
          id="query-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={promptInfo.placeholder}
          rows={4}
          className={cn(
            'w-full p-5 pb-14 rounded-2xl text-[15px] sm:text-base leading-relaxed resize-y min-h-[130px] transition-all bg-transparent',
            'text-[var(--text-primary)] focus:outline-none',
            'placeholder:text-[var(--text-faint)] placeholder:text-sm placeholder:font-normal',
            disabled ? 'opacity-60 cursor-not-allowed' : ''
          )}
          style={{
            fontFamily: 'var(--font-body)',
            border: 'none',
          }}
        />

        <div
          className="absolute bottom-3 left-4 right-4 flex items-center justify-between pointer-events-none"
        >
          <div className="flex items-center gap-2 text-[0.68rem] font-mono text-[var(--text-faint)]">
            <CornerDownLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Press Enter to submit · Shift+Enter for new line</span>
            <span className="sm:hidden">Enter to submit</span>
          </div>

          <div className="flex items-center gap-2 pointer-events-auto">
            {voiceSupported && (
              <button
                type="button"
                onClick={toggleVoiceInput}
                disabled={disabled || isSubmitting}
                className={cn(
                  'flex items-center justify-center w-9 h-9 rounded-xl transition-all shadow-md border border-[var(--border-hairline)]',
                  !disabled && !isSubmitting
                    ? 'cursor-pointer hover:scale-105 active:scale-95'
                    : 'opacity-40 cursor-not-allowed'
                )}
                style={{
                  background: isListening
                    ? 'color-mix(in srgb, var(--red) 18%, var(--surface-2))'
                    : 'var(--surface-2)',
                  color: isListening ? 'var(--red)' : 'var(--text-muted)',
                  borderColor: isListening
                    ? 'color-mix(in srgb, var(--red) 40%, var(--border-hairline))'
                    : 'var(--border-hairline)',
                }}
                title={
                  isListening
                    ? `Listening… (${language === 'hi' ? 'हिन्दी' : 'English'}) — click to stop`
                    : `Voice input (${language === 'hi' ? 'हिन्दी' : 'English'}) — click to start`
                }
                aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
              >
                {isListening ? (
                  <div className="flex items-center gap-0.5">
                    <span
                      className="w-1.5 h-1.5 rounded-full bg-[var(--red)] animate-ping"
                      style={{ animationDuration: '0.8s' }}
                    />
                    <Mic className="w-4 h-4" />
                  </div>
                ) : (
                  <Mic className="w-4 h-4" />
                )}
              </button>
            )}

            {onLanguageChange && (
              <div ref={languageDropdownRef} className="relative">
                <button
                  type="button"
                  onClick={() => setLangDropdownOpen((o) => !o)}
                  disabled={disabled || isSubmitting}
                  className={cn(
                    'flex items-center justify-center gap-1.5 px-2.5 h-9 rounded-xl transition-all shadow-md border border-[var(--border-hairline)]',
                    !disabled && !isSubmitting
                      ? 'cursor-pointer hover:scale-105 active:scale-95'
                      : 'opacity-40 cursor-not-allowed'
                  )}
                  style={{
                    background: 'var(--surface-2)',
                    color: 'var(--text-muted)',
                  }}
                  title="Query language"
                  aria-label="Select query language"
                >
                  <Globe className="w-3.5 h-3.5" />
                  <span className="text-[0.75rem] leading-none">
                    {LANGUAGE_OPTIONS.find((o) => o.value === language)?.flag ?? '🇬🇧'}
                  </span>
                  <span className="font-mono text-[0.68rem] font-bold tracking-wider">
                    {LANGUAGE_OPTIONS.find((o) => o.value === language)?.badge ?? 'EN'}
                  </span>
                </button>
                {langDropdownOpen && (
                  <div
                    className="absolute right-0 bottom-11 z-50 min-w-[150px] rounded-xl border border-[var(--border-hairline)] shadow-2xl overflow-hidden animate-fade-in-up"
                    style={{ background: 'var(--surface-1)' }}
                  >
                    {LANGUAGE_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => handleSelectLanguage(opt.value)}
                        className={cn(
                          'w-full flex items-center gap-2 px-3 py-2.5 text-left text-xs transition-all cursor-pointer',
                          language === opt.value
                            ? 'text-[var(--text-primary)] font-semibold'
                            : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                        )}
                        style={{
                          background:
                            language === opt.value
                              ? `color-mix(in srgb, ${domainVar} 14%, transparent)`
                              : 'transparent',
                        }}
                      >
                        <span className="text-base leading-none">{opt.flag}</span>
                        <span className="font-medium flex-1">{opt.label}</span>
                        <span className="font-mono text-[0.65rem] opacity-60 font-bold">{opt.badge}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <button
              type="button"
              disabled={!canSubmit || disabled || isSubmitting}
              onClick={onSubmit}
              className={cn(
                'flex items-center justify-center w-9 h-9 rounded-xl transition-all shadow-md',
                canSubmit && !isSubmitting
                  ? 'cursor-pointer hover:scale-105 active:scale-95'
                  : 'opacity-40 cursor-not-allowed'
              )}
              style={{
                background: canSubmit ? domainVar : 'var(--surface-2)',
                color: canSubmit ? '#05070D' : 'var(--text-faint)',
              }}
              title={canSubmit ? 'Execute Analysis' : 'Upload image(s) and type a question'}
              aria-label="Submit Analysis Prompt"
            >
              {isSubmitting ? (
                <Loader2 className="w-4 h-4 animate-spin text-white" />
              ) : (
                <ArrowUp className="w-5 h-5 stroke-[2.5]" />
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2 pt-1">
        <div
          className="flex items-center gap-1.5 text-xs font-semibold tracking-wide"
          style={{ color: domainVar }}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Suggested Analysis Questions:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((suggestion, idx) => (
            <button
              key={idx}
              type="button"
              disabled={disabled}
              onClick={() => onChange(suggestion)}
              className="px-3.5 py-2 rounded-lg text-xs transition-all text-left focus-visible:outline-none cursor-pointer border border-[var(--border-hairline)] bg-[var(--surface-2)] hover:bg-[var(--surface-2-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] leading-normal shadow-sm"
              style={{
                fontFamily: 'var(--font-body)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = `color-mix(in srgb, ${domainVar} 40%, transparent)`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-hairline)';
              }}
            >
              &ldquo;{suggestion}&rdquo;
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
