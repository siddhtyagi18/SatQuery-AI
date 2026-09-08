// components/MultilingualSummaryPanel.tsx
// Hindi + English bilingual result summary with browser TTS playback controls.
// Strictly additive — never re-runs analysis, never calls AI during playback.
'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import {
  Volume2,
  Pause,
  Play,
  Square,
  Languages,
  Radio,
  Globe2,
  IndianRupee,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MultilingualSummaries } from '@/lib/types/analysis';

type LangTab = 'hi' | 'en';

interface MultilingualSummaryPanelProps {
  summaries: MultilingualSummaries | null | undefined;
  fallbackAnswer?: string | null;
  className?: string;
}

type TtsState = 'idle' | 'playing' | 'paused';

const HINDI_LANG_TAG = 'hi-IN';
const ENGLISH_LANG_TAG = 'en-IN';

function pickVoice(voices: SpeechSynthesisVoice[], langTag: string): SpeechSynthesisVoice | null {
  if (!voices || voices.length === 0) return null;
  const exact = voices.find((v) => v.lang === langTag);
  if (exact) return exact;
  const prefix = langTag.split('-')[0].toLowerCase();
  const loose = voices.find((v) => (v.lang || '').toLowerCase().startsWith(prefix));
  if (loose) return loose;
  return null;
}

function hasAnyVoiceFor(voices: SpeechSynthesisVoice[], langTag: string): boolean {
  return pickVoice(voices, langTag) !== null;
}

function buildSimpleHindiFallback(enText: string): string {
  if (!enText) return 'उपग्रह तस्वीर का विश्लेषण पूरा हो गया है।';

  let text = enText;
  // Clean markdown
  text = text.replace(/[*_#`]+/g, ' ').replace(/\s+/g, ' ').trim();

  // Dictionary of remote sensing and general terms
  const TERMS: [RegExp, string][] = [
    [/\bDetected Changed Area\b/gi, 'बदला हुआ क्षेत्र'],
    [/\bChanged Area\b/gi, 'परिवर्तित क्षेत्र'],
    [/\bConfidence\b/gi, 'विश्वास स्तर'],
    [/\bHigh confidence\b/gi, 'उच्च विश्वास स्तर'],
    [/\bMedium confidence\b/gi, 'मध्यम विश्वास स्तर'],
    [/\bLow confidence\b/gi, 'कम विश्वास स्तर'],
    [/\bUrban expansion\b/gi, 'शहरी विस्तार'],
    [/\bVegetation decrease\b/gi, 'हरियाली में कमी'],
    [/\bVegetation increase\b/gi, 'हरियाली में वृद्धि'],
    [/\bVegetation cover\b/gi, 'वनस्पति आवरण'],
    [/\bWater bodies\b/gi, 'जल निकाय / तालाब'],
    [/\bWater body\b/gi, 'जल निकाय'],
    [/\bFlood inundation\b/gi, 'बाढ़ का फैलाव'],
    [/\bFlooded areas\b/gi, 'जलमग्न क्षेत्र'],
    [/\bFlood water\b/gi, 'बाढ़ का पानी'],
    [/\bAgricultural parcels\b/gi, 'कृषि खेत'],
    [/\bAgricultural land\b/gi, 'खेती की जमीन'],
    [/\bCrop health\b/gi, 'फसल की स्थिति'],
    [/\bNew construction\b/gi, 'नया निर्माण कार्य'],
    [/\bNew infrastructure\b/gi, 'नया बुनियादी ढांचा'],
    [/\bNew roads\b/gi, 'नई सड़कें'],
    [/\bNew buildings\b/gi, 'नए भवन'],
    [/\bBuildings\b/gi, 'भवन'],
    [/\bStructures\b/gi, 'ढांचे'],
    [/\bLand cover\b/gi, 'भूमि आवरण'],
    [/\bLand use\b/gi, 'भूमि उपयोग'],
    [/\bBetween\b/gi, 'के बीच'],
    [/\bBefore\b/gi, 'पहले'],
    [/\bAfter\b/gi, 'बाद में'],
    [/\bSatellite image\b/gi, 'उपग्रह तस्वीर'],
    [/\bSatellite scene\b/gi, 'उपग्रह दृश्य'],
    [/\bAnalysis completed successfully\b/gi, 'सफलतापूर्वक विश्लेषण पूरा हुआ'],
    [/\bObservable in the scene\b/gi, 'तस्वीर में दिखाई दे रहे हैं'],
    [/\bLocalized structural changes\b/gi, 'कुछ जगहों पर संरचनात्मक बदलाव'],
    [/\bAre observable\b/gi, 'दिखाई दे रहे हैं'],
    [/\bIs observable\b/gi, 'दिखाई दे रहा है'],
    [/\bNo significant change\b/gi, 'कोई बड़ा बदलाव नहीं दिखा'],
    [/\bSignificant changes\b/gi, 'महत्वपूर्ण बदलाव'],
    [/\bSpecular reflection\b/gi, 'सतही परावर्तन (रडार)'],
    [/\bDouble-bounce\b/gi, 'भवनों से रडार परावर्तन'],
    [/\bBackscatter\b/gi, 'रडार बैकस्कैटर'],
    [/\bConfirmed\b/gi, 'पुष्टि हुई'],
    [/\bVerified\b/gi, 'सत्यापित'],
  ];

  for (const [re, hi] of TERMS) {
    text = text.replace(re, hi);
  }

  return text;
}

export function MultilingualSummaryPanel({
  summaries,
  fallbackAnswer,
  className,
}: MultilingualSummaryPanelProps) {
  // ----- Pick the default tab: prefer Hindi if query was Hindi -----
  const defaultTab: LangTab = useMemo(() => {
    if (summaries?.language === 'hi') return 'hi';
    return 'en';
  }, [summaries]);

  const [activeTab, setActiveTab] = useState<LangTab>(defaultTab);
  useEffect(() => {
    setActiveTab(defaultTab);
  }, [defaultTab, summaries]);

  // ----- Extract text per language with reliable fallback -----
  const { summaryEn, summaryHi } = useMemo(() => {
    const en = summaries?.summary_en?.trim() || fallbackAnswer?.trim() || '';
    let hi = summaries?.summary_hi?.trim() || '';
    if (!hi && en) {
      hi = buildSimpleHindiFallback(en);
    }
    return { summaryEn: en, summaryHi: hi };
  }, [summaries, fallbackAnswer]);

  const activeText = activeTab === 'hi' ? summaryHi : summaryEn;

  // ----- TTS playback state -----
  const [ttsState, setTtsState] = useState<TtsState>('idle');
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [ttsSupported, setTtsSupported] = useState<boolean>(true);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Detect speech synthesis availability + load voices
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const synth = (window as any).speechSynthesis as SpeechSynthesis | undefined;
    if (!synth) {
      setTtsSupported(false);
      return;
    }
    const syncVoices = () => {
      try {
        const list = synth.getVoices() || [];
        setVoices(list);
      } catch {
        /* ignore */
      }
    };
    syncVoices();
    synth.addEventListener?.('voiceschanged', syncVoices);
    return () => {
      synth.removeEventListener?.('voiceschanged', syncVoices);
    };
  }, []);

  const canSpeakHi = useMemo(() => hasAnyVoiceFor(voices, HINDI_LANG_TAG), [voices]);
  const canSpeakEn = useMemo(
    () =>
      hasAnyVoiceFor(voices, ENGLISH_LANG_TAG) ||
      hasAnyVoiceFor(voices, 'en-US') ||
      hasAnyVoiceFor(voices, 'en-GB'),
    [voices]
  );
  const canSpeakActive = activeTab === 'hi' ? canSpeakHi : canSpeakEn;

  // Stop playback cleanly
  const stopPlayback = useCallback(() => {
    const synth = (window as any)?.speechSynthesis as SpeechSynthesis | undefined;
    if (synth) {
      try {
        synth.cancel();
      } catch {
        /* ignore */
      }
    }
    utteranceRef.current = null;
    setTtsState('idle');
  }, []);

  useEffect(() => {
    return () => stopPlayback();
  }, [stopPlayback]);

  // Speak a specific tab's text
  const playTextForTab = useCallback(
    (tab: LangTab) => {
      if (!ttsSupported) return;
      const synth = (window as any)?.speechSynthesis as SpeechSynthesis | undefined;
      if (!synth) return;

      const text = tab === 'hi' ? summaryHi : summaryEn;
      if (!text) return;

      try {
        synth.cancel();
      } catch {
        /* ignore */
      }

      const u = new SpeechSynthesisUtterance(text);
      const langTag = tab === 'hi' ? HINDI_LANG_TAG : ENGLISH_LANG_TAG;
      u.lang = langTag;
      // Slightly slower (~0.9x) for Hindi accessibility
      u.rate = tab === 'hi' ? 0.9 : 0.98;
      u.pitch = 1.0;
      u.volume = 1.0;

      const voice =
        pickVoice(voices, langTag) ||
        (tab === 'en' ? pickVoice(voices, 'en-US') || pickVoice(voices, 'en-GB') : null);
      if (voice) u.voice = voice;

      u.onstart = () => setTtsState('playing');
      u.onend = () => setTtsState('idle');
      u.onerror = () => setTtsState('idle');
      u.onpause = () => setTtsState((s) => (s === 'idle' ? s : 'paused'));
      u.onresume = () => setTtsState('playing');

      utteranceRef.current = u;
      synth.speak(u);
      setTtsState('playing');
    },
    [ttsSupported, summaryHi, summaryEn, voices]
  );

  // Tab switcher: If playing, stops current language and immediately starts new language
  const handleTabSwitch = (newTab: LangTab) => {
    if (newTab === activeTab) return;
    const wasPlaying = ttsState === 'playing';
    stopPlayback();
    setActiveTab(newTab);
    if (wasPlaying) {
      setTimeout(() => {
        playTextForTab(newTab);
      }, 60);
    }
  };

  const startPlayback = useCallback(() => {
    if (ttsState === 'paused' && utteranceRef.current) {
      const synth = (window as any)?.speechSynthesis as SpeechSynthesis | undefined;
      if (synth) {
        try {
          synth.resume();
          setTtsState('playing');
          return;
        } catch {
          /* fall through */
        }
      }
    }
    playTextForTab(activeTab);
  }, [ttsState, playTextForTab, activeTab]);

  const resumePlayback = useCallback(() => {
    const synth = (window as any)?.speechSynthesis as SpeechSynthesis | undefined;
    if (synth) {
      try {
        synth.resume();
        setTtsState('playing');
        return;
      } catch {
        /* fall through */
      }
    }
    playTextForTab(activeTab);
  }, [playTextForTab, activeTab]);

  const pausePlayback = useCallback(() => {
    const synth = (window as any)?.speechSynthesis as SpeechSynthesis | undefined;
    if (!synth) return;
    try {
      synth.pause();
      setTtsState('paused');
    } catch {
      setTtsState('idle');
    }
  }, []);

  const hasAnySummary = Boolean(summaryEn || summaryHi);
  if (!hasAnySummary) {
    return null;
  }

  const ttsButtonDisabled = !ttsSupported || !canSpeakActive || !activeText;

  return (
    <CornerFrame
      label="MULTILINGUAL SUMMARY · द्विभाषी परिणाम"
      domain="magenta"
      className={className}
    >
      <div className="panel overflow-hidden flex flex-col bg-[var(--bg-base)] transition-all duration-300 hover:border-[var(--magenta)]/35 hover:shadow-lg">
        {/* Toolbar: Language Tabs + TTS Controls */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex-wrap gap-3">
          {/* Tabs: [ 🇮🇳 हिंदी ] [ 🇬🇧 English ] */}
          <div className="flex items-center bg-[var(--bg-panel-elevated)] p-0.5 rounded-lg border border-[var(--border-hairline)] gap-1">
            <button
              type="button"
              onClick={() => handleTabSwitch('hi')}
              className={cn(
                'px-3 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer',
                activeTab === 'hi'
                  ? 'bg-[var(--magenta)] text-[#05070D] font-bold shadow-[0_0_10px_rgba(244,114,182,0.35)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              )}
            >
              <span>🇮🇳</span>
              <span>हिंदी</span>
            </button>
            <button
              type="button"
              onClick={() => handleTabSwitch('en')}
              className={cn(
                'px-3 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer',
                activeTab === 'en'
                  ? 'bg-[var(--cyan)] text-[#05070D] font-bold shadow-[0_0_10px_rgba(56,189,248,0.35)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              )}
            >
              <span>🇬🇧</span>
              <span>English</span>
            </button>
          </div>

          {/* TTS Playback controls */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {ttsSupported ? (
              <>
                <span className="text-[0.65rem] font-mono text-[var(--text-faint)] flex items-center gap-1 mr-1">
                  <Radio className="w-3 h-3" />
                  {activeTab === 'hi' ? 'Audio · ऑडियो' : 'Audio'}
                </span>

                {/* Primary Voice Action Button (Play / Resume) */}
                {ttsState === 'paused' ? (
                  <button
                    type="button"
                    onClick={resumePlayback}
                    className="flex items-center gap-1.5 px-3 py-1 rounded text-[0.68rem] font-mono border border-[var(--cyan)]/50 bg-[var(--cyan)]/20 text-[var(--cyan)] hover:bg-[var(--cyan)]/30 transition-all cursor-pointer shadow-[0_0_8px_rgba(56,189,248,0.25)]"
                    title="Resume playback"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>▶ Resume</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={startPlayback}
                    disabled={ttsButtonDisabled || ttsState === 'playing'}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1 rounded text-[0.68rem] font-mono border transition-all',
                      ttsState === 'playing'
                        ? 'bg-[var(--magenta)]/20 border-[var(--magenta)]/50 text-[var(--magenta)] cursor-default'
                        : 'bg-[var(--surface-2)] border-[var(--border-hairline)] text-[var(--text-primary)] hover:border-[var(--magenta)]/50 hover:bg-[var(--surface-3)] cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed'
                    )}
                    title={activeTab === 'hi' ? 'हिंदी में सुनें' : 'Listen in English'}
                  >
                    {ttsState === 'playing' ? (
                      <Volume2 className="w-3.5 h-3.5 animate-pulse" />
                    ) : (
                      <Volume2 className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {ttsState === 'playing'
                        ? 'Playing…'
                        : activeTab === 'hi'
                          ? '🔊 हिंदी में सुनें'
                          : '🔊 Listen in English'}
                    </span>
                  </button>
                )}

                {/* Pause Button */}
                <button
                  type="button"
                  onClick={pausePlayback}
                  disabled={ttsState !== 'playing'}
                  className="flex items-center gap-1 px-2.5 py-1 rounded text-[0.68rem] font-mono bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--amber)]/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                  title="Pause"
                >
                  <Pause className="w-3 h-3" />
                  <span>⏸ Pause</span>
                </button>

                {/* Stop Button */}
                <button
                  type="button"
                  onClick={stopPlayback}
                  disabled={ttsState === 'idle'}
                  className="flex items-center gap-1 px-2.5 py-1 rounded text-[0.68rem] font-mono bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--red)]/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                  title="Stop"
                >
                  <Square className="w-3 h-3" />
                  <span>⏹ Stop</span>
                </button>
              </>
            ) : (
              <span className="text-[0.62rem] font-mono text-[var(--text-faint)] flex items-center gap-1">
                <Languages className="w-3 h-3" />
                Browser TTS unavailable
              </span>
            )}
          </div>
        </div>

        {/* Summary body */}
        <div className="p-5 flex flex-col gap-3 min-h-[120px]">
          <div className="flex items-center gap-2 pb-2 border-b border-[var(--border-hairline)]">
            <span
              className={cn(
                'w-5 h-5 rounded flex items-center justify-center',
                activeTab === 'hi'
                  ? 'bg-[var(--magenta)]/15 text-[var(--magenta)]'
                  : 'bg-[var(--cyan)]/15 text-[var(--cyan)]'
              )}
            >
              <Sparkles className="w-3 h-3" />
            </span>
            <span className="hud-label" style={{ fontSize: '0.6rem' }}>
              {activeTab === 'hi'
                ? 'सादा भाषा में सत्यापित नतीजा · Simple Hindi'
                : 'Executive English Summary'}
            </span>
            {summaries?.generated_via_llm && (
              <span className="badge badge-cyan ml-auto" style={{ fontSize: '0.58rem' }}>
                LLM Refined
              </span>
            )}
            {!summaries?.generated_via_llm && summaryHi && (
              <span className="badge badge-amber ml-auto" style={{ fontSize: '0.58rem' }}>
                Dictionary Fallback
              </span>
            )}
          </div>

          <div
            className={cn(
              'text-[15.5px] leading-[1.75] font-body',
              activeTab === 'hi' ? 'text-[var(--text-primary)]' : 'text-[var(--text-primary)]'
            )}
            style={
              activeTab === 'hi'
                ? { fontFamily: "'Noto Sans Devanagari', 'Baloo 2', system-ui, sans-serif" }
                : undefined
            }
          >
            {activeText ? (
              <p className="whitespace-pre-line">{activeText}</p>
            ) : (
              <p className="text-[var(--text-faint)] italic font-mono text-xs">
                {activeTab === 'hi' ? 'हिंदी सारांश उपलब्ध नहीं।' : 'No summary available.'}
              </p>
            )}
          </div>

          {/* Bullet points when available (used for chunked TTS if user clicks bullets) */}
          {activeTab === 'hi' && summaries?.bullet_hi && summaries.bullet_hi.length > 0 && (
            <div className="mt-1 pt-3 border-t border-[var(--border-hairline)] flex flex-col gap-2">
              <span className="hud-label" style={{ fontSize: '0.58rem' }}>
                मुख्य बातें · Key Findings
              </span>
              {summaries.bullet_hi.map((b, i) => (
                <div key={i} className="flex items-start gap-2.5 pl-1 py-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--magenta)] mt-2 flex-shrink-0 shadow-[0_0_6px_var(--magenta)]" />
                  <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">{b}</p>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'en' && summaries?.bullet_en && summaries.bullet_en.length > 0 && (
            <div className="mt-1 pt-3 border-t border-[var(--border-hairline)] flex flex-col gap-2">
              <span className="hud-label" style={{ fontSize: '0.58rem' }}>
                Key Findings
              </span>
              {summaries.bullet_en.map((b, i) => (
                <div key={i} className="flex items-start gap-2.5 pl-1 py-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--cyan)] mt-2 flex-shrink-0 shadow-[0_0_6px_var(--cyan)]" />
                  <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">{b}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </CornerFrame>
  );
}
