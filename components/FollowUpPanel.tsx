// components/FollowUpPanel.tsx
// Contextual Follow-up conversational interface for SatQuery-AI.
// Reuses existing analysis context without re-running models.
// Supports English & Hindi voice input, text queries, and browser TTS playback.
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import type { AnalysisResult, FollowUpMessage, SpatialAction } from '@/lib/types/analysis';
import { api } from '@/lib/api';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { cleanVoiceTranscript } from '@/components/QueryInput';
import {
  Send,
  Mic,
  Volume2,
  Square,
  Sparkles,
  Bot,
  User,
  ArrowRight,
  Maximize2,
  CheckCircle2,
  Languages,
  Loader2,
  RotateCcw,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface FollowUpPanelProps {
  result: AnalysisResult;
  onSpatialAction?: (action: SpatialAction) => void;
  className?: string;
}

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

export function FollowUpPanel({ result, onSpatialAction, className }: FollowUpPanelProps) {
  const [messages, setMessages] = useState<FollowUpMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [activeSpeechId, setActiveSpeechId] = useState<string | null>(null);

  // Recognition ref
  const recognitionRef = useRef<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const previousAnalysisIdRef = useRef(result.id);

  // Reset conversation if analysis ID changes (TEST 7)
  useEffect(() => {
    if (previousAnalysisIdRef.current !== result.id) {
      setMessages([]);
      setInputValue('');
      setIsLoading(false);
      setIsListening(false);
      previousAnalysisIdRef.current = result.id;
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    }
  }, [result.id]);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Quick suggestions based on analysis mode
  const suggestions = [
    { label: 'How much area changed?', icon: '📐' },
    { label: 'Where exactly did it change?', icon: '🗺️' },
    { label: 'What is the confidence?', icon: '🎯' },
    { label: 'Explain this in Hindi', icon: '🇮🇳' },
  ];

  // Send message handler
  const handleSend = async (queryText?: string) => {
    const textToSend = (queryText || inputValue).trim();
    if (!textToSend || isLoading) return;

    // Detect language
    const isHindi = /[\u0900-\u097F]/.test(textToSend) || textToSend.toLowerCase().includes('hindi');
    const lang = isHindi ? 'hi' : 'en';

    const userMsg: FollowUpMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      language: lang,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await api.askFollowUp(result.id, textToSend, [...messages, userMsg], lang);

      const assistantMsg: FollowUpMessage = {
        id: `asst-${Date.now()}`,
        role: 'assistant',
        text: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        language: response.language,
        spatialAction: response.spatialAction,
        referencedMetrics: response.referencedMetrics,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // If spatial action returned, invoke callback to highlight/scroll map
      if (response.spatialAction && response.spatialAction.action !== 'none') {
        onSpatialAction?.(response.spatialAction);
      }
    } catch (err: any) {
      toast.error('Could not retrieve follow-up answer.');
      const errorMsg: FollowUpMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        text: 'Unable to process follow-up question. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        language: 'en',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Voice recording toggle
  const toggleVoice = useCallback(() => {
    if (typeof window === 'undefined') return;

    if (isListening) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {}
      }
      setIsListening(false);
      return;
    }

    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      toast.error('Speech recognition is not supported in this browser.');
      return;
    }

    try {
      const rec = new SpeechRec();
      rec.continuous = false;
      rec.interimResults = true;
      // Auto-detect language or default to en-IN with Hindi support
      rec.lang = 'en-IN';

      rec.onstart = () => {
        setIsListening(true);
      };

      rec.onresult = (e: any) => {
        let transcript = '';
        for (let i = e.resultIndex; i < e.results.length; ++i) {
          transcript += e.results[i][0].transcript;
        }
        if (transcript) {
          const cleaned = cleanVoiceTranscript(transcript);
          setInputValue(cleaned);
        }
      };

      rec.onerror = (e: any) => {
        console.warn('Speech recognition error:', e.error);
        setIsListening(false);
      };

      rec.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = rec;
      rec.start();
    } catch (err) {
      console.warn('Speech recognition start failed:', err);
      setIsListening(false);
    }
  }, [isListening]);

  // Audio TTS playback handler
  const handlePlayAudio = (msg: FollowUpMessage) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      toast.error('Voice synthesis not supported in this browser.');
      return;
    }

    if (activeSpeechId === msg.id) {
      window.speechSynthesis.cancel();
      setActiveSpeechId(null);
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(msg.text);
    const langTag = msg.language === 'hi' ? HINDI_LANG_TAG : ENGLISH_LANG_TAG;
    utterance.lang = langTag;
    utterance.rate = 0.95;

    const voices = window.speechSynthesis.getVoices();
    const voice = pickVoice(voices, langTag);
    if (voice) {
      utterance.voice = voice;
    }

    utterance.onend = () => setActiveSpeechId(null);
    utterance.onerror = () => setActiveSpeechId(null);

    setActiveSpeechId(msg.id);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <CornerFrame
      label="CONTEXTUAL FOLLOW-UP CONSOLE"
      domain={result.mode === 'bi_temporal' ? 'magenta' : 'cyan'}
      bracketSize={12}
      className={className}
    >
      <div className="panel p-5 flex flex-col gap-4 bg-[var(--bg-panel)] transition-all">
        {/* Header telemetry subtitle */}
        <div className="flex items-center justify-between flex-wrap gap-2 text-xs font-mono text-[var(--text-muted)] border-b border-[var(--border-hairline)] pb-3">
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-[var(--cyan)]" />
            <span>Active Session Memory: <strong className="text-[var(--text-primary)]">{result.id}</strong></span>
          </div>
          <span className="text-[0.68rem] px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            Zero-Rerun Telemetry Link
          </span>
        </div>

        {/* Conversation Message Feed */}
        <div className="flex flex-col gap-3 min-h-[160px] max-h-[360px] overflow-y-auto pr-1 select-text">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-6 text-center text-[var(--text-muted)] gap-2">
              <Sparkles className="w-6 h-6 text-[var(--cyan)]/60 animate-pulse" />
              <p className="text-xs max-w-md leading-relaxed">
                Ask questions about the detected changes, affected regions, severity, or request
                explanations in Hindi without re-running the satellite model pipeline.
              </p>
            </div>
          ) : (
            messages.map((msg) => {
              const isUser = msg.role === 'user';
              const isPlaying = activeSpeechId === msg.id;

              return (
                <div
                  key={msg.id}
                  className={cn(
                    'flex flex-col gap-1 max-w-[85%] rounded-lg p-3 text-xs leading-relaxed transition-all',
                    isUser
                      ? 'self-end bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-primary)]'
                      : 'self-start bg-[var(--bg-base)] border border-[var(--cyan)]/25 text-[var(--text-primary)] shadow-sm'
                  )}
                >
                  <div className="flex items-center justify-between gap-3 text-[0.65rem] font-mono text-[var(--text-muted)] mb-0.5">
                    <span className="flex items-center gap-1 font-semibold">
                      {isUser ? <User className="w-3 h-3 text-[var(--accent-signal)]" /> : <Bot className="w-3 h-3 text-[var(--cyan)]" />}
                      {isUser ? 'You' : 'SatQuery Grounded Telemetry'}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {msg.language === 'hi' && (
                        <span className="px-1 py-0.2 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[0.6rem]">
                          हिन्दी
                        </span>
                      )}
                      <span>{msg.timestamp}</span>
                    </div>
                  </div>

                  <p className="font-sans whitespace-pre-wrap">{msg.text}</p>

                  {/* Actions on assistant responses */}
                  {!isUser && (
                    <div className="flex items-center justify-between mt-2 pt-1.5 border-t border-[var(--border-hairline)] text-[0.68rem]">
                      <button
                        onClick={() => handlePlayAudio(msg)}
                        className={cn(
                          'flex items-center gap-1 px-2 py-0.5 rounded transition-all font-mono',
                          isPlaying
                            ? 'bg-rose-500/15 border border-rose-500/40 text-rose-300'
                            : 'bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--cyan)] hover:border-[var(--cyan)]/40'
                        )}
                        title={isPlaying ? 'Stop Audio' : 'Listen with Speech Synthesis'}
                      >
                        {isPlaying ? <Square className="w-2.5 h-2.5" /> : <Volume2 className="w-2.5 h-2.5" />}
                        {isPlaying ? 'Stop' : 'Voice Playback'}
                      </button>

                      {msg.spatialAction && msg.spatialAction.action !== 'none' && (
                        <button
                          onClick={() => onSpatialAction?.(msg.spatialAction!)}
                          className="flex items-center gap-1 text-[var(--cyan)] hover:underline font-mono"
                        >
                          <Maximize2 className="w-2.5 h-2.5" />
                          Focus on Map
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}

          {isLoading && (
            <div className="self-start flex items-center gap-2 px-3 py-2 rounded bg-[var(--bg-base)] border border-[var(--border-hairline)] text-xs text-[var(--text-muted)] font-mono">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--cyan)]" />
              <span>Grounding answer from existing evidence...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestion Chips */}
        <div className="flex items-center gap-1.5 flex-wrap pt-1">
          <span className="text-[0.65rem] font-mono text-[var(--text-muted)] uppercase tracking-wider mr-1">
            Suggested:
          </span>
          {suggestions.map((chip, idx) => (
            <button
              key={idx}
              disabled={isLoading}
              onClick={() => handleSend(chip.label)}
              className="flex items-center gap-1 px-2.5 py-1 rounded text-[0.7rem] font-mono bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--cyan)]/50 transition-all hover:scale-[1.02] disabled:opacity-50"
            >
              <span>{chip.icon}</span>
              <span>{chip.label}</span>
            </button>
          ))}
        </div>

        {/* Input Bar with Voice Button & Send */}
        <div className="relative flex items-center gap-2 mt-1">
          <div className="relative flex-1">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask a follow-up... (e.g. 'How much area changed?' or 'कहाँ पर हुआ?')"
              className="w-full px-3.5 py-2.5 pr-10 rounded text-xs bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--cyan)] font-sans transition-colors"
              disabled={isLoading}
            />

            {/* Mic button in input */}
            <button
              type="button"
              onClick={toggleVoice}
              disabled={isLoading}
              className={cn(
                'absolute right-2.5 top-1/2 -translate-y-1/2 p-1.5 rounded transition-all',
                isListening
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/50 animate-pulse'
                  : 'text-[var(--text-muted)] hover:text-[var(--cyan)]'
              )}
              title={isListening ? 'Stop listening' : 'Speak follow-up question (English / Hindi)'}
            >
              <Mic className="w-3.5 h-3.5" />
            </button>
          </div>

          <button
            type="button"
            onClick={() => handleSend()}
            disabled={isLoading || !inputValue.trim()}
            className="flex items-center justify-center px-4 py-2.5 rounded bg-[var(--cyan)]/20 border border-[var(--cyan)]/50 text-[var(--cyan)] hover:bg-[var(--cyan)]/30 transition-all disabled:opacity-40 disabled:pointer-events-none font-mono text-xs font-semibold shrink-0"
          >
            <Send className="w-3.5 h-3.5 mr-1" />
            Ask
          </button>
        </div>
      </div>
    </CornerFrame>
  );
}
