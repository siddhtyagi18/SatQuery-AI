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
} from 'lucide-react';

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

interface QueryInputProps {
  value: string;
  onChange: (query: string) => void;
  mode: AnalysisMode;
  onSubmit?: () => void;
  disabled?: boolean;
  canSubmit?: boolean;
  isSubmitting?: boolean;
  className?: string;
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
}: QueryInputProps) {
  const suggestions = SUGGESTIONS_BY_MODE[mode] ?? SUGGESTIONS_BY_MODE.single_image;
  const promptInfo = MODE_PROMPT_INFO[mode] ?? MODE_PROMPT_INFO.single_image;
  const charCount = value.length;
  const maxChars = 1000;
  const domain = promptInfo.domain;
  const domainVar = `var(--${domain})`;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Both Enter (without Shift) and Ctrl+Enter trigger submit
    if (e.key === 'Enter' && !e.shiftKey && canSubmit && onSubmit && !disabled) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {/* Prominent Header with Helper Text */}
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

      {/* ChatGPT-style Large Multi-Line Prompt Box */}
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

        {/* Bottom Toolbar inside the prompt container */}
        <div
          className="absolute bottom-3 left-4 right-4 flex items-center justify-between pointer-events-none"
        >
          <div className="flex items-center gap-2 text-[0.68rem] font-mono text-[var(--text-faint)]">
            <CornerDownLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Press Enter to submit · Shift+Enter for new line</span>
            <span className="sm:hidden">Enter to submit</span>
          </div>

          <button
            type="button"
            disabled={!canSubmit || disabled || isSubmitting}
            onClick={onSubmit}
            className={cn(
              'pointer-events-auto flex items-center justify-center w-9 h-9 rounded-xl transition-all shadow-md',
              canSubmit && !isSubmitting
                ? 'cursor-pointer hover:scale-105 active:scale-95'
                : 'opacity-40 cursor-not-allowed'
            )}
            style={{
              background: canSubmit
                ? domainVar
                : 'var(--surface-2)',
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

      {/* Suggested Inquiries / Prompt Chips */}
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
