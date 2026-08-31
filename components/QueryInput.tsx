// components/QueryInput.tsx
// Large query textarea with mode-adaptive suggested question chips.
// Level 2 surface with inset top highlight.
'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import type { AnalysisMode } from '@/lib/types/analysis';
import { Sparkles, CornerDownLeft } from 'lucide-react';

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

const MODE_LABEL: Record<AnalysisMode, string> = {
  single_image: 'Single Image',
  bi_temporal: 'Bi-Temporal',
  optical_sar: 'Optical+SAR',
};

const MODE_DOMAIN: Record<AnalysisMode, 'cyan' | 'magenta' | 'amber'> = {
  single_image: 'cyan',
  bi_temporal: 'magenta',
  optical_sar: 'amber',
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
  const charCount = value.length;
  const maxChars = 1000;
  const domain = MODE_DOMAIN[mode];
  const domainVar = `var(--${domain})`;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canSubmit && onSubmit && !disabled) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className={cn('flex flex-col gap-3', className)}>
      <div className="flex items-center justify-between">
        <label htmlFor="query-textarea" className="hud-label">
          Natural Language Query / Prompt
        </label>
        <span
          className="hud-label"
          style={{
            fontSize: '0.6rem',
            color: charCount > maxChars ? 'var(--red)' : 'var(--text-faint)',
          }}
        >
          {charCount} / {maxChars} CHARS
        </span>
      </div>

      {/* Level-2 surface with inset light source highlight on top */}
      <div
        className="panel-elevated relative rounded-md" data-level="2">
        <textarea
          id="query-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Ask anything about the satellite imagery... (e.g., 'What land cover types are visible? Locate all buildings')"
          rows={4}
          className={cn(
            'w-full p-4 rounded-md text-[15px] leading-relaxed resize-y min-h-[112px] transition-all bg-transparent',
            'text-[var(--text-primary)] focus:outline-none',
            'placeholder:text-[var(--text-faint)] placeholder:text-sm',
            disabled ? 'opacity-60 cursor-not-allowed' : ''
          )}
          style={{
            fontFamily: 'var(--font-body)',
            boxShadow: 'none',
            border: 'none',
          }}
          onFocus={(e) => {
            e.currentTarget.parentElement!.style.boxShadow = 'var(--glow-cyan)';
            e.currentTarget.parentElement!.style.borderColor = 'color-mix(in srgb, var(--cyan) 40%, transparent)';
          }}
          onBlur={(e) => {
            e.currentTarget.parentElement!.style.boxShadow = 'none';
            e.currentTarget.parentElement!.style.borderColor = 'var(--border-hairline)';
          }}
        />

        <div
          className="absolute bottom-2.5 right-3 pointer-events-none hidden sm:flex items-center gap-1.5 hud-label"
          style={{ fontSize: '0.65rem', color: 'var(--text-faint)' }}
        >
          <span>CTRL + ENTER</span>
          <CornerDownLeft className="w-3.5 h-3.5" />
          <span>TO EXECUTE</span>
        </div>
      </div>

      {/* Suggestions */}
      <div className="flex flex-col gap-2 pt-1">
        <div
          className="flex items-center gap-1.5 hud-label"
          style={{ color: domainVar, opacity: 0.9 }}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span style={{ fontSize: '0.6875rem' }}>
            Suggested Inquiries · {MODE_LABEL[mode]}:
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((suggestion, idx) => (
            <button
              key={idx}
              type="button"
              disabled={disabled}
              onClick={() => onChange(suggestion)}
              className="px-3 py-1.5 rounded text-xs transition-all text-left focus-visible:outline-none cursor-pointer"
              style={{
                fontFamily: 'var(--font-body)',
                color: 'var(--text-muted)',
                border: '1px solid var(--border-hairline)',
                background: 'var(--surface-2)',
                lineHeight: '1.4',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = 'var(--text-primary)';
                e.currentTarget.style.background = 'var(--surface-2-hover)';
                e.currentTarget.style.borderColor = `color-mix(in srgb, ${domainVar} 40%, transparent)`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = 'var(--text-muted)';
                e.currentTarget.style.background = 'var(--surface-2)';
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
