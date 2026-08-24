// components/ui/CornerFrame.tsx
// Reticle-corner bracket wrapper — "targeting HUD" framing motif.
// Signature visual identity — reserved for sensor/analysis active surfaces.
// NOT applied indiscriminately to every card — keeps it intentional.
'use client';

import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

type DomainColor = 'cyan' | 'magenta' | 'amber' | 'green';

interface CornerFrameProps {
  children: ReactNode;
  className?: string;
  intensity?: 'subtle' | 'normal' | 'strong';
  domain?: DomainColor;
  label?: string;
  bracketSize?: 12 | 14 | 16;
}

const OPACITY_MAP: Record<NonNullable<CornerFrameProps['intensity']>, number> = {
  subtle: 0.3,
  normal: 0.6,
  strong: 0.85,
};

const DOMAIN_VAR: Record<DomainColor, string> = {
  cyan:    'var(--cyan)',
  magenta: 'var(--magenta)',
  amber:   'var(--amber)',
  green:   'var(--green)',
};

export function CornerFrame({
  children,
  className,
  intensity = 'normal',
  domain = 'cyan',
  label,
  bracketSize = 14,
}: CornerFrameProps) {
  const opacity = OPACITY_MAP[intensity];
  const color = DOMAIN_VAR[domain];

  const bracketStyle: React.CSSProperties = {
    width: `${bracketSize}px`,
    height: `${bracketSize}px`,
    borderColor: color,
    opacity,
  };

  return (
    <div
      className={cn('relative corner-frame', className)}
      data-domain={domain}
    >
      <span
        aria-hidden="true"
        className="cf-bracket cf-tl"
        style={bracketStyle}
      />
      <span
        aria-hidden="true"
        className="cf-bracket cf-tr"
        style={bracketStyle}
      />
      <span
        aria-hidden="true"
        className="cf-bracket cf-bl"
        style={bracketStyle}
      />
      <span
        aria-hidden="true"
        className="cf-bracket cf-br"
        style={bracketStyle}
      />

      {label && (
        <span
          className="cf-label"
          style={{
            color,
            backgroundColor: 'var(--surface-1)',
            opacity: Math.min(opacity + 0.2, 1),
          }}
        >
          {label}
        </span>
      )}

      {children}
    </div>
  );
}
