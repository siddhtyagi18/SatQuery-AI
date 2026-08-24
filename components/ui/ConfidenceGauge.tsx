// components/ui/ConfidenceGauge.tsx
// Arc/semicircle gauge for displaying confidence scores (0–1)
'use client';

import { cn } from '@/lib/utils';

interface ConfidenceGaugeProps {
  score: number | null; // 0-1
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

function getColor(score: number): string {
  if (score >= 0.8) return '#3DDC84';
  if (score >= 0.6) return '#FFB020';
  return '#FF5C5C';
}

const SIZE_CONFIG = {
  sm: { r: 24, stroke: 5, viewBox: '0 0 60 60', cx: 30, cy: 30, textSize: '10px' },
  md: { r: 34, stroke: 6, viewBox: '0 0 80 80', cx: 40, cy: 40, textSize: '13px' },
  lg: { r: 44, stroke: 7, viewBox: '0 0 100 100', cx: 50, cy: 50, textSize: '16px' },
};

export function ConfidenceGauge({ score, size = 'md', className }: ConfidenceGaugeProps) {
  const cfg = SIZE_CONFIG[size];
  const { r, stroke, viewBox, cx, cy, textSize } = cfg;
  const circumference = 2 * Math.PI * r;
  // Use 75% of circumference for the arc (270 deg)
  const arcLen = circumference * 0.75;
  const scoreVal = score ?? 0;
  const filled = arcLen * scoreVal;
  const color = score != null ? getColor(scoreVal) : '#4A5270';

  return (
    <div className={cn('flex flex-col items-center gap-1', className)}>
      <svg
        viewBox={viewBox}
        width={size === 'lg' ? 100 : size === 'md' ? 80 : 60}
        height={size === 'lg' ? 100 : size === 'md' ? 80 : 60}
        style={{ transform: 'rotate(135deg)' }}
        aria-label={`Confidence gauge: ${score != null ? Math.round(score * 100) + '%' : 'not available'}`}
        role="img"
      >
        {/* Track */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke="rgba(62,208,255,0.08)"
          strokeWidth={stroke}
          strokeDasharray={`${arcLen} ${circumference}`}
          strokeLinecap="round"
        />
        {/* Fill */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.6s ease, stroke 0.3s ease' }}
        />
        {/* Score text — counter-rotate */}
        <text
          x={cx} y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={textSize}
          fontFamily="'JetBrains Mono', monospace"
          fontWeight="600"
          fill={color}
          style={{ transform: `rotate(-135deg)`, transformOrigin: `${cx}px ${cy}px` }}
        >
          {score != null ? `${Math.round(score * 100)}%` : '—'}
        </text>
      </svg>
    </div>
  );
}
