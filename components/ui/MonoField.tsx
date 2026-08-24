// components/ui/MonoField.tsx
// HUD-style label + mono value pair, used in ImageMetadata and trace panels.
'use client';

import { cn } from '@/lib/utils';

interface MonoFieldProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  unavailableText?: string;
  className?: string;
  valueClassName?: string;
}

export function MonoField({
  label,
  value,
  unit,
  unavailableText = 'N/A',
  className,
  valueClassName,
}: MonoFieldProps) {
  const displayValue = value != null ? `${value}${unit ? ` ${unit}` : ''}` : unavailableText;
  const isUnavailable = value == null;

  return (
    <div className={cn('flex flex-col gap-0.5', className)}>
      <span className="hud-label">{label}</span>
      <span
        className={cn(
          'hud-value',
          isUnavailable ? 'text-[var(--text-faint)] italic' : '',
          valueClassName
        )}
      >
        {displayValue}
      </span>
    </div>
  );
}
