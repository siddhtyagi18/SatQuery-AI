// components/GroundingOverlay.tsx
// Renders SVG/HTML bounding box overlays with labels and confidence scores.
'use client';

import { useState } from 'react';
import type { BoundingBox } from '@/lib/types/analysis';
import { cn } from '@/lib/utils';

interface GroundingOverlayProps {
  boxes: BoundingBox[];
  className?: string;
  onSelectBox?: (box: BoundingBox | null) => void;
}

export function GroundingOverlay({ boxes, className, onSelectBox }: GroundingOverlayProps) {
  const [hoveredBox, setHoveredBox] = useState<BoundingBox | null>(null);

  if (!boxes || boxes.length === 0) return null;

  return (
    <div className={cn('absolute inset-0 pointer-events-none z-10', className)}>
      {boxes.map((box, index) => {
        const isHovered = hoveredBox === box;
        const left = `${box.x * 100}%`;
        const top = `${box.y * 100}%`;
        const width = `${box.width * 100}%`;
        const height = `${box.height * 100}%`;

        // Color based on label or confidence
        const isWater = box.label.toLowerCase().includes('water');
        const isVeg = box.label.toLowerCase().includes('veg');
        const isSar = box.label.toLowerCase().includes('sar');
        const color = isWater ? '#3ED0FF' : isVeg ? '#3DDC84' : isSar ? '#FFB020' : '#FF5C5C';

        return (
          <div
            key={index}
            className="absolute transition-all duration-150 pointer-events-auto cursor-pointer"
            style={{
              left,
              top,
              width,
              height,
              border: `2px solid ${color}`,
              backgroundColor: isHovered ? `${color}25` : `${color}10`,
              boxShadow: isHovered ? `0 0 12px ${color}80` : `0 0 4px ${color}40`,
            }}
            onMouseEnter={() => {
              setHoveredBox(box);
              onSelectBox?.(box);
            }}
            onMouseLeave={() => {
              setHoveredBox(null);
              onSelectBox?.(null);
            }}
          >
            {/* Corner crosshairs on box */}
            <span
              className="absolute -top-1 -left-1 w-1.5 h-1.5"
              style={{ borderTop: `2px solid ${color}`, borderLeft: `2px solid ${color}` }}
            />
            <span
              className="absolute -bottom-1 -right-1 w-1.5 h-1.5"
              style={{ borderBottom: `2px solid ${color}`, borderRight: `2px solid ${color}` }}
            />

            {/* Label tag */}
            <div
              className={cn(
                'absolute -top-6 left-0 px-1.5 py-0.5 rounded text-[0.6rem] font-mono whitespace-nowrap',
                'flex items-center gap-1 shadow-md transition-all',
                isHovered ? 'scale-105 z-30' : 'z-20'
              )}
              style={{
                backgroundColor: 'rgba(9, 14, 23, 0.9)',
                border: `1px solid ${color}`,
                color: '#E8ECF4',
              }}
            >
              <span className="font-semibold" style={{ color }}>{box.label}</span>
              <span className="opacity-75">{(box.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
