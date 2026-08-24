// components/ToolRegistry.tsx
// Specialist models & tools registry view (architecture explainer for judges).
'use client';

import type { ToolDefinition } from '@/lib/types/analysis';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Cpu, Layers, GitCompare, Radar, Eye, ShieldCheck, Box } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ToolRegistryProps {
  tools: ToolDefinition[];
  className?: string;
}

const TASK_ICONS: Record<string, React.ElementType> = {
  vqa: Eye,
  captioning: Eye,
  grounding: Box,
  change_detection: GitCompare,
  change_vqa: GitCompare,
  change_description: GitCompare,
};

export function ToolRegistry({ tools, className }: ToolRegistryProps) {
  return (
    <div className={cn('flex flex-col gap-6', className)}>
      <div className="flex flex-col gap-1">
        <h2 className="text-xl font-bold text-[var(--text-primary)] font-heading">
          Specialist Model & Tool Registry
        </h2>
        <p className="text-xs text-[var(--text-muted)] leading-relaxed max-w-3xl">
          Modular vision-language and remote-sensing specialist agents orchestrated dynamically
          based on query intent and sensor input modalities.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tools.map((tool) => {
          const isOrchestrator = tool.id === 'orchestrator-v1';
          return (
            <CornerFrame key={tool.id} label={tool.id.toUpperCase()}>
              <div
                className={cn(
                  'panel p-5 flex flex-col justify-between gap-4 h-full transition-all',
                  isOrchestrator && 'border-[var(--accent-signal)]/40 bg-[var(--accent-signal)]/5'
                )}
              >
                <div className="flex flex-col gap-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2.5">
                      <div
                        className={cn(
                          'w-8 h-8 rounded flex items-center justify-center border',
                          isOrchestrator
                            ? 'bg-[var(--accent-signal)]/20 border-[var(--accent-signal)]/40 text-[var(--accent-signal)]'
                            : 'bg-[var(--bg-panel-elevated)] border-[var(--border-hairline)] text-[var(--text-primary)]'
                        )}
                      >
                        <Cpu className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-[var(--text-primary)] font-heading">
                          {tool.name}
                        </h3>
                        <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">
                          v{tool.version}
                        </span>
                      </div>
                    </div>

                    <StatusBadge status={tool.status} />
                  </div>

                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    {tool.description}
                  </p>
                </div>

                <div className="flex flex-col gap-2 pt-3 border-t border-[var(--border-hairline)]">
                  <div className="flex items-center justify-between text-[0.65rem] font-mono">
                    <span className="text-[var(--text-faint)]">Supported Tasks:</span>
                    <div className="flex flex-wrap gap-1">
                      {tool.taskTypes.map((task) => (
                        <span
                          key={task}
                          className="px-1.5 py-0.5 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-muted)]"
                        >
                          {task}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[0.65rem] font-mono">
                    <span className="text-[var(--text-faint)]">Modalities:</span>
                    <div className="flex flex-wrap gap-1">
                      {tool.supportedModalities.map((mod) => (
                        <span
                          key={mod}
                          className="px-1.5 py-0.5 rounded uppercase text-[0.6rem] bg-[var(--accent-signal)]/10 text-[var(--accent-signal)]"
                        >
                          {mod}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CornerFrame>
          );
        })}
      </div>
    </div>
  );
}
