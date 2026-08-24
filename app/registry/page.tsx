// app/registry/page.tsx
// Specialist Model & Tool Registry — doubles as an architectural explainer for judges.
'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { ToolDefinition } from '@/lib/types/analysis';
import { ToolRegistry } from '@/components/ToolRegistry';
import { RegistrySkeleton } from '@/components/ui/LoadingSkeletonPanel';
import { Cpu, Network, ShieldCheck } from 'lucide-react';

export default function RegistryPage() {
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listTools()
      .then((data) => {
        setTools(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4 border-b border-[var(--border-hairline)] pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Network className="w-5 h-5 text-[var(--accent-signal)]" />
            <h1 className="text-xl md:text-2xl font-bold font-heading text-[var(--text-primary)]">
              Multi-Agent Specialist Registry
            </h1>
          </div>
          <p className="text-xs text-[var(--text-muted)] font-mono">
            Ground-station neural toolset documentation and dispatch contract specifications.
          </p>
        </div>
      </div>

      {loading ? <RegistrySkeleton /> : <ToolRegistry tools={tools} />}
    </div>
  );
}
