// components/ui/LoadingSkeletonPanel.tsx
// Skeleton placeholder that matches the real content layout.
'use client';

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn('skeleton rounded', className)}
      style={{ minHeight: '1rem' }}
      aria-hidden="true"
    />
  );
}

// Pre-built panel skeletons for common page layouts
export function AnalysisResultSkeleton() {
  return (
    <div className="flex flex-col gap-4 animate-fade-in-up">
      <div className="panel p-5 flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <Skeleton className="h-5 w-24 rounded" />
          <Skeleton className="h-5 w-16 rounded" />
          <div className="flex-1" />
          <Skeleton className="h-5 w-32 rounded" />
        </div>
        <Skeleton className="h-3 w-3/4" />
      </div>

      <div className="panel p-5 flex flex-col gap-3">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-5/6" />
        <Skeleton className="h-3 w-4/5" />
        <Skeleton className="h-3 w-3/4" />
      </div>

      <div className="panel p-0 overflow-hidden">
        <Skeleton className="h-64 rounded" />
      </div>
    </div>
  );
}

export function HistoryTableSkeleton() {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="panel p-4 flex items-center gap-4">
          <Skeleton className="w-12 h-12 rounded flex-shrink-0" />
          <div className="flex-1 flex flex-col gap-2">
            <Skeleton className="h-3 w-1/3" />
            <Skeleton className="h-2.5 w-1/2" />
          </div>
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-8" />
        </div>
      ))}
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="panel p-5 flex flex-col gap-3">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-8 w-16" />
          </div>
        ))}
      </div>
      <HistoryTableSkeleton />
    </div>
  );
}

export function RegistrySkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="panel p-5 flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <Skeleton className="w-8 h-8 rounded" />
            <Skeleton className="h-4 w-40" />
          </div>
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
          <div className="flex gap-2">
            <Skeleton className="h-4 w-16 rounded" />
            <Skeleton className="h-4 w-12 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
}
