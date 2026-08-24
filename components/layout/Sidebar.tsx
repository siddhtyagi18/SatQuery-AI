// components/layout/Sidebar.tsx
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  PlusCircle,
  History,
  Award,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Satellite,
  Radio,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  exact?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard, exact: true },
  { name: 'New Analysis', href: '/analysis/new', icon: PlusCircle },
  { name: 'Analysis History', href: '/analysis/history', icon: History },
  { name: 'Benchmark & Eval', href: '/benchmark', icon: Award },
  { name: 'Specialist Registry', href: '/registry', icon: Cpu },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        'hidden md:flex flex-col justify-between border-r border-[var(--border-hairline)] bg-[var(--bg-panel)] transition-all duration-300 z-30 flex-shrink-0',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Top Section: Brand & Nav */}
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-hairline)] h-16">
          {!collapsed && (
            <Link href="/" className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded bg-gradient-to-br from-[#3ED0FF] to-[#0A8ECC] flex items-center justify-center text-[#05070D] font-bold shadow-md flex-shrink-0">
                <Satellite className="w-4.5 h-4.5" />
              </div>
              <div className="flex flex-col">
                <span className="font-heading font-bold text-sm tracking-wider text-[var(--text-primary)]">
                  SatQuery<span className="text-[var(--accent-signal)]">-AI</span>
                </span>
                <span className="font-mono text-[0.6rem] text-[var(--text-faint)] tracking-widest uppercase">
                  ISRO Mission Ctrl
                </span>
              </div>
            </Link>
          )}

          {collapsed && (
            <Link href="/" className="mx-auto" title="SatQuery-AI">
              <div className="w-8 h-8 rounded bg-gradient-to-br from-[#3ED0FF] to-[#0A8ECC] flex items-center justify-center text-[#05070D] font-bold">
                <Satellite className="w-4.5 h-4.5" />
              </div>
            </Link>
          )}
        </div>

        {/* Nav Links */}
        <nav className="flex flex-col gap-1 p-2 mt-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = item.exact
              ? pathname === item.href
              : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all group relative',
                  isActive
                    ? 'bg-[var(--accent-signal)]/10 text-[var(--accent-signal)] border border-[var(--accent-signal)]/30 font-semibold'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel-hover)] border border-transparent'
                )}
                title={collapsed ? item.name : undefined}
              >
                <Icon
                  className={cn(
                    'w-4 h-4 flex-shrink-0 transition-colors',
                    isActive ? 'text-[var(--accent-signal)]' : 'text-[var(--text-muted)] group-hover:text-[var(--text-primary)]'
                  )}
                />
                {!collapsed && <span>{item.name}</span>}

                {/* Active indicator bar */}
                {isActive && (
                  <span
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-4 rounded-r bg-[var(--accent-signal)]"
                    aria-hidden="true"
                  />
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Bottom Section: Telemetry & Collapse */}
      <div className="flex flex-col border-t border-[var(--border-hairline)] p-3 gap-3">
        {!collapsed && (
          <div className="p-2.5 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] flex flex-col gap-1 text-[0.65rem] font-mono">
            <div className="flex items-center justify-between text-[var(--text-muted)]">
              <span className="flex items-center gap-1">
                <Radio className="w-3 h-3 text-[var(--accent-success)] animate-pulse" />
                Telemetry Stream
              </span>
              <span className="text-[var(--accent-success)]">SYNCED</span>
            </div>
            <span className="text-[var(--text-faint)] truncate">
              Node: ISRO-GEO-VQA-01
            </span>
          </div>
        )}

        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center p-2 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel-hover)] transition-colors border border-[var(--border-hairline)]"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  );
}
