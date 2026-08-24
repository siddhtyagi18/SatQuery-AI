// components/layout/TopBar.tsx
// Three-zone top bar:
//   LEFT:  Product logo + name (branding zone)
//   MID:   Compact LIVE mission status badge (mission zone)
//   RIGHT: Grouped system controls pill — demo mode · theme · user (system zone)
'use client';

import { useTheme } from 'next-themes';
import {
  Sun, Moon, Radio, Shield, Satellite,
  Menu, X, PlusCircle, LayoutDashboard, History, Award, Cpu,
} from 'lucide-react';
import { DEMO_BADGE_TEXT } from '@/lib/config';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

export function TopBar() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header
      className="h-16 border-b border-[var(--border-hairline)] bg-[var(--surface-1)]/90 backdrop-blur-md z-40 sticky top-0"
    >
      <div className="h-full px-4 md:px-6 flex items-center justify-between gap-6">
        {/* ============================================================
           ZONE 1 — LEFT: Branding (logo + product name)
           ============================================================ */}
        <div className="flex items-center gap-3 min-w-0 flex-shrink-0">
          {/* Mobile hamburger */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-2)] border border-[var(--border-hairline)]"
            aria-label="Toggle mobile menu"
          >
            {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>

          {/* Product mark */}
          <div className="flex items-center gap-2.5">
            <div
              className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0"
              style={{
                background: 'linear-gradient(135deg, color-mix(in srgb, var(--cyan) 22%, transparent) 0%, color-mix(in srgb, var(--cyan) 6%, transparent) 100%)',
                border: '1px solid color-mix(in srgb, var(--cyan) 30%, transparent)',
              }}
            >
              <Satellite className="w-4 h-4" style={{ color: 'var(--cyan)' }} strokeWidth={2} />
            </div>
            <div className="flex flex-col leading-tight min-w-0">
              <span
                className="font-heading font-semibold text-sm tracking-tight"
                style={{ color: 'var(--text-primary)' }}
              >
                SatQuery-AI
              </span>
              <span className="hud-label hidden sm:inline" style={{ fontSize: '0.55rem' }}>
                Intelligence Console
              </span>
            </div>
          </div>
        </div>

        {/* ============================================================
           ZONE 2 — CENTER-LEFT: Mission status badge (single compact pill)
           ============================================================ */}
        <div className="flex-1 flex items-center justify-center max-md:hidden">
          <span
            className="badge badge-green"
            style={{ padding: '4px 10px', fontSize: '0.65rem' }}
          >
            <span className="relative flex h-1.5 w-1.5 -ml-0.5">
              <span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
            </span>
            LIVE — NRSC / SAC Hyderabad
          </span>
        </div>

        {/* Mobile LIVE indicator (single line) */}
        <div className="md:hidden flex items-center gap-2">
          <span className="badge badge-green" style={{ fontSize: '0.6rem', padding: '2px 6px' }}>
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
            </span>
            LIVE
          </span>
        </div>

        {/* ============================================================
           ZONE 3 — RIGHT: Grouped system controls pill
           Single visible bordered container = "these go together"
           ============================================================ */}
        <div className="flex items-center flex-shrink-0">
          <div
            className="flex items-center gap-1 p-1 rounded"
            style={{
              border: '1px solid var(--border-hairline)',
              background: 'var(--surface-2)',
            }}
          >
            {/* Demo mode badge */}
            <span
              className="badge badge-cyan"
              style={{ border: 'none', background: 'transparent', padding: '2px 8px' }}
            >
              <Radio className="w-3 h-3 animate-pulse-dot" />
              <span className="hidden sm:inline">{DEMO_BADGE_TEXT}</span>
              <span className="sm:hidden">DEMO</span>
            </span>

            {/* Divider */}
            <div
              className="w-px h-5 mx-0.5"
              style={{ background: 'var(--border-hairline)' }}
              aria-hidden
            />

            {/* Theme toggle */}
            {mounted && (
              <button
                type="button"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                className="p-1.5 rounded transition-colors hover:bg-[var(--surface-2-hover)]"
                title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
                aria-label="Toggle theme"
                style={{ color: 'var(--text-muted)' }}
              >
                {theme === 'dark'
                  ? <Sun className="w-4 h-4" style={{ color: 'var(--amber)' }} />
                  : <Moon className="w-4 h-4" style={{ color: 'var(--cyan)' }} />
                }
              </button>
            )}

            {/* Divider */}
            <div
              className="w-px h-5 mx-0.5 hidden sm:block"
              style={{ background: 'var(--border-hairline)' }}
              aria-hidden
            />

            {/* Mission operator identity */}
            <div className="hidden sm:flex items-center gap-2 pl-1 pr-1">
              <div
                className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0"
                style={{
                  background: 'color-mix(in srgb, var(--cyan) 12%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--cyan) 25%, transparent)',
                }}
              >
                <span
                  className="font-mono text-[0.6rem] font-bold"
                  style={{ color: 'var(--cyan)' }}
                >
                  SIH
                </span>
              </div>
              <div className="flex flex-col text-left leading-tight">
                <span
                  className="text-[0.65rem] font-semibold"
                  style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}
                >
                  Judge Demo
                </span>
                <span className="hud-label" style={{ fontSize: '0.52rem' }}>
                  L-3 Sci
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile nav dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-16 left-0 right-0 bg-[var(--surface-1)] border-b border-[var(--border-hairline)] p-3 flex flex-col gap-1 shadow-2xl animate-fade-in-up z-50">
          {[
            { name: 'Dashboard',        href: '/',                icon: LayoutDashboard },
            { name: 'New Analysis',     href: '/analysis/new',    icon: PlusCircle },
            { name: 'Analysis History', href: '/analysis/history', icon: History },
            { name: 'Benchmark & Eval', href: '/benchmark',        icon: Award },
            { name: 'Specialist Registry', href: '/registry',      icon: Cpu },
          ].map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                  'flex items-center gap-3 px-3 py-2 rounded badge',
                  isActive
                    ? 'badge-cyan'
                    : 'badge-neutral'
                )}
              >
                <Icon className="w-4 h-4" />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
