'use client';

import { usePathname } from 'next/navigation';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { AnimatedBackground } from '@/components/background/AnimatedBackground';
import { useAuth } from '@/lib/authContext';

const AUTH_ROUTES = ['/login', '/signup', '/forgot-password'];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { loading } = useAuth();
  const isAuthRoute = AUTH_ROUTES.some((r) => pathname === r || pathname.startsWith(r + '/'));

  // Always render auth pages (login, etc.) without the shell
  if (isAuthRoute) {
    return <>{children}</>;
  }

  // While we're restoring session, show a neutral blank screen — prevents
  // the sidebar/topbar flashing before the redirect to /login fires.
  if (loading) {
    return (
      <div
        className="fixed inset-0 flex items-center justify-center bg-[var(--surface-0)]"
        aria-label="Authenticating…"
      >
        <span
          className="font-mono text-xs tracking-widest uppercase animate-pulse"
          style={{ color: 'var(--text-faint)' }}
        >
          Authenticating…
        </span>
      </div>
    );
  }

  return (
    <div className="relative flex h-full w-full overflow-hidden bg-[var(--surface-0)]">
      <AnimatedBackground />
      <Sidebar />
      <div className="relative flex-1 flex flex-col min-w-0 h-full overflow-hidden z-10">
        <TopBar />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          {children}
        </main>
      </div>
    </div>
  );
}
