'use client';

import { usePathname } from 'next/navigation';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { AnimatedBackground } from '@/components/background/AnimatedBackground';

const AUTH_ROUTES = ['/login', '/signup', '/forgot-password'];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthRoute = AUTH_ROUTES.some((r) => pathname === r || pathname.startsWith(r + '/'));

  if (isAuthRoute) {
    return <>{children}</>;
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
