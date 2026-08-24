import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '@/components/ThemeProvider';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';
import { Toaster } from 'sonner';

export const metadata: Metadata = {
  title: 'SatQuery-AI — ISRO Multimodal Remote-Sensing Intelligence',
  description:
    'Ground station assistant for interactive vision-language remote-sensing image analysis, change detection, and cross-modal fusion.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased flex h-screen overflow-hidden bg-[var(--bg-base)] text-[var(--text-primary)]">
        <ThemeProvider>
          <div className="flex h-full w-full">
            {/* Collapsible Left Sidebar */}
            <Sidebar />

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
              <TopBar />
              <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
                {children}
              </main>
            </div>
          </div>
          <Toaster
            theme="dark"
            position="bottom-right"
            toastOptions={{
              style: {
                background: 'var(--bg-panel-elevated)',
                border: '1px solid var(--border-hairline)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
              },
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  );
}
