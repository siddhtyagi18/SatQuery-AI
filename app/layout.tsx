import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '@/components/ThemeProvider';
import { AuthProvider } from '@/lib/authContext';
import { AppShell } from '@/components/layout/AppShell';
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
      <body className="antialiased flex h-screen overflow-hidden bg-[var(--surface-0)] text-[var(--text-primary)]">
        <ThemeProvider>
          <AuthProvider>
            <AppShell>
              {children}
            </AppShell>
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
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
