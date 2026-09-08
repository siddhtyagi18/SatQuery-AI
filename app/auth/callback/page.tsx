'use client';

import { useEffect, useState, useRef, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { supabase, HAS_SUPABASE } from '@/lib/supabase';
import { supabaseAnalysisService } from '@/lib/supabase/services';
import { useAuth } from '@/lib/authContext';
import { Satellite, Radio, ShieldAlert, CheckCircle2, ArrowRight } from 'lucide-react';

function AuthCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuth();

  const [status, setStatus] = useState<'processing' | 'success' | 'error'>('processing');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Guard against React 18 Strict Mode double-invocation & re-renders
  const hasExecutedRef = useRef(false);

  useEffect(() => {
    if (hasExecutedRef.current) return;
    hasExecutedRef.current = true;

    let active = true;

    async function handleAuthCallback() {
      // Check for error parameters in query string and URL hash
      const errorParam =
        searchParams.get('error') ||
        (typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('error') : null);
      const errorDescription =
        searchParams.get('error_description') ||
        (typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('error_description') : null);

      let hashError: string | null = null;
      let hashErrorDesc: string | null = null;
      if (typeof window !== 'undefined' && window.location.hash) {
        try {
          const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ''));
          hashError = hashParams.get('error');
          hashErrorDesc = hashParams.get('error_description');
        } catch {
          // ignore hash parse errors
        }
      }

      const finalError = errorDescription || errorParam || hashErrorDesc || hashError;
      if (finalError) {
        if (!active) return;
        setStatus('error');
        setErrorMessage(finalError);
        return;
      }

      if (!HAS_SUPABASE || !supabase) {
        // Fallback for mock/offline environment
        if (!active) return;
        login('google.operator@isro.gov.in', 'Google Mission Specialist');
        setStatus('success');
        setTimeout(() => router.replace('/profile'), 600);
        return;
      }

      try {
        const code =
          searchParams.get('code') ||
          (typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('code') : null);

        let session = null;

        // 1. Check if an active session is already established
        const { data: initialData } = await supabase.auth.getSession();
        if (initialData?.session?.user) {
          session = initialData.session;
        } else if (code) {
          // 2. PKCE flow: Exchange authorization code for session
          const { data: exchangeData, error: exchangeError } =
            await supabase.auth.exchangeCodeForSession(code);
          if (exchangeError) {
            // Check if code was already exchanged by concurrent internal listener
            const { data: retryData } = await supabase.auth.getSession();
            if (retryData?.session?.user) {
              session = retryData.session;
            } else {
              throw exchangeError;
            }
          } else {
            session = exchangeData.session;
          }
        } else {
          // 3. Implicit flow: Check active session from URL hash or client state
          const { data: fallbackData } = await supabase.auth.getSession();
          session = fallbackData?.session ?? null;
        }

        if (session?.user) {
          if (active) {
            await provisionUser(session);
          }
          return;
        }

        // 4. If not yet resolved, listen for the incoming auth state change
        const { data: authListener } = supabase.auth.onAuthStateChange(
          async (event, currentSession) => {
            if (currentSession?.user && active) {
              authListener.subscription.unsubscribe();
              await provisionUser(currentSession);
            }
          }
        );

        // Safety timeout after 5 seconds if no session is captured
        setTimeout(() => {
          authListener.subscription.unsubscribe();
          if (active && status === 'processing') {
            setStatus('error');
            setErrorMessage('Could not establish a secure mission session from Google credentials.');
          }
        }, 5000);
      } catch (err) {
        if (!active) return;
        setStatus('error');
        setErrorMessage(
          err instanceof Error ? err.message : 'Failed to finalize Google authentication.'
        );
      }
    }

    async function provisionUser(session: {
      user: { id: string; email?: string; user_metadata?: Record<string, unknown> };
    }) {
      const email = session.user.email ?? 'operator@isro.gov.in';
      const fullName =
        (session.user.user_metadata?.['name'] as string | undefined) ||
        (session.user.user_metadata?.['full_name'] as string | undefined) ||
        email.split('@')[0] ||
        'Google Operator';
      const avatarUrl =
        (session.user.user_metadata?.['avatar_url'] as string | undefined) ||
        (session.user.user_metadata?.['picture'] as string | undefined);

      try {
        await supabaseAnalysisService.ensureProfileForUser(session.user.id, email, fullName);
      } catch (profileErr) {
        console.warn('[auth-callback] Profile creation note:', profileErr);
      }

      login(email, fullName, avatarUrl);

      if (!active) return;
      setStatus('success');
      setTimeout(() => {
        router.replace('/profile');
      }, 600);
    }

    handleAuthCallback();

    return () => {
      active = false;
    };
  }, [searchParams, router, login]);

  return (
    <div className="min-h-screen w-full bg-[var(--surface-0)] flex items-center justify-center p-6 text-[var(--text-primary)]">
      <div
        className="w-full max-w-md rounded-2xl border border-[var(--border-hairline)] p-8 shadow-2xl backdrop-blur-md relative overflow-hidden"
        style={{
          background: 'linear-gradient(145deg, var(--surface-1), var(--surface-0))',
        }}
      >
        {/* Glow Header */}
        <div
          className="absolute -top-24 -left-24 w-48 h-48 rounded-full pointer-events-none blur-[70px] opacity-25"
          style={{ background: 'var(--cyan)' }}
          aria-hidden
        />

        <div className="relative z-10 flex flex-col items-center text-center gap-5">
          {/* Logo */}
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center text-[#05070D] font-bold shadow-lg"
            style={{
              background: 'linear-gradient(135deg, #3ED0FF 0%, #0A8ECC 100%)',
              boxShadow: '0 0 24px -4px rgba(62,208,255,0.5)',
            }}
          >
            <Satellite className="w-7 h-7" strokeWidth={2.2} />
          </div>

          {status === 'processing' && (
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center gap-2 text-xs font-mono text-[var(--cyan)] px-3 py-1 rounded-full border border-[var(--cyan)]/30 bg-[var(--cyan)]/10">
                <Radio className="w-3.5 h-3.5 animate-spin" />
                <span>EXCHANGING CLEARANCE TOKEN</span>
              </div>
              <h2 className="text-xl font-bold font-heading">Verifying Google Identity…</h2>
              <p className="text-xs text-[var(--text-muted)] max-w-xs leading-relaxed">
                Establishing encrypted session telemetry with ISRO Ground Station services.
              </p>
            </div>
          )}

          {status === 'success' && (
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-[var(--green)]/15 border border-[var(--green)]/40 flex items-center justify-center text-[var(--green)]">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <h2 className="text-xl font-bold font-heading text-[var(--green)]">
                Clearance Verified
              </h2>
              <p className="text-xs text-[var(--text-muted)] max-w-xs">
                Google operator account authenticated successfully. Redirecting to mission console…
              </p>
            </div>
          )}

          {status === 'error' && (
            <div className="flex flex-col items-center gap-4 w-full">
              <div className="w-10 h-10 rounded-full bg-[var(--red)]/15 border border-[var(--red)]/40 flex items-center justify-center text-[var(--red)]">
                <ShieldAlert className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-lg font-bold font-heading text-[var(--red)]">
                  Authentication Failed
                </h2>
                <p className="text-xs text-[var(--text-muted)] mt-1 break-words max-w-xs font-mono">
                  {errorMessage || 'An error occurred during clearance verification.'}
                </p>
              </div>

              <div className="flex flex-col gap-2 w-full mt-2">
                <Link
                  href="/login"
                  className="btn-primary w-full flex items-center justify-center gap-2 text-xs py-2.5"
                >
                  <span>Return to Clearance Login</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>

                <button
                  type="button"
                  onClick={() => {
                    login('google.operator@isro.gov.in', 'Google Mission Specialist');
                    router.replace('/profile');
                  }}
                  className="w-full py-2 text-xs font-mono text-[var(--cyan)] hover:underline cursor-pointer"
                >
                  Continue with Demo Google Account →
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen w-full bg-[var(--surface-0)] flex items-center justify-center text-xs font-mono text-[var(--text-muted)]">
          <Radio className="w-4 h-4 animate-spin mr-2 text-[var(--cyan)]" />
          INITIALIZING MISSION TELEMETRY…
        </div>
      }
    >
      <AuthCallbackContent />
    </Suspense>
  );
}
