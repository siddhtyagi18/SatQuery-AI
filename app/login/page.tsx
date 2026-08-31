'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/authContext';
import {
  Satellite,
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  Radio,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 30);
    return () => clearTimeout(t);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error('Please enter both email and password');
      return;
    }
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 900));
    login(email);
    setSubmitting(false);
    toast.success('Authentication successful. Redirecting to mission control…');
    setTimeout(() => router.push('/'), 400);
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[var(--surface-0)]">
      {/* ============================================================
         LAYER 1: Deep-space starfield (CSS dots, 2 parallax layers)
         ============================================================ */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
        <div
          className="absolute inset-[-20%] opacity-90"
          style={{
            backgroundImage:
              'radial-gradient(1px 1px at 5% 12%, rgba(62,208,255,0.9), transparent 60%),' +
              'radial-gradient(1.2px 1.2px at 15% 42%, rgba(255,255,255,0.9), transparent 60%),' +
              'radial-gradient(0.8px 0.8px at 22% 78%, rgba(192,132,252,0.8), transparent 60%),' +
              'radial-gradient(1px 1px at 32% 18%, rgba(255,176,32,0.85), transparent 60%),' +
              'radial-gradient(1.1px 1.1px at 42% 62%, rgba(255,255,255,0.9), transparent 60%),' +
              'radial-gradient(0.9px 0.9px at 55% 28%, rgba(62,208,255,0.85), transparent 60%),' +
              'radial-gradient(1px 1px at 65% 82%, rgba(192,132,252,0.9), transparent 60%),' +
              'radial-gradient(0.8px 0.8px at 75% 48%, rgba(61,220,132,0.75), transparent 60%),' +
              'radial-gradient(1.3px 1.3px at 85% 12%, rgba(255,255,255,0.95), transparent 60%),' +
              'radial-gradient(1px 1px at 92% 68%, rgba(62,208,255,0.85), transparent 60%),' +
              'radial-gradient(0.9px 0.9px at 8% 88%, rgba(255,176,32,0.8), transparent 60%),' +
              'radial-gradient(1px 1px at 48% 92%, rgba(255,255,255,0.8), transparent 60%),' +
              'radial-gradient(0.7px 0.7px at 30% 5%, rgba(62,208,255,0.7), transparent 60%),' +
              'radial-gradient(1px 1px at 70% 35%, rgba(192,132,252,0.7), transparent 60%),' +
              'radial-gradient(0.8px 0.8px at 90% 40%, rgba(255,255,255,0.7), transparent 60%),' +
              'radial-gradient(1.2px 1.2px at 3% 55%, rgba(255,255,255,0.85), transparent 60%)',
            animation: 'starfield-drift 55s ease-in-out infinite alternate',
          }}
        />
        <div
          className="absolute inset-[-20%] opacity-60"
          style={{
            backgroundImage:
              'radial-gradient(1px 1px at 18% 32%, rgba(62,208,255,0.7), transparent 60%),' +
              'radial-gradient(0.8px 0.8px at 48% 12%, rgba(255,255,255,0.8), transparent 60%),' +
              'radial-gradient(1px 1px at 78% 58%, rgba(192,132,252,0.7), transparent 60%),' +
              'radial-gradient(0.9px 0.9px at 12% 68%, rgba(255,176,32,0.7), transparent 60%),' +
              'radial-gradient(1px 1px at 82% 22%, rgba(255,255,255,0.85), transparent 60%),' +
              'radial-gradient(0.7px 0.7px at 58% 88%, rgba(62,208,255,0.6), transparent 60%),' +
              'radial-gradient(1px 1px at 38% 48%, rgba(61,220,132,0.65), transparent 60%)',
            animation: 'starfield-drift 80s ease-in-out infinite alternate-reverse',
          }}
        />
      </div>

      {/* ============================================================
         LAYER 2: Nebula ambient pulses (cyan / magenta / amber)
         ============================================================ */}
      <div
        className="absolute rounded-[50%] pointer-events-none blur-[60px]"
        style={{
          width: 520,
          height: 520,
          top: -120,
          right: -80,
          background: 'radial-gradient(circle, var(--cyan) 0%, transparent 70%)',
          opacity: 0.14,
          animation: 'nebula-pulse 10s ease-in-out infinite',
        }}
        aria-hidden
      />
      <div
        className="absolute rounded-[50%] pointer-events-none blur-[60px]"
        style={{
          width: 420,
          height: 420,
          bottom: -100,
          left: -60,
          background: 'radial-gradient(circle, var(--magenta) 0%, transparent 70%)',
          opacity: 0.12,
          animation: 'nebula-pulse 12s ease-in-out infinite',
          animationDelay: '-4s',
        }}
        aria-hidden
      />
      <div
        className="absolute rounded-[50%] pointer-events-none blur-[50px]"
        style={{
          width: 320,
          height: 320,
          top: '38%',
          left: '45%',
          background: 'radial-gradient(circle, var(--amber) 0%, transparent 70%)',
          opacity: 0.08,
          animation: 'nebula-pulse 14s ease-in-out infinite',
          animationDelay: '-7s',
        }}
        aria-hidden
      />

      {/* ============================================================
         LAYER 3: Shooting stars (streaks with comet tails)
         ============================================================ */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
        <ShootingStar top="8%" left="-5%" delay="0s" duration="8s" />
        <ShootingStar top="22%" left="-10%" delay="3.5s" duration="9s" />
        <ShootingStar top="55%" left="0%" delay="6.5s" duration="7.5s" />
        <ShootingStar top="78%" left="-8%" delay="11s" duration="8.5s" />
        <ShootingStar top="35%" left="-3%" delay="15s" duration="7s" />
      </div>

      {/* ============================================================
         LAYER 4: Central orbital Earth + satellites (right side, hero)
         ============================================================ */}
      <div
        className="hidden lg:block absolute pointer-events-none"
        style={{ top: '50%', right: '8%', transform: 'translateY(-50%)' }}
        aria-hidden
      >
        <OrbitalScene />
      </div>

      {/* ============================================================
         LAYER 5: Grid overlay (mission control HUD feel)
         ============================================================ */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.035]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(62,208,255,0.5) 1px, transparent 1px),' +
            'linear-gradient(90deg, rgba(62,208,255,0.5) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
        aria-hidden
      />

      {/* ============================================================
         CONTENT: Login card (left-center on desktop, centered on mobile)
         ============================================================ */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-4 py-10 lg:justify-start lg:px-[8vw] xl:px-[10vw]">
        <div
          className={cn(
            'w-full max-w-md transition-all duration-700 ease-out',
            mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
          )}
        >
          {/* Brand / header */}
          <div
            className={cn(
              'mb-8 transition-all duration-700 ease-out delay-150',
              mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            )}
          >
            <Link href="/" className="inline-flex items-center gap-3 mb-6 group">
              <div className="relative">
                <div
                  className="absolute inset-0 rounded-lg opacity-70 blur-md group-hover:opacity-100 transition-opacity"
                  style={{ background: 'var(--cyan)' }}
                />
                <div
                  className="relative w-11 h-11 rounded-lg bg-gradient-to-br from-[#3ED0FF] to-[#0A8ECC] flex items-center justify-center text-[#05070D]"
                  style={{ boxShadow: '0 0 0 1px rgba(62,208,255,0.45), 0 0 28px -6px rgba(62,208,255,0.55)' }}
                >
                  <Satellite className="w-5.5 h-5.5" strokeWidth={2.1} />
                </div>
              </div>
              <div className="flex flex-col">
                <span className="font-heading font-bold text-lg tracking-wider">
                  SatQuery
                  <span style={{ color: 'var(--cyan)' }}>-AI</span>
                </span>
                <span
                  className="font-mono text-[0.65rem] tracking-[0.18em] uppercase"
                  style={{ color: 'var(--text-faint)' }}
                >
                  ISRO Ground Station · Secure Access
                </span>
              </div>
            </Link>

            <h1 className="text-display-lg mb-3" style={{ lineHeight: 1.1 }}>
              <span
                className="bg-clip-text text-transparent"
                style={{
                  backgroundImage:
                    'linear-gradient(135deg, var(--text-primary) 0%, var(--cyan) 50%, var(--magenta) 100%)',
                  animation: 'shimmer 8s ease-in-out infinite',
                  backgroundSize: '200% 200%',
                }}
              >
                Welcome back,
              </span>
              <br />
              Mission Controller.
            </h1>
            <p className="text-sm leading-relaxed max-w-sm" style={{ color: 'var(--text-muted)' }}>
              Authenticate to access the multimodal satellite intelligence console.
              All ground-station telemetry is encrypted in transit & at rest.
            </p>
          </div>

          {/* ============================================================
             LOGIN CARD with HUD corner brackets
             ============================================================ */}
          <div
            className={cn(
              'relative transition-all duration-700 ease-out delay-300',
              mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            )}
          >
            {/* HUD Corner frame (CSS) */}
            <HudCorners domain="cyan" />
            <div className="absolute top-0 left-5 -translate-y-1/2 z-10 flex items-center gap-2">
              <span
                className="px-2 font-mono text-[0.62rem] font-semibold tracking-[0.16em] uppercase"
                style={{
                  color: 'var(--cyan)',
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-hairline)',
                }}
              >
                AUTH · GATEWAY-01
              </span>
              <button
                type="button"
                onClick={() => {
                  setEmail('controller@isro.gov.in');
                  setPassword('ISRO-SatQuery-2026');
                  toast.info('Loaded demo mission controller credentials');
                }}
                className="px-2 py-0.5 rounded font-mono text-[0.58rem] font-semibold tracking-wider uppercase transition-all bg-[var(--cyan)]/15 border border-[var(--cyan)]/40 text-[var(--cyan)] hover:bg-[var(--cyan)]/25 cursor-pointer shadow-sm"
              >
                Auto-Fill Demo Credentials
              </button>
            </div>

            <div
              className="relative p-7 sm:p-8 rounded-md overflow-hidden"
              style={{
                background:
                  'linear-gradient(145deg, color-mix(in srgb, var(--surface-1) 92%, transparent) 0%, color-mix(in srgb, var(--surface-2) 88%, transparent) 100%)',
                border: '1px solid var(--border-hairline)',
                backdropFilter: 'blur(18px)',
                WebkitBackdropFilter: 'blur(18px)',
                boxShadow:
                  '0 0 0 1px rgba(62,208,255,0.08), 0 30px 80px -30px rgba(0,0,0,0.8), 0 0 80px -40px rgba(62,208,255,0.25)',
              }}
            >
              {/* Inner animated scan sweep */}
              <div
                className="absolute inset-0 pointer-events-none overflow-hidden"
                aria-hidden
                style={{ borderRadius: 'inherit' }}
              >
                <div
                  className="absolute left-0 right-0 h-px"
                  style={{
                    top: 0,
                    background:
                      'linear-gradient(90deg, transparent, color-mix(in srgb, var(--cyan) 60%, transparent), transparent)',
                    animation: 'scan-sweep 6s ease-in-out infinite',
                    opacity: 0.5,
                  }}
                />
              </div>

              <form onSubmit={handleSubmit} className="flex flex-col gap-5 relative z-10">
                {/* Email field */}
                <div
                  className={cn(
                    'transition-all duration-500 ease-out delay-[420ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <label className="hud-label mb-2 block">Ground ID · Email</label>
                  <div className="group relative">
                    <Mail
                      className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors pointer-events-none"
                      style={{ color: 'var(--text-faint)' }}
                    />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="controller@isro.gov.in"
                      className="w-full pl-11 pr-4 py-3 rounded-md text-sm font-medium transition-all outline-none"
                      style={{
                        background: 'var(--surface-0)',
                        color: 'var(--text-primary)',
                        border: '1px solid var(--border-hairline)',
                        fontFamily: 'var(--font-mono)',
                        letterSpacing: '0.01em',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'var(--cyan)';
                        e.currentTarget.style.boxShadow =
                          '0 0 0 3px color-mix(in srgb, var(--cyan) 15%, transparent), 0 0 24px -8px color-mix(in srgb, var(--cyan) 50%, transparent)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-hairline)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>

                {/* Password field */}
                <div
                  className={cn(
                    'transition-all duration-500 ease-out delay-[520ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <div className="flex items-center justify-between mb-2">
                    <label className="hud-label">Clearance · Password</label>
                    <button
                      type="button"
                      onClick={() => toast.info('Password recovery: Demo mode active. Use the Auto-Fill Demo Credentials button above.')}
                      className="font-mono text-[0.65rem] font-semibold tracking-wide transition-colors hover:underline underline-offset-2 cursor-pointer"
                      style={{ color: 'var(--cyan)' }}
                    >
                      FORGOT?
                    </button>
                  </div>
                  <div className="group relative">
                    <Lock
                      className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors pointer-events-none"
                      style={{ color: 'var(--text-faint)' }}
                    />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••••••"
                      className="w-full pl-11 pr-12 py-3 rounded-md text-sm font-medium transition-all outline-none"
                      style={{
                        background: 'var(--surface-0)',
                        color: 'var(--text-primary)',
                        border: '1px solid var(--border-hairline)',
                        fontFamily: 'var(--font-mono)',
                        letterSpacing: '0.08em',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'var(--cyan)';
                        e.currentTarget.style.boxShadow =
                          '0 0 0 3px color-mix(in srgb, var(--cyan) 15%, transparent), 0 0 24px -8px color-mix(in srgb, var(--cyan) 50%, transparent)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-hairline)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((s) => !s)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded transition-colors"
                      style={{ color: 'var(--text-faint)' }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--cyan)')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-faint)')}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? (
                        <EyeOff className="w-4 h-4" strokeWidth={1.8} />
                      ) : (
                        <Eye className="w-4 h-4" strokeWidth={1.8} />
                      )}
                    </button>
                  </div>
                </div>

                {/* Remember me + status */}
                <div
                  className={cn(
                    'flex items-center justify-between transition-all duration-500 ease-out delay-[620ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <label className="inline-flex items-center gap-2.5 cursor-pointer select-none group">
                    <span className="relative inline-flex items-center justify-center">
                      <input
                        type="checkbox"
                        checked={remember}
                        onChange={(e) => setRemember(e.target.checked)}
                        className="sr-only peer"
                      />
                      <span
                        className="w-4 h-4 rounded border transition-all flex items-center justify-center"
                        style={{
                          background: remember
                            ? 'color-mix(in srgb, var(--cyan) 22%, transparent)'
                            : 'var(--surface-0)',
                          borderColor: remember ? 'var(--cyan)' : 'var(--border-hairline)',
                          boxShadow: remember
                            ? '0 0 12px -2px color-mix(in srgb, var(--cyan) 60%, transparent)'
                            : 'none',
                        }}
                      >
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 10 10"
                          fill="none"
                          style={{
                            opacity: remember ? 1 : 0,
                            transform: remember ? 'scale(1)' : 'scale(0.4)',
                            transition: 'all 160ms ease-out',
                          }}
                        >
                          <path
                            d="M1.5 5L4 7.5L8.5 2.5"
                            stroke="var(--cyan)"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </span>
                    </span>
                    <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                      Keep this session open
                    </span>
                  </label>

                  <span className="inline-flex items-center gap-1.5 font-mono text-[0.6rem] uppercase tracking-widest"
                    style={{ color: 'var(--green)' }}
                  >
                    <span className="relative flex h-1.5 w-1.5">
                      <span
                        className="absolute inline-flex h-full w-full rounded-full bg-current opacity-60"
                        style={{ animation: 'pulse-dot 1.6s ease-in-out infinite' }}
                      />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
                    </span>
                    Secure · TLS 1.3
                  </span>
                </div>

                {/* Submit CTA */}
                <div
                  className={cn(
                    'transition-all duration-500 ease-out delay-[720ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <button
                    type="submit"
                    disabled={submitting}
                    className="btn-primary w-full relative overflow-hidden group"
                    style={{
                      padding: '14px 24px',
                      fontSize: '0.78rem',
                      letterSpacing: '0.14em',
                      background: submitting
                        ? 'var(--surface-2)'
                        : 'linear-gradient(135deg, var(--cyan) 0%, color-mix(in srgb, var(--cyan) 50%, var(--magenta)) 100%)',
                      animation: submitting ? 'none' : 'glow-breathe 3.5s ease-in-out infinite',
                    }}
                  >
                    <span className="relative z-10 inline-flex items-center justify-center gap-2">
                      {submitting ? (
                        <>
                          <Radio
                            className="w-4 h-4"
                            style={{ animation: 'satellite-drift 1.2s ease-in-out infinite' }}
                          />
                          AUTHENTICATING…
                        </>
                      ) : (
                        <>
                          ENTER MISSION CONTROL
                          <ArrowRight className="w-4 h-4 arrow-slide" />
                        </>
                      )}
                    </span>
                    {/* Hover shimmer */}
                    <span
                      className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity"
                      style={{
                        background:
                          'linear-gradient(115deg, transparent 30%, color-mix(in srgb, white 28%, transparent) 50%, transparent 70%)',
                        animation: 'shimmer 1.8s linear infinite',
                        backgroundSize: '200% 100%',
                      }}
                    />
                  </button>
                </div>

                {/* Divider */}
                <div
                  className={cn(
                    'relative flex items-center gap-3 transition-all duration-500 ease-out delay-[820ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <div className="h-px flex-1" style={{ background: 'var(--border-hairline)' }} />
                  <span className="font-mono text-[0.6rem] uppercase tracking-[0.2em]"
                    style={{ color: 'var(--text-faint)' }}
                  >
                    Or continue with
                  </span>
                  <div className="h-px flex-1" style={{ background: 'var(--border-hairline)' }} />
                </div>

                {/* OAuth buttons */}
                <div
                  className={cn(
                    'grid grid-cols-2 gap-3 transition-all duration-500 ease-out delay-[920ms]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                  )}
                >
                  <OAuthButton
                    icon={<GoogleGlyph />}
                    label="Google"
                    domain="cyan"
                    onLogin={() => {
                      login('google.operator@isro.gov.in');
                      toast.success('Authenticated via Google ISRO SSO. Entering mission control…');
                      setTimeout(() => router.push('/'), 400);
                    }}
                  />
                  <OAuthButton
                    icon={<GithubOctocat />}
                    label="GitHub"
                    domain="neutral"
                    onLogin={() => {
                      login('github.specialist@isro.gov.in');
                      toast.success('Authenticated via GitHub Enterprise. Entering mission control…');
                      setTimeout(() => router.push('/'), 400);
                    }}
                  />
                </div>
              </form>
            </div>

            {/* Footer sign-up prompt */}
            <p
              className={cn(
                'mt-6 text-center text-sm transition-all duration-500 ease-out delay-[1050ms]',
                mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
              )}
              style={{ color: 'var(--text-muted)' }}
            >
              New operator?{' '}
              <button
                type="button"
                onClick={() => {
                  setEmail('controller@isro.gov.in');
                  setPassword('ISRO-SatQuery-2026');
                  toast.info('Instant demo access configured. Click Enter Mission Control.');
                }}
                className="font-semibold transition-colors hover:underline underline-offset-2 cursor-pointer"
                style={{ color: 'var(--cyan)' }}
              >
                Request ground-station access →
              </button>
            </p>
          </div>
        </div>
      </div>

      {/* Bottom telemetry strip */}
      <div
        className={cn(
          'absolute bottom-4 left-1/2 -translate-x-1/2 z-10 px-4 py-2 rounded font-mono text-[0.62rem] tracking-[0.16em] uppercase transition-all duration-700 ease-out delay-[1200ms]',
          mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
        )}
        style={{
          color: 'var(--text-faint)',
          background: 'color-mix(in srgb, var(--surface-1) 70%, transparent)',
          border: '1px solid var(--border-subtle)',
          backdropFilter: 'blur(6px)',
        }}
      >
        <span className="inline-flex items-center gap-3">
          <span>NODE · ISRO-GEO-VQA-01</span>
          <span className="w-px h-3" style={{ background: 'var(--border-hairline)' }} />
          <span>UPLINK · 2.2 GHz</span>
          <span className="w-px h-3" style={{ background: 'var(--border-hairline)' }} />
          <span>LATENCY · 42 ms</span>
          <span className="w-px h-3" style={{ background: 'var(--border-hairline)' }} />
          <span className="inline-flex items-center gap-1.5" style={{ color: 'var(--green)' }}>
            <span className="relative flex h-1.5 w-1.5">
              <span
                className="absolute inline-flex h-full w-full rounded-full bg-current opacity-60"
                style={{ animation: 'pulse-dot 1.6s ease-in-out infinite' }}
              />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
            </span>
            LINK NOMINAL
          </span>
        </span>
      </div>
    </div>
  );
}

/* ==================================================================
   Sub-components
   ================================================================== */

function ShootingStar({
  top,
  left,
  delay,
  duration,
}: {
  top: string;
  left: string;
  delay: string;
  duration: string;
}) {
  return (
    <div
      className="absolute"
      style={{
        top,
        left,
        width: 3,
        height: 3,
        borderRadius: '50%',
        background: '#fff',
        boxShadow: '0 0 6px 2px rgba(255,255,255,0.75)',
        animation: `shooting-star ${duration} ease-in ${delay} infinite`,
        opacity: 0,
      }}
    >
      <div
        className="absolute"
        style={{
          top: '50%',
          right: 3,
          height: 1.5,
          transform: 'translateY(-50%)',
          background: 'linear-gradient(90deg, rgba(255,255,255,0.9), transparent)',
          animation: `comet-tail ${duration} ease-in ${delay} infinite`,
          width: 0,
          opacity: 0,
        }}
      />
    </div>
  );
}

function OrbitalScene() {
  return (
    <div className="relative" style={{ width: 440, height: 440 }}>
      {/* Outermost rotating radar ring */}
      <div
        className="absolute inset-0 rounded-full"
        style={{
          border: '1px dashed rgba(62,208,255,0.18)',
          animation: 'orbit 40s linear infinite',
        }}
      />
      {/* Second orbit ring */}
      <div
        className="absolute rounded-full"
        style={{
          inset: 48,
          border: '1px solid rgba(192,132,252,0.14)',
        }}
      />
      {/* Inner orbit ring */}
      <div
        className="absolute rounded-full"
        style={{
          inset: 96,
          border: '1px solid rgba(255,176,32,0.12)',
          animation: 'orbit-reverse 30s linear infinite',
        }}
      />
      {/* Radar sweep arm */}
      <div
        className="absolute inset-0 rounded-full overflow-hidden"
        style={{ opacity: 0.55 }}
      >
        <div
          className="absolute top-1/2 left-1/2 origin-left"
          style={{
            width: '50%',
            height: 1,
            background:
              'linear-gradient(90deg, rgba(62,208,255,0.7), rgba(62,208,255,0) 85%)',
            transform: 'translateY(-50%)',
            transformOrigin: 'left center',
            animation: 'radar-sweep 5s linear infinite',
            boxShadow: '0 0 18px 2px rgba(62,208,255,0.35)',
          }}
        />
        <div
          className="absolute top-1/2 left-1/2 rounded-full"
          style={{
            width: '50%',
            height: '50%',
            transform: 'translate(-50%, -50%)',
            background:
              'conic-gradient(from 0deg, rgba(62,208,255,0.12) 0deg, rgba(62,208,255,0) 55deg)',
            animation: 'radar-sweep 5s linear infinite',
            filter: 'blur(2px)',
          }}
        />
      </div>

      {/* Central Earth */}
      <div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          width: 160,
          height: 160,
          background:
            'radial-gradient(circle at 30% 30%, #1e4a6f 0%, #0a2540 40%, #051525 80%, #030a14 100%)',
          boxShadow:
            '0 0 0 1px rgba(62,208,255,0.25), inset -20px -20px 60px rgba(0,0,0,0.7), inset 10px 10px 40px rgba(62,208,255,0.25), 0 0 80px -10px rgba(62,208,255,0.45)',
          overflow: 'hidden',
        }}
      >
        {/* Continents */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            backgroundImage:
              'radial-gradient(ellipse 30px 14px at 35% 40%, rgba(61,220,132,0.28), transparent 65%),' +
              'radial-gradient(ellipse 22px 10px at 58% 55%, rgba(61,220,132,0.22), transparent 65%),' +
              'radial-gradient(ellipse 18px 24px at 68% 30%, rgba(61,220,132,0.25), transparent 65%),' +
              'radial-gradient(ellipse 26px 14px at 22% 62%, rgba(61,220,132,0.2), transparent 65%),' +
              'radial-gradient(ellipse 14px 18px at 48% 72%, rgba(61,220,132,0.22), transparent 65%)',
            filter: 'blur(1px)',
          }}
        />
        {/* Clouds */}
        <div
          className="absolute inset-0 rounded-full opacity-50"
          style={{
            backgroundImage:
              'radial-gradient(ellipse 50px 10px at 28% 30%, rgba(255,255,255,0.12), transparent 70%),' +
              'radial-gradient(ellipse 40px 8px at 65% 50%, rgba(255,255,255,0.1), transparent 70%),' +
              'radial-gradient(ellipse 30px 7px at 40% 78%, rgba(255,255,255,0.08), transparent 70%)',
            animation: 'starfield-drift 25s ease-in-out infinite alternate',
          }}
        />
        {/* Atmosphere halo */}
        <div
          className="absolute -inset-1 rounded-full pointer-events-none"
          style={{
            background:
              'radial-gradient(circle, transparent 58%, rgba(62,208,255,0.18) 68%, transparent 80%)',
            filter: 'blur(4px)',
          }}
        />
      </div>

      {/* Orbiting satellites — outer */}
      <div
        className="absolute left-1/2 top-1/2"
        style={{ width: 1, height: 1 }}
      >
        <SatelliteDot radius={200} duration={14} color="cyan" delay={-2} drift />
      </div>
      <div
        className="absolute left-1/2 top-1/2"
        style={{ width: 1, height: 1 }}
      >
        <SatelliteDot radius={200} duration={14} color="magenta" delay={-8} />
      </div>
      {/* Mid orbit */}
      <div
        className="absolute left-1/2 top-1/2"
        style={{ width: 1, height: 1 }}
      >
        <SatelliteDot radius={148} duration={10} color="amber" delay={-4} />
      </div>
      {/* Inner orbit (reverse) */}
      <div
        className="absolute left-1/2 top-1/2"
        style={{ width: 1, height: 1 }}
      >
        <SatelliteDot radius={96} duration={8} color="cyan" reverse delay={-6} />
      </div>

      {/* Radar pulse from center */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 rounded-full pointer-events-none">
        <div
          className="absolute inset-0 rounded-full"
          style={{
            border: '1px solid var(--cyan)',
            animation: 'radar-pulse 2.8s ease-out infinite',
          }}
        />
        <div
          className="absolute inset-0 rounded-full"
          style={{
            border: '1px solid var(--cyan)',
            animation: 'radar-pulse 2.8s ease-out infinite',
            animationDelay: '-1.4s',
          }}
        />
      </div>
    </div>
  );
}

function SatelliteDot({
  radius,
  duration,
  color,
  reverse,
  delay,
  drift,
}: {
  radius: number;
  duration: number;
  color: 'cyan' | 'magenta' | 'amber';
  reverse?: boolean;
  delay?: number;
  drift?: boolean;
}) {
  const vars: Record<string, string> = {
    cyan: 'var(--cyan)',
    magenta: 'var(--magenta)',
    amber: 'var(--amber)',
  };
  const c = vars[color];
  return (
    <div
      style={{
        position: 'absolute',
        width: 1,
        height: 1,
        animation: `${reverse ? 'orbit-reverse' : 'orbit'} ${duration}s linear infinite`,
        animationDelay: `${delay ?? 0}s`,
        transformOrigin: 'center center',
        ['--orbit-radius' as never]: `${radius}px`,
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: 6,
          height: 6,
          margin: '-3px 0 0 -3px',
          transform: `translateX(${radius}px)`,
          borderRadius: '50%',
          background: c,
          boxShadow: `0 0 10px ${c}, 0 0 22px color-mix(in srgb, ${c} 55%, transparent)`,
          animation: drift ? `satellite-drift 3.5s ease-in-out infinite` : 'none',
        }}
      />
    </div>
  );
}

function HudCorners({ domain }: { domain: 'cyan' | 'magenta' | 'amber' }) {
  const color =
    domain === 'cyan'
      ? 'var(--cyan)'
      : domain === 'magenta'
      ? 'var(--magenta)'
      : 'var(--amber)';
  const bracket = {
    position: 'absolute' as const,
    width: 18,
    height: 18,
    borderStyle: 'solid' as const,
    borderWidth: 1.5,
    borderColor: color,
    opacity: 0.7,
    pointerEvents: 'none' as const,
    zIndex: 2,
  };
  return (
    <>
      <span style={{ ...bracket, top: -1, left: -1, borderRight: 0, borderBottom: 0 }} />
      <span style={{ ...bracket, top: -1, right: -1, borderLeft: 0, borderBottom: 0 }} />
      <span style={{ ...bracket, bottom: -1, left: -1, borderRight: 0, borderTop: 0 }} />
      <span style={{ ...bracket, bottom: -1, right: -1, borderLeft: 0, borderTop: 0 }} />
    </>
  );
}

function OAuthButton({
  icon,
  label,
  domain,
  onLogin,
}: {
  icon: React.ReactNode;
  label: string;
  domain: 'cyan' | 'neutral';
  onLogin?: () => void;
}) {
  const accent = domain === 'cyan' ? 'var(--cyan)' : 'var(--text-primary)';
  return (
    <button
      type="button"
      onClick={onLogin}
      className="group relative flex items-center justify-center gap-2.5 px-4 py-3 rounded-md text-xs font-semibold tracking-wider uppercase transition-all cursor-pointer"
      style={{
        background: 'var(--surface-0)',
        color: 'var(--text-muted)',
        border: '1px solid var(--border-hairline)',
        fontFamily: 'var(--font-mono)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--text-primary)';
        e.currentTarget.style.borderColor = accent;
        e.currentTarget.style.boxShadow = `0 0 0 3px color-mix(in srgb, ${accent} 10%, transparent), 0 0 22px -6px color-mix(in srgb, ${accent} 35%, transparent)`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--text-muted)';
        e.currentTarget.style.borderColor = 'var(--border-hairline)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <span className="w-4 h-4 flex items-center justify-center shrink-0">{icon}</span>
      {label}
    </button>
  );
}

function GoogleGlyph() {
  return (
    <svg viewBox="0 0 48 48" width="16" height="16" aria-hidden>
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.3-.4-3.5z" />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 16 19 13 24 13c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.5-5.2l-6.2-5.2C29.3 35.7 26.9 37 24 37c-5.4 0-9.8-3.5-11.3-8.1l-6.5 5C9.5 39.5 16.2 44 24 44z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.3-4.1 5.6l6.2 5.2c-.4.4 6.6-4.8 6.6-14.8 0-1.3-.1-2.3-.4-3.5z" />
    </svg>
  );
}

function GithubOctocat() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden>
      <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.73-4.04-1.61-4.04-1.61-.55-1.38-1.34-1.75-1.34-1.75-1.09-.74.08-.73.08-.73 1.21.08 1.85 1.24 1.85 1.24 1.07 1.83 2.81 1.3 3.5 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23.96-.27 1.98-.4 3-.4s2.04.13 3 .4c2.29-1.55 3.3-1.23 3.3-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.29 0 .32.22.7.82.58A12 12 0 0024 12.5C24 5.87 18.63.5 12 .5z" />
    </svg>
  );
}
