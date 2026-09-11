'use client';

import { useState } from 'react';
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
  Radar,
  Globe,
  Activity,
  User as UserIcon,
  Calendar,
  Sparkles,
  ShieldCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

export default function LoginPage() {
  const router = useRouter();
  const { login, signup, loginWithGoogle } = useAuth();

  // Mode: 'login' | 'signup'
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');

  // Form fields
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [dob, setDob] = useState('1996-01-01');
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // Google OAuth state
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email.trim() || !password.trim()) {
      toast.error('Please enter both email and password');
      return;
    }

    if (authMode === 'signup' && !name.trim()) {
      toast.error('Please enter your full name');
      return;
    }

    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 600));

    if (authMode === 'signup') {
      signup(email.trim(), name.trim(), dob);
      toast.success('Account created successfully. Welcome to SatQuery-AI!');
    } else {
      login(email.trim(), name.trim() || undefined);
      toast.success('Authentication successful. Welcome back!');
    }

    setSubmitting(false);
    // As requested: Login/Signup -> User Profile -> Dashboard
    setTimeout(() => router.push('/profile'), 400);
  };

  const handleFillDemo = () => {
    setEmail('controller@isro.gov.in');
    setPassword('ISRO-SatQuery-2026');
    setName('ISRO Ground Controller');
    toast.info('Loaded demo mission controller credentials');
  };

  const handleGoogleSignIn = async () => {
    setIsGoogleLoading(true);
    const toastId = toast.loading('Connecting to Google Mission Portal…');
    try {
      const res = await loginWithGoogle();
      if (res?.error) {
        toast.error(`Google Sign-In: ${res.error}`, { id: toastId });
        setIsGoogleLoading(false);
        return;
      }
      if (res?.url) {
        toast.success('Redirecting to Google…', { id: toastId });
        window.location.href = res.url;
      } else {
        toast.dismiss(toastId);
        setIsGoogleLoading(false);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Google OAuth failed', { id: toastId });
      setIsGoogleLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[var(--surface-0)] flex flex-col lg:grid lg:grid-cols-2 text-[var(--text-primary)]">
      {/* ============================================================
         LEFT COLUMN: Brand, Mission Showcase & Capabilities (Desktop)
         ============================================================ */}
      <div className="relative flex flex-col justify-between p-8 sm:p-12 lg:p-16 border-b lg:border-b-0 lg:border-r border-[var(--border-hairline)] bg-gradient-to-br from-[var(--surface-1)] to-[var(--surface-0)] overflow-hidden">
        {/* Subtle background grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.04] pointer-events-none"
          style={{
            backgroundImage:
              'linear-gradient(rgba(62,208,255,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(62,208,255,0.8) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
          aria-hidden
        />

        {/* Ambient glow in corner */}
        <div
          className="absolute -top-32 -left-32 w-96 h-96 rounded-full pointer-events-none blur-[100px] opacity-20"
          style={{ background: 'var(--cyan)' }}
          aria-hidden
        />

        <div className="relative z-10 flex flex-col gap-10">
          {/* Top Branding Header */}
          <Link href="/" className="inline-flex items-center gap-3.5 group self-start">
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-[#05070D] font-bold shadow-lg transition-transform group-hover:scale-105"
              style={{
                background: 'linear-gradient(135deg, #3ED0FF 0%, #0A8ECC 100%)',
                boxShadow: '0 0 24px -4px rgba(62,208,255,0.5)',
              }}
            >
              <Satellite className="w-6 h-6" strokeWidth={2.2} />
            </div>
            <div className="flex flex-col">
              <span className="font-heading font-bold text-xl tracking-tight">
                SatQuery<span style={{ color: 'var(--cyan)' }}>-AI</span>
              </span>
              <span className="font-mono text-[0.68rem] tracking-widest uppercase text-[var(--text-muted)]">
                ISRO Ground Station Console
              </span>
            </div>
          </Link>

          {/* Core Value Proposition */}
          <div className="flex flex-col gap-4 max-w-xl">
            <div className="inline-flex items-center gap-2 self-start px-3 py-1 rounded-full text-xs font-mono font-medium border border-[var(--cyan)]/30 bg-[var(--cyan)]/10 text-[var(--cyan)]">
              <Radio className="w-3 h-3 animate-pulse" />
              <span>MULTIMODAL REMOTE SENSING AI</span>
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight font-heading leading-[1.15]">
              Intelligent Satellite Imagery Query &amp; Analysis
            </h1>

            <p className="text-base text-[var(--text-muted)] leading-relaxed">
              Accelerate Earth observation analysis with natural language. Query high-resolution optical, SAR, and bi-temporal payloads with specialist vision-language models.
            </p>
          </div>

          {/* Animated Orbital Radar Telemetry Visual */}
          <div
            className="relative w-full rounded-2xl border border-[var(--border-hairline)] overflow-hidden p-5 flex flex-col justify-between shadow-2xl backdrop-blur-md"
            style={{
              background:
                'linear-gradient(145deg, color-mix(in srgb, var(--surface-1) 88%, transparent), color-mix(in srgb, var(--surface-0) 92%, transparent))',
              minHeight: '340px',
            }}
          >
            {/* Top HUD Header */}
            <div className="flex items-center justify-between z-10 text-[0.7rem] font-mono">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--cyan)] opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--cyan)]" />
                </span>
                <span className="text-[var(--cyan)] font-semibold tracking-wider flex items-center gap-1.5">
                  <Radar className="w-3.5 h-3.5 animate-spin" style={{ animationDuration: '6s' }} />
                  ORBITAL SWEEP // GEO-SYNC
                </span>
              </div>
              <div className="flex items-center gap-2 text-[var(--text-muted)]">
                <Activity className="w-3.5 h-3.5 text-[var(--green)]" />
                <span>REAL-TIME TRACKING</span>
              </div>
            </div>

            {/* Central Orbital & Radar Scene */}
            <div className="relative w-full h-56 sm:h-64 my-auto flex items-center justify-center overflow-hidden">
              {/* Starfield ambient dots */}
              <div className="starfield opacity-60" aria-hidden />

              {/* Radar Scope Outer Ring with Cardinal Ticks */}
              <div
                className="w-56 h-56 sm:w-64 sm:h-64 rounded-full border border-[var(--cyan)]/25 relative flex items-center justify-center"
                style={{
                  boxShadow: '0 0 30px -5px rgba(62,208,255,0.15)',
                }}
              >
                {/* Cardinal direction labels */}
                <span className="absolute -top-4 text-[0.6rem] font-mono text-[var(--text-faint)] tracking-widest">
                  000° N
                </span>
                <span className="absolute -bottom-4 text-[0.6rem] font-mono text-[var(--text-faint)] tracking-widest">
                  180° S
                </span>
                <span className="absolute -left-5 text-[0.6rem] font-mono text-[var(--text-faint)] tracking-widest">
                  270° W
                </span>
                <span className="absolute -right-5 text-[0.6rem] font-mono text-[var(--text-faint)] tracking-widest">
                  090° E
                </span>

                {/* Range Rings (Concentric Circles) */}
                <div className="w-44 h-44 rounded-full border border-[var(--border-hairline)] border-dashed absolute" />
                <div className="w-32 h-32 rounded-full border border-[var(--border-hairline)] absolute" />
                <div className="w-20 h-20 rounded-full border border-[var(--border-hairline)] border-dashed absolute" />

                {/* Crosshair Axes */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-full h-px border-b border-dashed border-[var(--border-hairline)]" />
                  <div className="h-full w-px border-r border-dashed border-[var(--border-hairline)] absolute" />
                </div>

                {/* 360-Degree Rotating Radar Sweep Beam */}
                <div
                  className="absolute inset-0 rounded-full pointer-events-none"
                  style={{
                    background:
                      'conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(62,208,255,0.03) 290deg, rgba(62,208,255,0.28) 360deg)',
                    animation: 'orbital-spin 4s linear infinite',
                  }}
                />
                {/* Radar Sweep Leading Line */}
                <div
                  className="absolute inset-0 pointer-events-none flex items-center justify-center"
                  style={{ animation: 'orbital-spin 4s linear infinite' }}
                >
                  <div className="w-1/2 h-[1.5px] origin-left bg-gradient-to-r from-transparent via-[var(--cyan)]/40 to-[var(--cyan)] shadow-[0_0_8px_#3ED0FF] ml-auto" />
                </div>

                {/* Orbit Track 1: CARTOSAT-3 (Optical) */}
                <div
                  className="w-48 h-48 rounded-full absolute pointer-events-none"
                  style={{ animation: 'orbital-spin 14s linear infinite' }}
                >
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-1.5">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--cyan)] opacity-80" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-[var(--cyan)] shadow-[0_0_10px_#3ED0FF]" />
                    </span>
                    <div
                      className="px-1.5 py-0.5 rounded bg-[var(--surface-2)]/90 border border-[var(--cyan)]/40 text-[0.58rem] font-mono text-[var(--cyan)] whitespace-nowrap shadow-sm backdrop-blur-sm"
                      style={{ animation: 'orbital-spin-reverse 14s linear infinite' }}
                    >
                      CARTOSAT-3 · 0.28m
                    </div>
                  </div>
                </div>

                {/* Orbit Track 2: RISAT-1A (SAR Radar) */}
                <div
                  className="w-36 h-36 rounded-full absolute pointer-events-none"
                  style={{ animation: 'orbital-spin-reverse 18s linear infinite' }}
                >
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-1.5">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--amber)] opacity-80" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--amber)] shadow-[0_0_8px_#FFB020]" />
                    </span>
                    <div
                      className="px-1.5 py-0.5 rounded bg-[var(--surface-2)]/90 border border-[var(--amber)]/40 text-[0.58rem] font-mono text-[var(--amber)] whitespace-nowrap shadow-sm backdrop-blur-sm"
                      style={{ animation: 'orbital-spin 18s linear infinite' }}
                    >
                      RISAT-1A · SAR
                    </div>
                  </div>
                </div>

                {/* Orbit Track 3: OCEANSAT-3 (VNIR/SWIR) */}
                <div
                  className="w-24 h-24 rounded-full absolute pointer-events-none"
                  style={{ animation: 'orbital-spin 10s linear infinite' }}
                >
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 flex items-center gap-1.5">
                    <span className="relative flex h-2 w-2">
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--magenta)] shadow-[0_0_8px_#C084FC]" />
                    </span>
                    <div
                      className="px-1.5 py-0.5 rounded bg-[var(--surface-2)]/90 border border-[var(--magenta)]/40 text-[0.58rem] font-mono text-[var(--magenta)] whitespace-nowrap shadow-sm backdrop-blur-sm"
                      style={{ animation: 'orbital-spin-reverse 10s linear infinite' }}
                    >
                      OCEANSAT-3
                    </div>
                  </div>
                </div>

                {/* Center ISRO Earth Ground Node */}
                <div className="relative z-10 flex flex-col items-center justify-center">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center border border-[var(--cyan)]/60 text-[var(--cyan)] relative"
                    style={{
                      background:
                        'radial-gradient(circle, rgba(62,208,255,0.25) 0%, var(--surface-1) 80%)',
                      boxShadow: '0 0 16px rgba(62,208,255,0.4)',
                    }}
                  >
                    <Globe className="w-5 h-5 animate-pulse text-[var(--cyan)]" />
                    <span className="absolute -inset-1.5 rounded-full border border-[var(--cyan)]/30 animate-ping pointer-events-none" />
                  </div>
                  <span className="mt-1 text-[0.55rem] font-mono uppercase tracking-wider text-[var(--text-faint)]">
                    ISRO-MCF
                  </span>
                </div>
              </div>
            </div>

            {/* Bottom HUD Telemetry Strip */}
            <div className="grid grid-cols-3 gap-2 pt-3 border-t border-[var(--border-hairline)] text-center">
              <div className="flex flex-col">
                <span className="text-[0.6rem] font-mono text-[var(--text-faint)] uppercase">
                  CONSTELLATION
                </span>
                <span className="text-[0.72rem] font-mono font-semibold text-[var(--green)] flex items-center justify-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--green)] animate-pulse" />
                  3 IN LOCK
                </span>
              </div>

              <div className="flex flex-col">
                <span className="text-[0.6rem] font-mono text-[var(--text-faint)] uppercase">
                  POLAR ORBIT
                </span>
                <span className="text-[0.72rem] font-mono font-semibold text-[var(--cyan)]">
                  ALT 506.4 KM
                </span>
              </div>

              <div className="flex flex-col">
                <span className="text-[0.6rem] font-mono text-[var(--text-faint)] uppercase">
                  SENSOR FUSION
                </span>
                <span className="text-[0.72rem] font-mono font-semibold text-[var(--amber)]">
                  OPTICAL + SAR
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Telemetry Status Strip */}
        <div className="relative z-10 pt-10 mt-auto flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-[var(--text-faint)]">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-[var(--green)]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
              </span>
              NODE: ISRO-GEO-01
            </span>
            <span>·</span>
            <span>DOWNLINK 2.2 GHz</span>
            <span>·</span>
            <span>TLS 1.3 ENCRYPTED</span>
          </div>

          <span className="text-[0.65rem] uppercase tracking-wider">
            SIH Mission Project
          </span>
        </div>
      </div>

      {/* ============================================================
         RIGHT COLUMN: Modern Split-Screen Authentication Form
         ============================================================ */}
      <div className="flex items-center justify-center p-6 sm:p-12 lg:p-16 relative">
        <div className="w-full max-w-md flex flex-col gap-6">
          {/* Card Container */}
          <div
            className="p-8 sm:p-10 rounded-2xl border border-[var(--border-hairline)] shadow-2xl relative"
            style={{
              background: 'var(--surface-1)',
              backdropFilter: 'blur(20px)',
            }}
          >
            {/* Quick Demo Pre-fill Banner */}
            <div className="flex items-center justify-between gap-3 mb-6 p-2.5 rounded-lg bg-[var(--surface-2)] border border-[var(--border-hairline)]">
              <div className="flex items-center gap-2 min-w-0">
                <Sparkles className="w-4 h-4 text-[var(--cyan)] flex-shrink-0" />
                <span className="text-xs text-[var(--text-muted)] truncate">
                  Testing the system?
                </span>
              </div>
              <button
                type="button"
                onClick={handleFillDemo}
                className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-[var(--cyan)]/15 border border-[var(--cyan)]/40 text-[var(--cyan)] hover:bg-[var(--cyan)]/25 transition-colors cursor-pointer flex-shrink-0"
              >
                Auto-Fill Demo
              </button>
            </div>

            {/* Auth Mode Toggle Tabs (Sign In / Sign Up) */}
            <div className="flex rounded-lg p-1 bg-[var(--surface-2)] border border-[var(--border-hairline)] mb-6">
              <button
                type="button"
                onClick={() => setAuthMode('login')}
                className={cn(
                  'flex-1 py-2 text-xs font-semibold rounded-md transition-all cursor-pointer text-center',
                  authMode === 'login'
                    ? 'bg-[var(--surface-1)] text-[var(--text-primary)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                )}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => setAuthMode('signup')}
                className={cn(
                  'flex-1 py-2 text-xs font-semibold rounded-md transition-all cursor-pointer text-center',
                  authMode === 'signup'
                    ? 'bg-[var(--surface-1)] text-[var(--text-primary)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                )}
              >
                Create Account
              </button>
            </div>

            {/* Form Title & Subtitle */}
            <div className="mb-6">
              <h2 className="text-xl font-bold font-heading text-[var(--text-primary)]">
                {authMode === 'login' ? 'Operator Clearance Sign In' : 'Create Operator Profile'}
              </h2>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                {authMode === 'login'
                  ? 'Enter your credentials to access the mission intelligence console.'
                  : 'Register a new ground-station operator profile.'}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              {/* Full Name field (Sign Up only) */}
              {authMode === 'signup' && (
                <div>
                  <label className="hud-label block mb-1.5">Operator Full Name</label>
                  <div className="relative">
                    <UserIcon className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g. Controller Sharma"
                      required
                      className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm outline-none transition-all"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'var(--cyan)';
                        e.currentTarget.style.boxShadow = '0 0 0 2px rgba(62,208,255,0.2)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-hairline)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Email field */}
              <div>
                <label className="hud-label block mb-1.5">Clearance Email / Ground ID</label>
                <div className="relative">
                  <Mail className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="controller@isro.gov.in"
                    required
                    className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm outline-none transition-all font-mono"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-hairline)',
                      color: 'var(--text-primary)',
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = 'var(--cyan)';
                      e.currentTarget.style.boxShadow = '0 0 0 2px rgba(62,208,255,0.2)';
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = 'var(--border-hairline)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  />
                </div>
              </div>

              {/* Date of Birth field (Sign Up only) */}
              {authMode === 'signup' && (
                <div>
                  <label className="hud-label block mb-1.5">Date of Birth</label>
                  <div className="relative">
                    <Calendar className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type="date"
                      value={dob}
                      onChange={(e) => setDob(e.target.value)}
                      required
                      className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm outline-none transition-all"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'var(--cyan)';
                        e.currentTarget.style.boxShadow = '0 0 0 2px rgba(62,208,255,0.2)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-hairline)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Password field */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="hud-label">Clearance Password</label>
                  {authMode === 'login' && (
                    <button
                      type="button"
                      onClick={() =>
                        toast.info('Demo Mode: Click "Auto-Fill Demo" above for instant credentials.')
                      }
                      className="text-[0.68rem] text-[var(--cyan)] hover:underline cursor-pointer font-mono"
                    >
                      Forgot?
                    </button>
                  )}
                </div>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    required
                    className="w-full pl-10 pr-10 py-2.5 rounded-lg text-sm outline-none transition-all font-mono"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-hairline)',
                      color: 'var(--text-primary)',
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = 'var(--cyan)';
                      e.currentTarget.style.boxShadow = '0 0 0 2px rgba(62,208,255,0.2)';
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = 'var(--border-hairline)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)] hover:text-[var(--text-primary)] transition-colors p-1"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Remember Me Checkbox */}
              {authMode === 'login' && (
                <div className="flex items-center justify-between pt-1">
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={remember}
                      onChange={(e) => setRemember(e.target.checked)}
                      className="rounded border-[var(--border-hairline)] bg-[var(--surface-2)] text-[var(--cyan)] focus:ring-0"
                    />
                    <span className="text-xs text-[var(--text-muted)]">
                      Keep this session open
                    </span>
                  </label>
                  <span className="text-[0.62rem] font-mono text-[var(--green)] flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" /> TLS 1.3
                  </span>
                </div>
              )}

              {/* Submit CTA Button */}
              <button
                type="submit"
                disabled={submitting}
                className="btn-primary w-full flex items-center justify-center gap-2 mt-2 shadow-lg cursor-pointer"
                style={{ padding: '12px 20px', fontSize: '0.84rem' }}
              >
                {submitting ? (
                  <>
                    <Radio className="w-4 h-4 animate-spin" />
                    <span>AUTHENTICATING OPERATOR…</span>
                  </>
                ) : authMode === 'login' ? (
                  <>
                    <span>Enter Mission Control</span>
                    <ArrowRight className="w-4 h-4 arrow-slide" />
                  </>
                ) : (
                  <>
                    <span>Create Profile &amp; Continue</span>
                    <ArrowRight className="w-4 h-4 arrow-slide" />
                  </>
                )}
              </button>

              {/* Divider */}
              <div className="flex items-center gap-3 my-1">
                <div className="h-px flex-1 bg-[var(--border-hairline)]" />
                <span className="text-[0.62rem] uppercase tracking-widest text-[var(--text-faint)] font-mono">
                  Or continue with SSO
                </span>
                <div className="h-px flex-1 bg-[var(--border-hairline)]" />
              </div>

              {/* SSO Buttons */}
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  disabled={isGoogleLoading}
                  onClick={handleGoogleSignIn}
                  className="flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium border border-[var(--border-hairline)] bg-[var(--surface-2)] hover:bg-[var(--surface-2-hover)] transition-all cursor-pointer disabled:opacity-60"
                >
                  {isGoogleLoading ? (
                    <Radio className="w-3.5 h-3.5 animate-spin text-[var(--cyan)]" />
                  ) : (
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24">
                      <path
                        fill="#EA4335"
                        d="M12 5c1.6 0 3 .6 4.1 1.7l3.1-3.1C17.3 1.8 14.8 1 12 1 7.5 1 3.7 3.6 1.9 7.3l3.7 2.9C6.5 7.3 9 5 12 5z"
                      />
                      <path
                        fill="#4285F4"
                        d="M23.5 12.3c0-.8-.1-1.7-.2-2.3H12v4.6h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.9z"
                      />
                      <path
                        fill="#FBBC05"
                        d="M5.6 14.8c-.2-.7-.4-1.5-.4-2.8 0-1.3.2-2.1.4-2.8L1.9 6.3C.7 8.7 0 10.3 0 12s.7 3.3 1.9 5.7l3.7-2.9z"
                      />
                      <path
                        fill="#34A853"
                        d="M12 23c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3 0-5.5-2.3-6.4-5.2L1.9 16c1.8 3.7 5.6 7 10.1 7z"
                      />
                    </svg>
                  )}
                  <span>{isGoogleLoading ? 'Connecting…' : 'Google SSO'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    login('github.specialist@isro.gov.in', 'GitHub Specialist');
                    toast.success('Authenticated via GitHub Enterprise');
                    setTimeout(() => router.push('/profile'), 400);
                  }}
                  className="flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium border border-[var(--border-hairline)] bg-[var(--surface-2)] hover:bg-[var(--surface-2-hover)] transition-all cursor-pointer"
                >
                  <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                  </svg>
                  <span>GitHub</span>
                </button>
              </div>
            </form>
          </div>

          {/* Footer Toggle Text */}
          <div className="text-center text-xs text-[var(--text-muted)]">
            {authMode === 'login' ? (
              <span>
                Need access credentials?{' '}
                <button
                  type="button"
                  onClick={() => setAuthMode('signup')}
                  className="font-semibold text-[var(--cyan)] hover:underline cursor-pointer"
                >
                  Create an operator account →
                </button>
              </span>
            ) : (
              <span>
                Already have an operator profile?{' '}
                <button
                  type="button"
                  onClick={() => setAuthMode('login')}
                  className="font-semibold text-[var(--cyan)] hover:underline cursor-pointer"
                >
                  Sign in instead →
                </button>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
