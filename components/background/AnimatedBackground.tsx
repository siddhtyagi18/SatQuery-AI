// components/background/AnimatedBackground.tsx
// Cinematic remote-sensing mission control atmospheric background.
// Layers: Ambient Gradient Mesh -> Geospatial Coordinate Grid -> Orbital Radar Ring -> Telemetry Particles.
// Smooth mouse parallax with lerp smoothing + translate3d + pointer-events: none.
'use client';

import { useEffect, useRef } from 'react';

export function AnimatedBackground() {
  const orbitalRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const particlesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Respect reduced motion preferences
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    let mouseX = 0;
    let mouseY = 0;
    let currentX = 0;
    let currentY = 0;
    let animationFrameId: number;

    const handleMouseMove = (e: MouseEvent) => {
      // Normalize mouse coordinates to [-1, 1] relative to viewport center
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      mouseX = (e.clientX - centerX) / centerX;
      mouseY = (e.clientY - centerY) / centerY;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });

    const animate = () => {
      // Lerp smoothing (linear interpolation) for fluid organic response
      const ease = 0.04;
      currentX += (mouseX - currentX) * ease;
      currentY += (mouseY - currentY) * ease;

      // Parallax layer depths (extremely subtle to avoid any distraction)
      if (orbitalRef.current) {
        orbitalRef.current.style.transform = `translate3d(${currentX * -18}px, ${currentY * -18}px, 0)`;
      }
      if (gridRef.current) {
        gridRef.current.style.transform = `translate3d(${currentX * -10}px, ${currentY * -10}px, 0)`;
      }
      if (particlesRef.current) {
        particlesRef.current.style.transform = `translate3d(${currentX * -24}px, ${currentY * -24}px, 0)`;
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div
      className="fixed inset-0 overflow-hidden pointer-events-none z-0 select-none"
      aria-hidden="true"
    >
      {/* -------------------------------------------------------------
          LAYER 1: Ambient Gradient Atmosphere (Slow organic flow)
          ------------------------------------------------------------- */}
      <div className="absolute inset-0 bg-[var(--surface-0)]">
        {/* Primary deep navy/cyan atmospheric wash */}
        <div
          className="absolute -top-[20%] -left-[10%] w-[70vw] h-[70vw] max-w-[900px] max-h-[900px] rounded-full opacity-25 dark:opacity-20 blur-[120px] transition-opacity duration-1000"
          style={{
            background: 'radial-gradient(circle, rgba(62, 208, 255, 0.4) 0%, rgba(10, 30, 60, 0.2) 50%, transparent 70%)',
            animation: 'ambient-float-1 28s ease-in-out infinite alternate',
          }}
        />

        {/* Secondary magenta / deep orbital resonance glow */}
        <div
          className="absolute -bottom-[20%] -right-[10%] w-[65vw] h-[65vw] max-w-[850px] max-h-[850px] rounded-full opacity-20 dark:opacity-15 blur-[140px] transition-opacity duration-1000"
          style={{
            background: 'radial-gradient(circle, rgba(192, 132, 252, 0.3) 0%, rgba(20, 10, 40, 0.15) 50%, transparent 70%)',
            animation: 'ambient-float-2 36s ease-in-out infinite alternate-reverse',
          }}
        />

        {/* Subtle amber radar node glow */}
        <div
          className="absolute top-[40%] right-[15%] w-[45vw] h-[45vw] max-w-[550px] max-h-[550px] rounded-full opacity-15 dark:opacity-10 blur-[100px]"
          style={{
            background: 'radial-gradient(circle, rgba(255, 176, 32, 0.2) 0%, transparent 65%)',
            animation: 'ambient-float-3 42s ease-in-out infinite alternate',
          }}
        />
      </div>

      {/* -------------------------------------------------------------
          LAYER 2: Geospatial Coordinate Grid & Crosshairs
          ------------------------------------------------------------- */}
      <div
        ref={gridRef}
        className="absolute inset-[-10%] opacity-30 dark:opacity-25 will-change-transform"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(62, 208, 255, 0.07) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(62, 208, 255, 0.07) 1px, transparent 1px)
          `,
          backgroundSize: '80px 80px',
          backgroundPosition: 'center center',
          animation: 'geo-grid-drift 90s linear infinite',
        }}
      >
        {/* Subtle coordinate crosshair markers */}
        <div className="absolute top-[18%] left-[12%] text-[9px] font-mono text-[var(--cyan)] opacity-40 tracking-widest">
          + 17°22&apos;31&quot;N 78°28&apos;28&quot;E [NRSC/SAC]
        </div>
        <div className="absolute top-[68%] left-[22%] text-[9px] font-mono text-[var(--cyan)] opacity-35 tracking-widest">
          + 13°03&apos;42&quot;N 80°14&apos;38&quot;E [ISTRAC]
        </div>
        <div className="absolute top-[35%] right-[14%] text-[9px] font-mono text-[var(--cyan)] opacity-35 tracking-widest">
          + 30°20&apos;08&quot;N 78°02&apos;38&quot;E [IIRS]
        </div>
        <div className="absolute bottom-[14%] right-[25%] text-[9px] font-mono text-[var(--cyan)] opacity-30 tracking-widest">
          + 08°31&apos;48&quot;N 76°54&apos;14&quot;E [VSSC]
        </div>
      </div>

      {/* -------------------------------------------------------------
          LAYER 3: Concentric Orbital Rings & Scanning Radar Sweep
          ------------------------------------------------------------- */}
      <div
        ref={orbitalRef}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[850px] h-[850px] pointer-events-none opacity-25 dark:opacity-20 will-change-transform"
      >
        {/* Outermost dotted orbit ring */}
        <div
          className="absolute inset-0 rounded-full border border-dashed border-[var(--cyan)] opacity-30"
          style={{ animation: 'orbital-spin 120s linear infinite' }}
        />

        {/* Middle continuous orbit ring */}
        <div
          className="absolute inset-[15%] rounded-full border border-[var(--cyan)] opacity-25"
          style={{ animation: 'orbital-spin-reverse 90s linear infinite' }}
        >
          {/* Orbital satellite telemetry ping node */}
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-[var(--cyan)] shadow-[0_0_8px_var(--cyan)]" />
        </div>

        {/* Inner radar ring with rotating sweep gradient */}
        <div className="absolute inset-[32%] rounded-full border border-[var(--cyan)] opacity-20 overflow-hidden">
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: 'conic-gradient(from 0deg, transparent 0deg, rgba(62, 208, 255, 0.25) 45deg, transparent 90deg)',
              animation: 'radar-sweep 14s linear infinite',
            }}
          />
        </div>

        {/* Center reticle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8">
          <div className="absolute inset-0 border border-[var(--cyan)] opacity-40 rounded-full" />
          <div className="absolute top-0 bottom-0 left-1/2 w-px -translate-x-1/2 bg-[var(--cyan)] opacity-30" />
          <div className="absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-[var(--cyan)] opacity-30" />
        </div>
      </div>

      {/* -------------------------------------------------------------
          LAYER 4: Floating Geospatial Sensor Nodes / Particles
          ------------------------------------------------------------- */}
      <div ref={particlesRef} className="absolute inset-0 will-change-transform">
        <div className="particle-node p-1" style={{ top: '15%', left: '20%', animationDelay: '0s' }} />
        <div className="particle-node p-2" style={{ top: '32%', left: '78%', animationDelay: '-4s' }} />
        <div className="particle-node p-3" style={{ top: '65%', left: '15%', animationDelay: '-8s' }} />
        <div className="particle-node p-4" style={{ top: '78%', left: '82%', animationDelay: '-12s' }} />
        <div className="particle-node p-5" style={{ top: '48%', left: '42%', animationDelay: '-16s' }} />
        <div className="particle-node p-6" style={{ top: '22%', left: '55%', animationDelay: '-20s' }} />
      </div>

      {/* -------------------------------------------------------------
          LAYER 5: Slow Scanning Atmospheric Light Beam
          ------------------------------------------------------------- */}
      <div
        className="absolute inset-0 opacity-20 dark:opacity-15 pointer-events-none"
        style={{
          background: 'linear-gradient(105deg, transparent 40%, rgba(62, 208, 255, 0.14) 50%, transparent 60%)',
          backgroundSize: '200% 100%',
          animation: 'scan-light-beam 24s ease-in-out infinite alternate',
        }}
      />
    </div>
  );
}
