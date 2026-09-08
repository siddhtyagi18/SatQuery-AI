// API_MODE controls which API client is used.
// Set NEXT_PUBLIC_API_MODE=mock in .env.local to use the mock layer for
// offline development without the FastAPI backend.
// Defaults to 'live' so the real backend is used out of the box.
export const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const isRemoteWithoutBackend = (() => {
  if (typeof window === 'undefined') {
    if (process.env.VERCEL || process.env.NEXT_PUBLIC_VERCEL_ENV) {
      return !process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_URL.includes('localhost');
    }
    return false;
  }
  const host = window.location.hostname;
  const isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0';
  const backendIsLocal = FASTAPI_BASE_URL.includes('localhost') || FASTAPI_BASE_URL.includes('127.0.0.1');
  return !isLocalHost && backendIsLocal;
})();

export const API_MODE: 'mock' | 'live' = (() => {
  const explicit = process.env.NEXT_PUBLIC_API_MODE as 'mock' | 'live' | undefined;

  // If explicitly set to mock, always use mock
  if (explicit === 'mock') return 'mock';

  // Safety: even if explicitly set to 'live', if we're on a remote host
  // (Vercel, etc.) with no real backend URL configured, force mock mode
  // to prevent "Upload failed" / network errors hitting localhost from a
  // production domain.
  if (isRemoteWithoutBackend) {
    return 'mock';
  }

  if (explicit === 'live') return 'live';

  return 'live';
})();

export const APP_VERSION = '0.1.0-demo';

const _DEMO_MODE_ENV = process.env.NEXT_PUBLIC_DEMO_MODE?.toLowerCase();
export const DEMO_MODE =
  _DEMO_MODE_ENV === 'true' || _DEMO_MODE_ENV === '1'
    ? true
    : _DEMO_MODE_ENV === 'false' || _DEMO_MODE_ENV === '0'
    ? false
    : API_MODE === 'mock';

export const DEMO_BADGE_TEXT = DEMO_MODE
  ? 'DEMO MODE — LOCAL AUTH'
  : 'LIVE MODE';

