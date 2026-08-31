// API_MODE controls which API client is used.
// Set NEXT_PUBLIC_API_MODE=mock in .env.local to use the mock layer for
// offline development without the FastAPI backend.
// Defaults to 'live' so the real backend is used out of the box.
export const API_MODE = (
  (process.env.NEXT_PUBLIC_API_MODE as 'mock' | 'live' | undefined) ?? 'live'
) as 'mock' | 'live';


// When API_MODE is 'live', all SatQueryApi calls will be proxied to this URL.
// Example: 'https://satquery-ai-api.isro.gov.in' or 'http://localhost:8000'
export const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const APP_VERSION = '0.1.0-demo';
export const DEMO_MODE = API_MODE === 'mock';
export const DEMO_BADGE_TEXT = 'DEMO MODE — MOCK API';
