// lib/config.ts
// Central configuration. Change API_MODE to 'live' and set FASTAPI_BASE_URL
// to point at a real FastAPI backend — no component code needs to change.

export const API_MODE: 'mock' | 'live' = 'mock';

// When API_MODE is 'live', all SatQueryApi calls will be proxied to this URL.
// Example: 'https://satquery-ai-api.isro.gov.in' or 'http://localhost:8000'
export const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const APP_VERSION = '0.1.0-demo';
export const DEMO_MODE = API_MODE === 'mock';
export const DEMO_BADGE_TEXT = 'DEMO MODE — MOCK API';
