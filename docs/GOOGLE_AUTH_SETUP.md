# Google Authentication Setup Guide for SatQuery-AI

This document provides complete, step-by-step instructions for enabling Google OAuth with your Supabase backend for SatQuery-AI.

---

## 1. Project Details

- **Supabase Project URL**: `https://vrymjnolvxnrchgohqzg.supabase.co`
- **Supabase Project Reference**: `vrymjnolvxnrchgohqzg`
- **Authorized Redirect URI (Callback)**:  
  `https://vrymjnolvxnrchgohqzg.supabase.co/auth/v1/callback`

---

## 2. Step 1: Create OAuth Credentials in Google Cloud Console

1. Open the [Google Cloud Console Credentials Page](https://console.cloud.google.com/apis/credentials).
2. Select or create a Google Cloud Project (e.g. `satquery-ai` or `isro-mission-control`).
3. If not yet configured, configure the **OAuth consent screen**:
   - User Type: **External** (or Internal for Google Workspace).
   - App Name: `SatQuery-AI`.
   - User support email: Select your email.
   - Developer contact email: Enter your email.
   - Scopes: Add `.../auth/userinfo.email` and `.../auth/userinfo.profile`.
   - Click **Save and Continue**.
4. Go to **Credentials** -> Click **+ CREATE CREDENTIALS** -> Select **OAuth client ID**.
5. Choose Application type: **Web application**.
6. Name: `SatQuery-AI Supabase Auth`.
7. Under **Authorized JavaScript origins**, add:
   - `http://localhost:3000`
   - `https://vrymjnolvxnrchgohqzg.supabase.co`
8. Under **Authorized redirect URIs**, add:
   - `https://vrymjnolvxnrchgohqzg.supabase.co/auth/v1/callback`
9. Click **Create**.
10. Copy your **Client ID** and **Client Secret**.

---

## 3. Step 2: Enable Google Provider in Supabase Dashboard

1. Open [Supabase Auth Providers](https://supabase.com/dashboard/project/vrymjnolvxnrchgohqzg/auth/providers).
2. Scroll to **Google** and click to expand.
3. Toggle **Enable Sign in with Google** to **ON**.
4. Paste the **Client ID** from Step 1.
5. Paste the **Client Secret** from Step 1.
6. Click **Save**.

---

## 4. Step 3: Verify Redirect URLs in Supabase

1. In Supabase Dashboard, go to **Authentication** -> **URL Configuration**.
2. Ensure **Site URL** is set to `http://localhost:3000` (or your production URL).
3. Under **Redirect URLs**, add:
   - `http://localhost:3000/**`
   - `http://localhost:3000/auth/callback`

---

## 5. Flow Overview

```
User clicks "Google SSO"
        │
        ▼
supabase.auth.signInWithOAuth({ provider: 'google' })
        │
        ▼
Redirects to accounts.google.com
        │ (User selects Google Account)
        ▼
Google redirects to Supabase:
https://vrymjnolvxnrchgohqzg.supabase.co/auth/v1/callback
        │
        ▼
Supabase exchanges token & redirects to:
http://localhost:3000/auth/callback?code=...
        │
        ▼
app/auth/callback/page.tsx exchanges code for session,
provisions user in `public.profiles`,
and redirects to /profile (Mission Console)
```
