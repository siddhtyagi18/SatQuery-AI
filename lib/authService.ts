import { supabase, HAS_SUPABASE } from '@/lib/supabase';
import { DEMO_MODE } from '@/lib/config';
import { supabaseAnalysisService } from '@/lib/supabase/services';
import type { Session, User as SupabaseUser } from '@supabase/supabase-js';

export interface AuthUser {
  email: string;
  name: string;
  role: string;
}

export interface AuthSession {
  user: AuthUser;
  accessToken?: string;
  expiresAt?: number;
}

export type AuthMode = 'supabase' | 'mock';

export interface AuthStateChangeEvent {
  event: 'SIGNED_IN' | 'SIGNED_OUT' | 'TOKEN_REFRESHED' | 'USER_UPDATED' | 'INITIAL_SESSION';
  session: AuthSession | null;
}

export type AuthStateChangeListener = (event: AuthStateChangeEvent) => void;

const USE_SUPABASE_AUTH: AuthMode =
  HAS_SUPABASE && !DEMO_MODE ? 'supabase' : 'mock';

const MOCK_STORAGE_KEY = 'satquery_auth_session';

function supabaseUserToAuthUser(sbUser: SupabaseUser | undefined): AuthUser | null {
  if (!sbUser) return null;
  const email = sbUser.email ?? 'unknown@satquery.ai';
  const name =
    (sbUser.user_metadata?.['name'] as string | undefined) ||
    (sbUser.user_metadata?.['full_name'] as string | undefined) ||
    email.split('@')[0] ||
    'Operator';
  return {
    email,
    name,
    role: 'ISRO Specialist',
  };
}

function supabaseSessionToAuthSession(sbSession: Session | null): AuthSession | null {
  if (!sbSession) return null;
  const user = supabaseUserToAuthUser(sbSession.user);
  if (!user) return null;
  return {
    user,
    accessToken: sbSession.access_token,
    expiresAt: sbSession.expires_at,
  };
}

function mockBuildUser(email: string): AuthUser {
  return {
    email,
    name: email.includes('@') ? email.split('@')[0] : email,
    role: 'ISRO Specialist',
  };
}

function mockGetStoredSession(): AuthSession | null {
  try {
    const raw = typeof window !== 'undefined' ? sessionStorage.getItem(MOCK_STORAGE_KEY) : null;
    if (!raw) return null;
    const user = JSON.parse(raw) as AuthUser;
    return { user };
  } catch {
    return null;
  }
}

function mockStoreSession(user: AuthUser | null): void {
  try {
    if (typeof window === 'undefined') return;
    if (user) {
      sessionStorage.setItem(MOCK_STORAGE_KEY, JSON.stringify(user));
    } else {
      sessionStorage.removeItem(MOCK_STORAGE_KEY);
    }
  } catch {
    // Ignore storage errors
  }
}

interface AuthService {
  getMode(): AuthMode;
  signUp(email: string, password: string): Promise<{ session: AuthSession | null; needsEmailConfirmation?: boolean }>;
  signIn(email: string, password: string): Promise<{ session: AuthSession | null }>;
  signOut(): Promise<void>;
  getCurrentSession(): Promise<AuthSession | null>;
  getCurrentUserId(): Promise<string | null>;
  onAuthStateChange(listener: AuthStateChangeListener): () => void;
}

class SupabaseAuthService implements AuthService {
  getMode(): AuthMode {
    return 'supabase';
  }

  async signUp(email: string, password: string) {
    if (!supabase) {
      throw new Error('Supabase client is not configured');
    }
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          name: email.split('@')[0] || 'Operator',
        },
      },
    });
    if (error) {
      throw new Error(error.message || 'Sign up failed');
    }
    if (data.user) {
      const fullName =
        (data.user.user_metadata?.['name'] as string | undefined) ||
        (data.user.user_metadata?.['full_name'] as string | undefined) ||
        undefined;
      try {
        await supabaseAnalysisService.ensureProfileForUser(data.user.id, data.user.email ?? email, fullName);
      } catch (profileErr) {
        throw new Error(
          `Account created but profile setup failed: ${
            profileErr instanceof Error ? profileErr.message : 'unknown error'
          }. Please try signing in again.`
        );
      }
    }
    const session = supabaseSessionToAuthSession(data.session ?? null);
    const needsEmailConfirmation =
      !!data.user && !data.session && !data.user.email_confirmed_at ? true : false;
    return { session, needsEmailConfirmation };
  }

  async signIn(email: string, password: string) {
    if (!supabase) {
      throw new Error('Supabase client is not configured');
    }
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) {
      throw new Error(error.message || 'Sign in failed');
    }
    const session = supabaseSessionToAuthSession(data.session ?? null);
    return { session };
  }

  async signOut(): Promise<void> {
    if (!supabase) return;
    const { error } = await supabase.auth.signOut();
    if (error) {
      throw new Error(error.message || 'Sign out failed');
    }
  }

  async getCurrentSession(): Promise<AuthSession | null> {
    if (!supabase) return null;
    const { data } = await supabase.auth.getSession();
    return supabaseSessionToAuthSession(data.session);
  }

  async getCurrentUserId(): Promise<string | null> {
    if (!supabase) return null;
    try {
      const { data } = await supabase.auth.getUser();
      return data.user?.id ?? null;
    } catch {
      const { data: sess } = await supabase.auth.getSession();
      return sess.session?.user?.id ?? null;
    }
  }

  onAuthStateChange(listener: AuthStateChangeListener): () => void {
    if (!supabase) {
      queueMicrotask(() => listener({ event: 'INITIAL_SESSION', session: null }));
      return () => {};
    }
    const { data: subscription } = supabase.auth.onAuthStateChange((event, sbSession) => {
      const mappedEvent: AuthStateChangeEvent['event'] =
        event === 'INITIAL_SESSION'
          ? 'INITIAL_SESSION'
          : event === 'SIGNED_IN'
          ? 'SIGNED_IN'
          : event === 'SIGNED_OUT'
          ? 'SIGNED_OUT'
          : event === 'TOKEN_REFRESHED'
          ? 'TOKEN_REFRESHED'
          : event === 'USER_UPDATED'
          ? 'USER_UPDATED'
          : 'SIGNED_IN';
      listener({
        event: mappedEvent,
        session: supabaseSessionToAuthSession(sbSession),
      });
    });
    return () => {
      subscription.subscription.unsubscribe();
    };
  }
}

class MockAuthService implements AuthService {
  private listeners: Set<AuthStateChangeListener> = new Set();
  private currentMockSession: AuthSession | null = null;

  constructor() {
    this.currentMockSession = mockGetStoredSession();
  }

  getMode(): AuthMode {
    return 'mock';
  }

  async signUp(email: string, _password: string) {
    void _password;
    await new Promise((r) => setTimeout(r, 800));
    const user = mockBuildUser(email);
    mockStoreSession(user);
    this.currentMockSession = { user };
    this.emit('SIGNED_IN', this.currentMockSession);
    return { session: this.currentMockSession, needsEmailConfirmation: false };
  }

  async signIn(email: string, _password: string) {
    void _password;
    await new Promise((r) => setTimeout(r, 800));
    const user = mockBuildUser(email);
    mockStoreSession(user);
    this.currentMockSession = { user };
    this.emit('SIGNED_IN', this.currentMockSession);
    return { session: this.currentMockSession };
  }

  async signOut(): Promise<void> {
    await new Promise((r) => setTimeout(r, 200));
    mockStoreSession(null);
    this.currentMockSession = null;
    this.emit('SIGNED_OUT', null);
  }

  async getCurrentSession(): Promise<AuthSession | null> {
    const stored = mockGetStoredSession();
    if (stored) {
      this.currentMockSession = stored;
    }
    return this.currentMockSession ?? stored;
  }

  async getCurrentUserId(): Promise<string | null> {
    const session = this.currentMockSession ?? mockGetStoredSession();
    if (!session) return null;
    const email = session.user.email;
    let hash = 0;
    for (let i = 0; i < email.length; i++) {
      hash = (hash * 31 + email.charCodeAt(i)) >>> 0;
    }
    const hex = hash.toString(16).padStart(8, '0');
    const rest = email.padEnd(24, '0').replace(/[^0-9a-f]/g, '0').slice(0, 24);
    return `${hex}-${rest.slice(0, 4)}-${rest.slice(4, 8)}-${rest.slice(8, 12)}-${rest.slice(12, 24)}`;
  }

  onAuthStateChange(listener: AuthStateChangeListener): () => void {
    this.listeners.add(listener);
    const session = mockGetStoredSession() ?? this.currentMockSession;
    this.currentMockSession = session;
    queueMicrotask(() => {
      listener({ event: 'INITIAL_SESSION', session });
    });
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(event: AuthStateChangeEvent['event'], session: AuthSession | null): void {
    for (const listener of this.listeners) {
      try {
        listener({ event, session });
      } catch {
        // Ignore listener errors
      }
    }
  }
}

const authService: AuthService =
  USE_SUPABASE_AUTH === 'supabase'
    ? new SupabaseAuthService()
    : new MockAuthService();

export function getAuthMode(): AuthMode {
  return authService.getMode();
}

export async function signUp(email: string, password: string) {
  return authService.signUp(email, password);
}

export async function signIn(email: string, password: string) {
  return authService.signIn(email, password);
}

export async function signOut() {
  return authService.signOut();
}

export async function getCurrentSession() {
  return authService.getCurrentSession();
}

export async function getCurrentUserId() {
  return authService.getCurrentUserId();
}

export function onAuthStateChange(listener: AuthStateChangeListener): () => void {
  return authService.onAuthStateChange(listener);
}
