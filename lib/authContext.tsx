'use client';

import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import {
  onAuthStateChange,
  signOut as authSignOut,
  getAuthMode,
  getCurrentSession,
  type AuthUser,
} from '@/lib/authService';

export type { AuthUser } from '@/lib/authService';
export type { AuthSession } from '@/lib/authService';

interface AuthContextType {
  user: AuthUser | null;
  isAuthenticated: boolean;
  login: (email?: string) => void;
  logout: () => Promise<void>;
  loading: boolean;
  authMode: 'supabase' | 'mock';
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  isAuthenticated: false,
  login: () => {},
  logout: async () => {},
  loading: true,
  authMode: 'mock',
});

const AUTH_ROUTES = ['/login', '/signup', '/forgot-password'];
const MOCK_STORAGE_KEY = 'satquery_auth_session';

function buildMockUser(email: string): AuthUser {
  return {
    email,
    name: email.includes('@') ? email.split('@')[0] : email,
    role: 'ISRO Specialist',
  };
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [authMode] = useState<'supabase' | 'mock'>(getAuthMode());
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    let mounted = true;

    getCurrentSession()
      .then((session) => {
        if (!mounted) return;
        if (session?.user) {
          setUser(session.user);
        }
        setLoading(false);
      })
      .catch(() => {
        if (!mounted) return;
        setLoading(false);
      });

    const unsubscribe = onAuthStateChange((event) => {
      if (!mounted) return;
      if (event.event === 'INITIAL_SESSION') {
        if (event.session?.user) {
          setUser(event.session.user);
        }
        setLoading(false);
      } else if (event.event === 'SIGNED_OUT') {
        setUser(null);
        setLoading(false);
      } else {
        if (event.session?.user) {
          setUser(event.session.user);
        }
        setLoading(false);
      }
    });

    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (loading) return;
    const isAuthRoute = AUTH_ROUTES.some(
      (r) => pathname === r || pathname.startsWith(r + '/')
    );
    if (!user && !isAuthRoute) {
      router.replace('/login');
    } else if (user && isAuthRoute) {
      router.replace('/');
    }
  }, [loading, user, pathname, router]);

  const login = (email?: string) => {
    const userEmail = email || 'controller@isro.gov.in';
    const u = buildMockUser(userEmail);
    try {
      if (typeof window !== 'undefined') {
        sessionStorage.setItem(MOCK_STORAGE_KEY, JSON.stringify(u));
      }
    } catch {
      // Ignore
    }
    setUser(u);
    setLoading(false);
  };

  const logout = async () => {
    try {
      await authSignOut();
    } catch {
      // Ignore sign-out errors from auth service, ensure local state is cleared
    }
    try {
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem(MOCK_STORAGE_KEY);
      }
    } catch {
      // Ignore
    }
    setUser(null);
    router.replace('/login');
  };

  return (
    <AuthContext.Provider
      value={{ user, isAuthenticated: !!user, login, logout, loading, authMode }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
