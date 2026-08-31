'use client';

import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';

interface User {
  email: string;
  name: string;
  role: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email?: string) => void;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  isAuthenticated: false,
  login: () => {},
  logout: () => {},
  loading: true,
});

const AUTH_ROUTES = ['/login', '/signup', '/forgot-password'];

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();
  // Track if we've done the initial session restore
  const initialised = useRef(false);

  // STEP 1: Restore session from storage — runs once on mount only
  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    try {
      const stored = sessionStorage.getItem('satquery_auth_session');
      if (stored) {
        setUser(JSON.parse(stored));
      }
    } catch {
      // sessionStorage unavailable (SSR guard) — ignore
    } finally {
      setLoading(false);
    }
  }, []);

  // STEP 2: Route guard — runs after loading completes whenever pathname changes
  useEffect(() => {
    if (loading) return; // Wait until we know the auth state
    const isAuthRoute = AUTH_ROUTES.some((r) => pathname === r || pathname.startsWith(r + '/'));
    if (!user && !isAuthRoute) {
      router.replace('/login');
    }
  }, [loading, user, pathname, router]);

  const login = (email?: string) => {
    const userEmail = email || 'controller@isro.gov.in';
    const u: User = {
      email: userEmail,
      name: userEmail.includes('@') ? userEmail.split('@')[0] : userEmail,
      role: 'ISRO Specialist',
    };
    // Persist BEFORE setting state so the guard effect never sees null
    try {
      sessionStorage.setItem('satquery_auth_session', JSON.stringify(u));
    } catch {}
    setUser(u);
  };

  const logout = () => {
    try {
      sessionStorage.removeItem('satquery_auth_session');
    } catch {}
    setUser(null);
    router.replace('/login');
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
