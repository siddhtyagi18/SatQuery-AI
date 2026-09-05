import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? '';

export const HAS_SUPABASE = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase: SupabaseClient | null = HAS_SUPABASE
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;
