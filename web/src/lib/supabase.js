import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

/** True when the Supabase env vars are set; the login page explains what to set otherwise. */
export const supabaseConfigured = Boolean(url && anonKey);

// PKCE: Google sign-in returns ?code=..., which supabase-js exchanges in this same browser
export const supabase = supabaseConfigured
  ? createClient(url, anonKey, { auth: { flowType: "pkce", persistSession: true, autoRefreshToken: true } })
  : null;
