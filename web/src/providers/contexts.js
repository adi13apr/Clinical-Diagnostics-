import { createContext, useContext } from "react";

export const ThemeContext = createContext(null);
export const AuthContext = createContext(null);

export const DiagnosisContext = createContext(null);

/** The current upload, its prediction and report; kept while you visit other pages. */
export function useDiagnosis() {
  const ctx = useContext(DiagnosisContext);
  if (!ctx) throw new Error("useDiagnosis must be used inside DiagnosisProvider");
  return ctx;
}

/** { theme: "light"|"dark"|"system", resolved: "light"|"dark", setTheme } */
export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}

/** { session, user, loading, configured, signIn, signUp, signInWithGoogle, signOut } */
export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
