import { useCallback, useEffect, useMemo, useState } from "react";
import { ThemeContext } from "@/providers/contexts";

const STORAGE_KEY = "theme";
const media = () => window.matchMedia("(prefers-color-scheme: dark)");

function readStored() {
  try {
    return localStorage.getItem(STORAGE_KEY) || "system";
  } catch {
    return "system"; // storage blocked (private mode)
  }
}

/** Light, dark or follow the system; the choice is remembered in this browser. */
export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(readStored);
  const [systemDark, setSystemDark] = useState(() => media().matches);

  useEffect(() => {
    const m = media();
    const onChange = (e) => setSystemDark(e.matches);
    m.addEventListener("change", onChange);
    return () => m.removeEventListener("change", onChange);
  }, []);

  const resolved = theme === "system" ? (systemDark ? "dark" : "light") : theme;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);

  const setTheme = useCallback((next) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // not remembered, still applied
    }
  }, []);

  const value = useMemo(() => ({ theme, resolved, setTheme }), [theme, resolved, setTheme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
