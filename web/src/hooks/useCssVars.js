import { useMemo, useSyncExternalStore } from "react";

// The theme is a class on <html> (set by ThemeProvider after render), so watch the class itself
function subscribe(onChange) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => observer.disconnect();
}
const htmlClass = () => document.documentElement.className;

/**
 * Current values of CSS custom properties, re-read whenever the theme class changes.
 * Chart libraries write colours into SVG attributes, where var(--x) isn't reliable.
 * @param {string[]} names e.g. ["--chart-normal", "--chart-grid"]
 * @returns {Record<string, string>}
 */
export function useCssVars(names) {
  const theme = useSyncExternalStore(subscribe, htmlClass);
  const key = names.join(",");
  return useMemo(() => {
    void theme; // recompute when the class changes
    const style = getComputedStyle(document.documentElement);
    return Object.fromEntries(key.split(",").map((n) => [n, style.getPropertyValue(n).trim()]));
  }, [key, theme]);
}
