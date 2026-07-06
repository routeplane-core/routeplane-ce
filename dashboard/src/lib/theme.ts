// Theme state — light/dark, persisted to localStorage (theme only, never
// secrets). Honors the OS preference on first visit.
import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "rp.theme";

function initial(): Theme {
  const stored = localStorage.getItem(KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function apply(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initial);

  useEffect(() => {
    apply(theme);
    localStorage.setItem(KEY, theme);
  }, [theme]);

  return {
    theme,
    toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    set: setTheme,
  };
}

// Apply synchronously on module load to avoid a flash of the wrong theme.
apply(initial());
