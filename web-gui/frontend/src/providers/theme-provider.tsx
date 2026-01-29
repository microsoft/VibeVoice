'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Theme, lightTheme, darkTheme, availableThemes, ThemeId } from '@/lib/theme';

interface ThemeContextType {
  theme: Theme;
  themeId: ThemeId;
  setTheme: (themeId: ThemeId) => void;
  availableThemes: Theme[];
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Validate themeId values to prevent using corrupted localStorage entries
  const isValidThemeId = (value: unknown): value is ThemeId => (
    value === 'light' || value === 'dark' || value === 'system'
  );

  // Read saved preference synchronously on first render to avoid flash
  // Read once and reuse to avoid duplicate localStorage reads and possible inconsistencies
  const initialThemeId: ThemeId = (() => {
    try {
      const saved = localStorage.getItem('vibevoice-theme');
      return isValidThemeId(saved) ? saved : 'system';
    } catch (e) {
      // localStorage may be unavailable in some environments
      return 'system';
    }
  })();

  const [themeId, setThemeIdState] = useState<ThemeId>(() => initialThemeId);

  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      const id = initialThemeId;
      if (id === 'system') {
        const prefersDark = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return prefersDark ? darkTheme : lightTheme;
      }
      return id === 'dark' ? darkTheme : lightTheme;
    } catch (e) {
      return lightTheme;
    }
  });

  // Apply theme based on themeId
  useEffect(() => {
    const applyTheme = (id: ThemeId) => {
      let selectedTheme: Theme;
      
      if (id === 'system') {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        selectedTheme = prefersDark ? darkTheme : lightTheme;
      } else {
        selectedTheme = id === 'dark' ? darkTheme : lightTheme;
      }

      setThemeState(selectedTheme);
      document.documentElement.setAttribute('data-theme', selectedTheme.id);
      
      // Apply CSS variables
      applyThemeVariables(selectedTheme);
    };

    applyTheme(themeId);

    // Listen for system theme changes when using 'system' mode
    if (themeId === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = () => applyTheme('system');
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [themeId]);

  // Save theme to localStorage when changed
  const setTheme = (newThemeId: ThemeId) => {
    setThemeIdState(newThemeId);
    localStorage.setItem('vibevoice-theme', newThemeId);
  };

  // Toggle between light and dark
  const toggleTheme = () => {
    const newThemeId: ThemeId = theme.id === 'light' ? 'dark' : 'light';
    setTheme(newThemeId);
  };

  return (
    <ThemeContext.Provider value={{ theme, themeId, setTheme, availableThemes, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function applyThemeVariables(theme: Theme) {
  const root = document.documentElement;
  
  // Primary colors
  Object.entries(theme.colors.primary).forEach(([key, value]) => {
    root.style.setProperty(`--color-primary-${key}`, value);
  });
  
  // Background colors
  root.style.setProperty('--color-background', theme.colors.background.DEFAULT);
  root.style.setProperty('--color-surface', theme.colors.background.surface);
  root.style.setProperty('--color-elevated', theme.colors.background.elevated);
  
  // Foreground colors
  root.style.setProperty('--color-foreground', theme.colors.foreground.DEFAULT);
  root.style.setProperty('--color-muted', theme.colors.foreground.muted);
  root.style.setProperty('--color-subtle', theme.colors.foreground.subtle);
  
  // Border colors
  root.style.setProperty('--color-border', theme.colors.border.DEFAULT);
  root.style.setProperty('--color-border-subtle', theme.colors.border.subtle);
  
  // Semantic colors
  root.style.setProperty('--color-success', theme.colors.semantic.success);
  root.style.setProperty('--color-warning', theme.colors.semantic.warning);
  root.style.setProperty('--color-error', theme.colors.semantic.error);
  root.style.setProperty('--color-info', theme.colors.semantic.info);
  
  // Border radius
  root.style.setProperty('--radius-sm', theme.borderRadius.sm);
  root.style.setProperty('--radius-md', theme.borderRadius.md);
  root.style.setProperty('--radius-lg', theme.borderRadius.lg);
  root.style.setProperty('--radius-xl', theme.borderRadius.xl);
  
  // Shadows
  root.style.setProperty('--shadow-sm', theme.shadows.sm);
  root.style.setProperty('--shadow-md', theme.shadows.md);
  root.style.setProperty('--shadow-lg', theme.shadows.lg);
  root.style.setProperty('--shadow-xl', theme.shadows.xl);
  
  // Font families
  root.style.setProperty('--font-sans', theme.typography.fontFamily.sans.join(', '));
  root.style.setProperty('--font-mono', theme.typography.fontFamily.mono.join(', '));
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}
