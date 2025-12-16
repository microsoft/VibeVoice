import { Moon, Sun } from 'lucide-react';

interface ThemeToggleProps {
  isDark: boolean;
  onToggle: () => void;
}

export function ThemeToggle({ isDark, onToggle }: ThemeToggleProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="p-2.5 rounded-xl bg-surface-100 dark:bg-surface-800 
                 border border-surface-200 dark:border-surface-700
                 hover:bg-surface-200 dark:hover:bg-surface-700
                 transition-all duration-200"
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? (
        <Sun className="w-5 h-5 text-amber-500" />
      ) : (
        <Moon className="w-5 h-5 text-primary-600" />
      )}
    </button>
  );
}

