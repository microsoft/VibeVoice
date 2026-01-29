'use client';

import * as React from 'react';

type IconProps = React.SVGProps<SVGSVGElement>;

export const Play: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M5 3v18l15-9L5 3z" />
    </svg>
  )
};

export const Pause: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <rect x="6" y="4" width="4" height="16" />
      <rect x="14" y="4" width="4" height="16" />
    </svg>
  )
};

export const Download: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M12 3v12" />
      <path d="M5 12l7 7 7-7" />
      <path d="M5 21h14" />
    </svg>
  )
};

export const FileText: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M8 13h8" />
      <path d="M8 17h8" />
      <path d="M8 9h4" />
    </svg>
  )
};

export const Mic2: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M19 11a7 7 0 0 1-14 0" />
      <path d="M12 19v4" />
      <path d="M8 23h8" />
    </svg>
  )
};

export const Settings: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09c.66 0 1.2-.48 1.51-1a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06c.45.45 1.05.6 1.65.45H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09c0 .6.35 1.2 1 1.51.6.15 1.2 0 1.65-.45l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06c-.45.45-.6 1.05-.45 1.65V9c0 .6.48 1.2 1 1.51H21a2 2 0 1 1 0 4h-.09c-.66 0-1.2.48-1.51 1z" />
    </svg>
  )
};

export const Upload: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v13" />
    </svg>
  )
};

export const X: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  )
};

export const SlidersHorizontal: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M4 6h16" />
      <circle cx="9" cy="6" r="1.25" fill="currentColor" />

      <path d="M4 12h16" />
      <circle cx="14" cy="12" r="1.25" fill="currentColor" />

      <path d="M4 18h16" />
      <circle cx="8" cy="18" r="1.25" fill="currentColor" />
    </svg>
  )
};

export const Moon: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
};

export const Sun: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="M4.93 4.93l1.41 1.41" />
      <path d="M17.66 17.66l1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="M4.93 19.07l1.41-1.41" />
      <path d="M17.66 6.34l1.41-1.41" />
    </svg>
  )
};

export const Monitor: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8" />
      <path d="M12 17v4" />
    </svg>
  )
};

export const Check: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  )
};

export const Loader2: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M21 12a9 9 0 1 1-9-9" />
    </svg>
  )
};

export const SkipForward: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M5 4v16l11-8L5 4z" />
      <path d="M19 5v14" />
    </svg>
  )
};

export const SkipBack: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M19 4v16L8 12l11-8z" />
      <path d="M5 5v14" />
    </svg>
  )
};

export const Volume2: React.FC<IconProps> = ({ className, ...props }) => {
  const labelled = 'aria-label' in props || !!(props as any).title;
  return (
    <svg
      aria-hidden={labelled ? undefined : true}
      role={labelled ? 'img' : undefined}
      className={className}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M11 5L6 9H2v6h4l5 4V5z" />
    <path d="M15.5 8.5a5 5 0 0 1 0 7" />
    <path d="M17.5 6.5a7 7 0 0 1 0 11" />
    </svg>
  )
};
