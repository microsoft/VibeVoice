import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Script from 'next/script';
import { ThemeProvider } from "@/providers/theme-provider";
import { Toaster } from "@/components/ui/sonner";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "VibeVoice-Narrator",
  description: "Markdown-to-Speech conversion tool",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`} suppressHydrationWarning>
      {/* Initialize theme before React hydration to avoid flash of wrong theme */}
      <Script id="theme-init" strategy="beforeInteractive">
        {`(function(){try{var t=localStorage.getItem('vibevoice-theme'); if(t){ if(t==='system'){ var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light'); } else { document.documentElement.setAttribute('data-theme', t === 'dark' ? 'dark':'light'); } } }catch(e){} })();`}
      </Script>
        <ThemeProvider>
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
