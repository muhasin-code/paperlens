import React from "react";
import { Sun, Moon, Activity, RefreshCw } from "lucide-react";

interface HeaderProps {
  theme: "dark" | "light";
  onToggleTheme: () => void;
  backendStatus: "checking" | "online" | "offline";
  onCheckHealth: () => void;
}

export default function Header({
  theme,
  onToggleTheme,
  backendStatus,
  onCheckHealth,
}: HeaderProps) {
  const isDark = theme === "dark";

  return (
    <header
      id="app-header"
      className={`fixed top-0 left-0 right-0 z-50 h-16 border-b transition-all duration-300 ${
        isDark
          ? "bg-slate-950/80 border-slate-900 text-slate-100 backdrop-blur-md"
          : "bg-white/80 border-slate-200 text-slate-900 backdrop-blur-md"
      }`}
    >
      <div className="max-w-7xl mx-auto h-full px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Logo/Wordmark */}
        <div className="flex items-center space-x-3">
          <span className="font-display text-xl font-bold tracking-tight bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent">
            PaperLens
          </span>
          <span
            className={`hidden sm:inline-block text-xs font-mono px-2 py-0.5 rounded-full border ${
              isDark
                ? "bg-slate-900 border-slate-800 text-slate-400"
                : "bg-slate-100 border-slate-200 text-slate-500"
            }`}
          >
            v1.0.0
          </span>
        </div>

        {/* Right controls */}
        <div className="flex items-center space-x-4">
          {/* Health check status indicator */}
          <div
            className={`flex items-center space-x-2 text-xs font-mono px-2.5 py-1 rounded-full border transition-all duration-300 ${
              isDark
                ? "bg-slate-900/50 border-slate-800/80"
                : "bg-slate-50 border-slate-200"
            }`}
          >
            <span className="relative flex h-2 w-2">
              {backendStatus === "checking" && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              )}
              {backendStatus === "online" && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  backendStatus === "online"
                    ? "bg-emerald-500"
                    : backendStatus === "offline"
                    ? "bg-rose-500"
                    : "bg-amber-500"
                }`}
              ></span>
            </span>

            <span className={isDark ? "text-slate-400" : "text-slate-600"}>
              {backendStatus === "online"
                ? "RAG online"
                : backendStatus === "offline"
                ? "RAG offline"
                : "Checking..."}
            </span>

            <button
              id="refresh-health-btn"
              onClick={onCheckHealth}
              disabled={backendStatus === "checking"}
              className={`p-0.5 rounded transition-all hover:bg-slate-200 dark:hover:bg-slate-800 ${
                backendStatus === "checking" ? "animate-spin opacity-50" : ""
              }`}
              title="Re-check backend health"
            >
              <RefreshCw className="h-3 w-3 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300" />
            </button>
          </div>

          {/* Theme Toggle */}
          <button
            id="theme-toggle-btn"
            onClick={onToggleTheme}
            className={`p-2 rounded-lg border transition-all duration-200 ${
              isDark
                ? "bg-slate-900 border-slate-800 hover:bg-slate-800 text-amber-400 hover:text-amber-300"
                : "bg-slate-50 border-slate-200 hover:bg-slate-100 text-slate-700 hover:text-slate-900"
            }`}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
