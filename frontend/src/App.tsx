/// <reference types="vite/client" />
import React, { useState, useEffect, useRef } from "react";
import {
  Search,
  ArrowRight,
  StopCircle,
  Copy,
  Check,
  AlertTriangle,
  Activity,
  Clock,
  Layers,
  Sparkles,
  BookOpen
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import Markdown from "react-markdown";

import { Citation } from "./types";
import Header from "./components/Header";
import CitationCard from "./components/CitationCard";

export default function App() {
  // Theme state
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Health and Status state
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");

  // Input states
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);

  // Streaming & Response states
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasQueried, setHasQueried] = useState(false);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Copy feedback state
  const [copied, setCopied] = useState(false);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const queryInputRef = useRef<HTMLTextAreaElement | null>(null);

  // API Base URL
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  // Check health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  // Update HTML class on theme change
  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
      root.classList.remove("light");
    } else {
      root.classList.add("light");
      root.classList.remove("dark");
    }
  }, [theme]);

  // Handle Health check
  const checkBackendHealth = async () => {
    setBackendStatus("checking");
    try {
      const response = await fetch(`${apiBaseUrl}/health`, { method: "GET" });
      if (response.ok) {
        setBackendStatus("online");
      } else {
        setBackendStatus("offline");
      }
    } catch (err) {
      console.error("Health check fetch failed:", err);
      setBackendStatus("offline");
    }
  };

  // Stop/cancel the stream
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setIsLoading(false);
    }
  };

  // Submit query
  const handleSubmit = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();

    const queryText = customQuery !== undefined ? customQuery : query;
    if (!queryText.trim()) return;

    // Reset state for new query (single-turn)
    setError(null);
    setAnswer("");
    setCitations([]);
    setLatency(null);
    setIsLoading(true);
    setHasQueried(true);
    setIsStreaming(true);

    const startTime = performance.now();

    // Setup AbortController
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Clear and refocus input as per requirements
    if (customQuery === undefined) {
      setQuery("");
    }
    setTimeout(() => {
      queryInputRef.current?.focus();
    }, 50);

    try {
      const response = await fetch(`${apiBaseUrl}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          query: queryText,
          top_k: topK,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status} (${response.statusText || "Error"})`);
      }

      // Hide core loading state as soon as response headers arrive
      setIsLoading(false);

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No readable response stream found.");
      }

      const decoder = new TextDecoder("utf-8");
      let partialLine = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const textChunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + textChunk).split("\n");
        partialLine = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("data: ")) {
            const jsonStr = trimmed.substring(6).trim();
            if (jsonStr === "[DONE]") {
              break;
            }

            try {
              const parsed = JSON.parse(jsonStr);

              // Handle answer chunk/token
              if (parsed.answer !== undefined) {
                setAnswer((prev) => prev + parsed.answer);
              } else if (parsed.token !== undefined) {
                setAnswer((prev) => prev + parsed.token);
              } else if (parsed.text !== undefined) {
                setAnswer((prev) => prev + parsed.text);
              }

              // Handle citations (can be full array or individual item updates)
              if (parsed.citations !== undefined && Array.isArray(parsed.citations)) {
                setCitations(parsed.citations);
              } else if (parsed.citation !== undefined && parsed.citation !== null) {
                setCitations((prev) => {
                  const exists = prev.some((c) => c.chunk_id === parsed.citation.chunk_id);
                  if (exists) return prev;
                  return [...prev, parsed.citation];
                });
              }

              // Handle latency (if computed on backend)
              if (parsed.latency_ms !== undefined) {
                setLatency(parsed.latency_ms);
              }
            } catch (err) {
              console.warn("Could not parse event stream chunk:", err, trimmed);
            }
          }
        }
      }

      // Set finish latency
      const endTime = performance.now();
      setLatency((prev) => prev !== null ? prev : Math.round(endTime - startTime));

    } catch (err: any) {
      if (err.name === "AbortError") {
        console.log("Stream aborted successfully");
      } else {
        console.error("Fetch error:", err);
        setError(err.message || "Could not reach the RAG backend. Please check your connections.");
      }
    } finally {
      setIsStreaming(false);
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  // Keyboard action handler
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Handle clicking on query suggestions
  const handleChipClick = (suggestion: string) => {
    setQuery(suggestion);
    handleSubmit(undefined, suggestion);
  };

  // Copy output helper
  const copyAnswerToClipboard = async () => {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy answer:", err);
    }
  };

  const isDark = theme === "dark";

  return (
    <div
      id="app-container"
      className={`min-h-screen pt-16 flex flex-col transition-colors duration-300 ${
        isDark
          ? "bg-slate-950 text-slate-100 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950"
          : "bg-slate-50 text-slate-900 bg-gradient-to-b from-slate-50 to-slate-100/60"
      }`}
    >
      {/* Zone 1 - Header */}
      <Header
        theme={theme}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        backendStatus={backendStatus}
        onCheckHealth={checkBackendHealth}
      />

      {/* Main Content Arena */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col">

        {/* Zone 2 - Centered Query Interface */}
        <div
          id="query-interface-zone"
          className={`flex flex-col transition-all duration-500 ease-in-out ${
            hasQueried
              ? "mt-4 mb-8 shrink-0"
              : "my-auto py-12 justify-center items-center h-[55vh]"
          } w-full max-w-3xl mx-auto`}
        >
          {/* Welcome Wordmark & Subtitle (Only shown when no queries have been run) */}
          <AnimatePresence>
            {!hasQueried && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, height: 0, marginBottom: 0, overflow: "hidden" }}
                transition={{ duration: 0.4 }}
                className="text-center mb-8"
              >
                <h2 className="font-display text-4xl sm:text-5xl font-bold tracking-tight mb-3">
                  Research Intelligence, <br/>
                  <span className="bg-gradient-to-r from-blue-500 via-indigo-400 to-teal-400 bg-clip-text text-transparent">
                    Grounded & Cited.
                  </span>
                </h2>
                <p className={`text-sm sm:text-base max-w-md mx-auto ${
                  isDark ? "text-slate-400" : "text-slate-500"
                }`}>
                  Query hundreds of machine learning papers and get cited, trustworthy answers back instantly.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Core Input Panel */}
          <form
            id="query-form"
            onSubmit={handleSubmit}
            className={`w-full rounded-2xl border transition-all duration-300 ${
              isDark
                ? "bg-slate-900/40 border-slate-900 focus-within:border-blue-900/60 focus-within:shadow-2xl focus-within:shadow-blue-500/5 focus-within:bg-slate-900/60"
                : "bg-white border-slate-200 focus-within:border-blue-400 focus-within:shadow-lg focus-within:shadow-slate-200/50"
            } p-2.5 sm:p-3.5 flex flex-col`}
          >
            <div className="flex gap-2.5 items-start">
              <div className="pt-2.5 pl-2">
                <Search className={`h-5 w-5 ${isDark ? "text-slate-500" : "text-slate-400"}`} />
              </div>
              <textarea
                id="query-textarea"
                ref={queryInputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about ML research papers..."
                rows={2}
                disabled={isStreaming}
                className={`flex-1 resize-none bg-transparent outline-none border-none py-1.5 px-0.5 text-base leading-relaxed ${
                  isDark
                    ? "text-slate-100 placeholder-slate-500"
                    : "text-slate-950 placeholder-slate-400"
                }`}
              />
              <div className="shrink-0 pt-1">
                {isStreaming ? (
                  <button
                    type="button"
                    id="stop-query-btn"
                    onClick={handleStop}
                    className="flex items-center space-x-1.5 px-4 py-2.5 rounded-xl text-xs font-semibold bg-rose-500 hover:bg-rose-600 text-white shadow-lg shadow-rose-500/20 transition-all duration-200"
                  >
                    <StopCircle className="h-4 w-4" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    id="submit-query-btn"
                    disabled={!query.trim() || backendStatus === "checking"}
                    className={`flex items-center space-x-1 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-lg transition-all duration-200 ${
                      query.trim() && backendStatus !== "checking"
                        ? "bg-blue-500 hover:bg-blue-600 text-white shadow-blue-500/10 cursor-pointer hover:scale-[1.02] active:scale-[0.98]"
                        : "bg-slate-800 text-slate-500 shadow-none cursor-not-allowed opacity-50 dark:bg-slate-900"
                    }`}
                  >
                    <span>Ask</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Slider to configure Top_K */}
            <div className={`mt-3.5 pt-3.5 border-t flex flex-wrap items-center justify-between gap-3 text-xs ${
              isDark ? "border-slate-900 text-slate-400" : "border-slate-100 text-slate-500"
            }`}>
              <div className="flex items-center space-x-2">
                <Layers className="h-3.5 w-3.5 text-slate-500" />
                <span className="font-medium">Sources to retrieve:</span>
                <span className={`font-mono font-semibold px-2 py-0.5 rounded ${
                  isDark ? "bg-slate-900 text-blue-400" : "bg-slate-100 text-blue-600"
                }`}>
                  {topK}
                </span>
              </div>
              <div className="flex items-center space-x-3 flex-1 max-w-[200px] sm:max-w-xs justify-end">
                <span className="text-[10px] font-mono">3</span>
                <input
                  id="sources-slider"
                  type="range"
                  min="3"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  disabled={isStreaming}
                  className="w-full h-1 bg-slate-800 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <span className="text-[10px] font-mono">20</span>
              </div>
            </div>
          </form>

          {/* Quick Query Suggestions Chips */}
          <div className="w-full mt-4 flex flex-col space-y-2">
            {!hasQueried && (
              <span className={`text-[11px] font-mono tracking-wider uppercase ${
                isDark ? "text-slate-600" : "text-slate-400"
              }`}>
                Suggested ML Queries
              </span>
            )}
            <div className="flex flex-wrap gap-2">
              <button
                id="chip-suggestion-1"
                type="button"
                onClick={() => handleChipClick("What are the key differences between LoRA and full fine-tuning?")}
                disabled={isStreaming}
                className={`text-xs px-3.5 py-2 rounded-full border transition-all text-left duration-200 cursor-pointer ${
                  isDark
                    ? "bg-slate-900/30 border-slate-900/60 text-slate-300 hover:bg-slate-900/80 hover:border-slate-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300 shadow-sm"
                }`}
              >
                What are the key differences between LoRA and full fine-tuning?
              </button>
              <button
                id="chip-suggestion-2"
                type="button"
                onClick={() => handleChipClick("How do recent papers approach learning rate scheduling?")}
                disabled={isStreaming}
                className={`text-xs px-3.5 py-2 rounded-full border transition-all text-left duration-200 cursor-pointer ${
                  isDark
                    ? "bg-slate-900/30 border-slate-900/60 text-slate-300 hover:bg-slate-900/80 hover:border-slate-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300 shadow-sm"
                }`}
              >
                How do recent papers approach learning rate scheduling?
              </button>
              <button
                id="chip-suggestion-3"
                type="button"
                onClick={() => handleChipClick("What datasets are most commonly used for instruction tuning?")}
                disabled={isStreaming}
                className={`text-xs px-3.5 py-2 rounded-full border transition-all text-left duration-200 cursor-pointer ${
                  isDark
                    ? "bg-slate-900/30 border-slate-900/60 text-slate-300 hover:bg-slate-900/80 hover:border-slate-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300 shadow-sm"
                }`}
              >
                What datasets are most commonly used for instruction tuning?
              </button>
            </div>
          </div>

          {/* Inline Error State */}
          <AnimatePresence>
            {error && (
              <motion.div
                id="error-block"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="w-full mt-4 p-4 rounded-xl border border-rose-500/20 bg-rose-500/5 text-rose-500 flex items-start space-x-3"
              >
                <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-semibold">Backend Communication Error</p>
                  <p className="text-xs opacity-90 mt-1">{error}</p>
                  <button
                    id="retry-query-btn"
                    onClick={() => handleSubmit()}
                    className="mt-3 text-xs font-mono font-bold bg-rose-500 hover:bg-rose-600 text-white px-3 py-1.5 rounded-lg transition-colors cursor-pointer"
                  >
                    Retry Query
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Zone 3 - Response Area (Shown after first query execution) */}
        <AnimatePresence>
          {hasQueried && !error && (
            <motion.div
              id="response-area-zone"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="w-full flex-1 flex flex-col"
            >
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

                {/* Left Column - Answer Output */}
                <div className="lg:col-span-8 space-y-6">
                  <div
                    className={`p-6 sm:p-8 rounded-2xl border transition-all duration-300 ${
                      isDark
                        ? "bg-slate-900/20 border-slate-900"
                        : "bg-white border-slate-200/80 shadow-sm"
                    }`}
                  >
                    {/* Answer Header: Latency + Status */}
                    <div className="flex justify-between items-center pb-4 mb-6 border-b border-dashed dark:border-slate-900 border-slate-100">
                      <div className="flex items-center space-x-2">
                        <Sparkles className="h-4 w-4 text-blue-500" />
                        <span className="font-display text-sm font-bold tracking-tight">
                          Grounded Response
                        </span>
                      </div>

                      {/* Query latency badge */}
                      {latency !== null && (
                        <div
                          id="latency-badge"
                          className={`flex items-center space-x-1.5 text-xs font-mono px-2.5 py-1 rounded-full ${
                            isDark ? "bg-slate-900 text-slate-400" : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          <Clock className="h-3.5 w-3.5" />
                          <span>{latency} ms</span>
                        </div>
                      )}
                    </div>

                    {/* Shimmer skeleton while waiting for first tokens */}
                    {isLoading && !answer && (
                      <div id="skeleton-shimmer" className="space-y-4">
                        <div className="h-4 w-11/12 rounded animate-shimmer" />
                        <div className="h-4 w-full rounded animate-shimmer" />
                        <div className="h-4 w-10/12 rounded animate-shimmer" />
                        <div className="h-4 w-8/12 rounded animate-shimmer" />
                        <p className={`text-xs font-mono mt-4 ${
                          isDark ? "text-slate-500" : "text-slate-400"
                        }`}>
                          Generating answer from RAG context...
                        </p>
                      </div>
                    )}

                    {/* Text Stream Output */}
                    <div className="relative">
                      {answer && (
                        <div className="markdown-body">
                          <Markdown>{answer}</Markdown>
                          {/* Pulse cursor while streaming */}
                          {isStreaming && (
                            <span className="inline-block w-1.5 h-4.5 ml-1 bg-blue-500 animate-pulse relative top-0.5" />
                          )}
                        </div>
                      )}

                      {!isLoading && !answer && (
                        <p className="text-sm italic text-slate-500">
                          Waiting for stream payload...
                        </p>
                      )}
                    </div>

                    {/* Copy Button (Appears when streaming completes) */}
                    {answer && !isStreaming && (
                      <div className="mt-8 pt-4 border-t dark:border-slate-900 border-slate-100 flex justify-end">
                        <button
                          id="copy-answer-btn"
                          onClick={copyAnswerToClipboard}
                          className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg border transition-all ${
                            isDark
                              ? "bg-slate-900 border-slate-800 hover:bg-slate-800 text-slate-300 hover:text-white"
                              : "bg-slate-50 border-slate-200 hover:bg-slate-100 text-slate-600 hover:text-slate-900 shadow-sm"
                          }`}
                        >
                          {copied ? (
                            <>
                              <Check className="h-3.5 w-3.5 text-emerald-500" />
                              <span className="text-emerald-500">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-3.5 w-3.5" />
                              <span>Copy answer</span>
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column - Citations List */}
                <div className="lg:col-span-4 space-y-4">
                  <div className="flex items-center justify-between pb-2">
                    <div className="flex items-center space-x-2">
                      <BookOpen className="h-4 w-4 text-blue-500" />
                      <h3 className="font-display font-bold text-sm tracking-tight uppercase">
                        Sources
                      </h3>
                    </div>
                    <span className={`text-xs font-mono px-2 py-0.5 rounded-full font-bold ${
                      isDark ? "bg-slate-900 text-blue-400" : "bg-blue-100 text-blue-700"
                    }`}>
                      {citations.length} cited
                    </span>
                  </div>

                  {/* Horizontal scroll on mobile, structured list/grid on desktop */}
                  {citations.length === 0 ? (
                    <div className={`p-8 rounded-2xl border text-center ${
                      isDark ? "bg-slate-900/10 border-slate-900/60" : "bg-white border-slate-200"
                    }`}>
                      {isLoading ? (
                        <p className="text-xs text-slate-500 animate-pulse">Retrieving citations...</p>
                      ) : (
                        <p className="text-xs text-slate-500">No sources cited for this query.</p>
                      )}
                    </div>
                  ) : (
                    <div
                      id="citations-container"
                      className="flex lg:flex-col gap-4 overflow-x-auto pb-4 lg:pb-0 lg:overflow-x-visible snap-x snap-mandatory"
                    >
                      {citations.map((citation, idx) => (
                        <div key={citation.chunk_id || idx} className="w-[280px] sm:w-[320px] lg:w-full shrink-0 snap-center">
                          <CitationCard
                            citation={citation}
                            index={idx}
                            theme={theme}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Aesthetic minimalistic footer */}
      <footer className={`h-12 border-t mt-auto transition-colors duration-300 text-[10px] font-mono flex items-center justify-center ${
        isDark ? "bg-slate-950/40 border-slate-900 text-slate-600" : "bg-slate-50 border-slate-200 text-slate-400"
      }`}>
        <p>PaperLens © 2026. Armed with high-precision local RAG.</p>
      </footer>
    </div>
  );
}
