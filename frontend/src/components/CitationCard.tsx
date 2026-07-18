import React from "react";
import { ExternalLink, BookOpen } from "lucide-react";
import { motion } from "motion/react";
import { Citation } from "../types";

interface CitationCardProps {
  citation: Citation;
  index: number;
  theme: "dark" | "light";
}

export default function CitationCard({
  citation,
  index,
  theme,
}: CitationCardProps) {
  const isDark = theme === "dark";

  // Format authors
  const formatAuthors = (authors: string[]) => {
    if (!authors || authors.length === 0) return "Unknown Author";
    if (authors.length === 1) return authors[0];
    if (authors.length === 2) return `${authors[0]} & ${authors[1]}`;
    return `${authors[0]} et al.`;
  };

  // Format page range
  const formatPageRange = (start: number, end: number) => {
    if (!start) return "";
    if (!end || start === end) return `p. ${start}`;
    return `pp. ${start}–${end}`;
  };

  // Get percentage
  const confidence = citation.score <= 1
    ? Math.round(citation.score * 100)
    : Math.round(citation.score);

  // Colored section pill based on label
  const getSectionColor = (label: string) => {
    const norm = label.toLowerCase();
    if (norm.includes("method") || norm.includes("approach") || norm.includes("model") || norm.includes("algorithm")) {
      return isDark
        ? "bg-blue-950/40 text-blue-300 border-blue-900/40"
        : "bg-blue-50 text-blue-700 border-blue-200";
    }
    if (norm.includes("result") || norm.includes("evaluation") || norm.includes("experiment") || norm.includes("finding") || norm.includes("performance")) {
      return isDark
        ? "bg-emerald-950/40 text-emerald-300 border-emerald-900/40"
        : "bg-emerald-50 text-emerald-700 border-emerald-200";
    }
    if (norm.includes("intro") || norm.includes("background") || norm.includes("overview") || norm.includes("related")) {
      return isDark
        ? "bg-slate-800/60 text-slate-300 border-slate-700/50"
        : "bg-slate-100 text-slate-700 border-slate-200";
    }
    if (norm.includes("discussion") || norm.includes("conclusion") || norm.includes("future") || norm.includes("summary")) {
      return isDark
        ? "bg-purple-950/40 text-purple-300 border-purple-900/40"
        : "bg-purple-50 text-purple-700 border-purple-200";
    }
    if (norm.includes("abstract")) {
      return isDark
        ? "bg-amber-950/40 text-amber-300 border-amber-900/40"
        : "bg-amber-50 text-amber-700 border-amber-200";
    }
    return isDark
      ? "bg-teal-950/40 text-teal-300 border-teal-900/40"
      : "bg-teal-50 text-teal-700 border-teal-200";
  };

  const arxivUrl = `https://arxiv.org/abs/${citation.arxiv_id}`;

  return (
    <motion.div
      id={`citation-card-${citation.chunk_id}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        delay: index * 0.1,
        duration: 0.35,
        ease: "easeOut",
      }}
      className={`group flex flex-col p-4 rounded-xl border transition-all duration-300 ${
        isDark
          ? "bg-slate-900/40 border-slate-900 hover:border-blue-900/50 hover:bg-slate-900/60 shadow-lg shadow-black/10"
          : "bg-white border-slate-200 hover:border-blue-300 hover:bg-slate-50/50 shadow-sm"
      }`}
    >
      {/* Header with section and arXiv link */}
      <div className="flex items-center justify-between gap-2 mb-2.5">
        <span
          className={`text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full border ${getSectionColor(
            citation.section_label
          )}`}
        >
          {citation.section_label}
        </span>
        <a
          id={`arxiv-link-${citation.arxiv_id}`}
          href={arxivUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={`p-1 rounded transition-colors ${
            isDark
              ? "text-slate-400 hover:text-blue-400 hover:bg-slate-800/80"
              : "text-slate-500 hover:text-blue-600 hover:bg-slate-100"
          }`}
          title={`View on arXiv: ${citation.arxiv_id}`}
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>

      {/* Paper Title (bold, truncated to 2 lines) */}
      <h4
        className={`font-semibold text-sm leading-snug line-clamp-2 mb-1.5 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors duration-200 ${
          isDark ? "text-slate-100" : "text-slate-900"
        }`}
        title={citation.title}
      >
        {citation.title}
      </h4>

      {/* Authors and Page numbers */}
      <div className="flex items-center justify-between text-xs text-slate-400 dark:text-slate-500 mb-3">
        <span className="truncate max-w-[70%]" title={citation.authors.join(", ")}>
          {formatAuthors(citation.authors)}
        </span>
        <span className="flex items-center space-x-1 shrink-0">
          <BookOpen className="h-3 w-3 opacity-60" />
          <span>{formatPageRange(citation.page_start, citation.page_end)}</span>
        </span>
      </div>

      {/* Confidence Score Bar */}
      <div className="mt-auto space-y-1">
        <div className="flex justify-between items-center text-[10px] font-mono">
          <span className={isDark ? "text-slate-500" : "text-slate-400"}>Confidence</span>
          <span className={`font-semibold ${isDark ? "text-blue-400" : "text-blue-600"}`}>
            {confidence}%
          </span>
        </div>
        <div className={`w-full h-1 rounded-full overflow-hidden ${
          isDark ? "bg-slate-800" : "bg-slate-100"
        }`}>
          <div
            className={`h-full rounded-full bg-gradient-to-r from-blue-500 to-teal-400 transition-all duration-500`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
    </motion.div>
  );
}
