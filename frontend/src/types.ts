export interface Citation {
  chunk_id: string;
  arxiv_id: string;
  title: string;
  authors: string[];
  section_label: string;
  page_start: number;
  page_end: number;
  score: number; // retrieval similarity score (0-1)
  rank: number;
}

export interface QueryRequest {
  query: string;
  top_k: number;
}

export interface StreamChunk {
  token?: string;
  answer?: string;
  citations?: Citation[];
  latency_ms?: number;
  generation_time_ms?: number;
  total_time_ms?: number;
  confidence?: number;
  model?: string;
  done?: boolean;
}
