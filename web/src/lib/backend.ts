export type Json = Record<string, any>;

export type TrackItem = {
  track_id: string;
  score?: number;
  track_name?: string;
  artist_name?: string;
  country?: string;
  genre?: string;
  popularity?: number;
  stream_count?: number;
  release_date?: string;

  // optional tag join fields
  tags?: Record<string, any>;
  tags_missing?: boolean;
  tags_provider?: string | null;
  tags_updated_at?: number;
};

export type FeedResponse = {
  ok: boolean;
  country: string;
  n: number;
  sections: Record<string, TrackItem[]>;
  debug?: any;
};

export type TagsFiltersResponse = {
  ok: boolean;
  moods: { key: string; label: string; count: number }[];
  debug?: any;
};

const BACKEND = "/api/backend";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`GET ${path} failed: ${res.status} ${txt}`);
  }
  return (await res.json()) as T;
}

export function getHomeFeed(params: {
  country: string;
  n?: number;
  debug?: boolean;
  include_tags?: boolean;
  mood_rails?: boolean;
  moods_k?: number;
  mood_rail_n?: number;
}): Promise<FeedResponse> {
  const usp = new URLSearchParams();
  usp.set("country", params.country);
  if (params.n != null) usp.set("n", String(params.n));
  if (params.debug) usp.set("debug", "true");
  if (params.include_tags) usp.set("include_tags", "true");
  if (params.mood_rails) usp.set("mood_rails", "true");
  if (params.moods_k != null) usp.set("moods_k", String(params.moods_k));
  if (params.mood_rail_n != null) usp.set("mood_rail_n", String(params.mood_rail_n));
  return fetchJson<FeedResponse>(`/feed/home?${usp.toString()}`);
}

export function getTagsFilters(params: { limit?: number; debug?: boolean }): Promise<TagsFiltersResponse> {
  const usp = new URLSearchParams();
  if (params.limit != null) usp.set("limit", String(params.limit));
  if (params.debug) usp.set("debug", "true");
  return fetchJson<TagsFiltersResponse>(`/tags/filters?${usp.toString()}`);
}

export function getMoodFeed(params: {
  country: string;
  mood: string;
  n?: number;
  debug?: boolean;
  fallback?: boolean;
  include_tags?: boolean;
}): Promise<FeedResponse & { mood?: string; sections: { mood: TrackItem[] } }> {
  const usp = new URLSearchParams();
  usp.set("country", params.country);
  usp.set("mood", params.mood);
  if (params.n != null) usp.set("n", String(params.n));
  if (params.debug) usp.set("debug", "true");
  if (params.fallback != null) usp.set("fallback", params.fallback ? "true" : "false");
  if (params.include_tags) usp.set("include_tags", "true");
  return fetchJson<any>(`/feed/mood?${usp.toString()}`);
}