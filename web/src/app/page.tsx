import Link from "next/link";
import { getHomeFeed, getTagsFilters, TrackItem } from "@/lib/backend";

function TrackRow({ items }: { items: TrackItem[] }) {
  return (
    <div className="flex gap-3 overflow-x-auto py-2">
      {items.map((it) => (
        <div
          key={it.track_id}
          className="min-w-[260px] rounded-xl border border-neutral-200 bg-white p-4"
        >
          <div className="text-sm text-neutral-500">{it.genre ?? "—"}</div>
          <div className="mt-1 font-semibold">{it.track_name ?? it.track_id}</div>
          <div className="text-sm text-neutral-600">{it.artist_name ?? "Unknown artist"}</div>
          <div className="mt-3 text-xs text-neutral-500">
            popularity: {it.popularity ?? "—"} · streams: {it.stream_count ?? "—"}
          </div>
        </div>
      ))}
    </div>
  );
}

function Section({ title, items }: { title: string; items: TrackItem[] }) {
  if (!items || items.length === 0) return null;
  return (
    <section className="mt-8">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      <TrackRow items={items} />
    </section>
  );
}

export default async function HomePage() {
  const country = "Brazil";

  const [feed, filters] = await Promise.all([
    getHomeFeed({
      country,
      n: 50,
      debug: false,
      include_tags: false,
      mood_rails: true,
      moods_k: 2,
      mood_rail_n: 5,
    }),
    getTagsFilters({ limit: 10, debug: false }),
  ]);

  const sections = feed.sections ?? {};
  const moodRails = Object.keys(sections).filter((k) => k.startsWith("mood__"));

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold">MusicRec — Home</h1>
        <div className="text-sm text-neutral-600">
          Country: <span className="font-medium">{country}</span>
        </div>
      </div>

      {/* Mood chips (from /tags/filters) */}
      <section className="mt-6 rounded-xl border border-neutral-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">Explore by mood</h2>
          <div className="text-xs text-neutral-500">Source: /tags/filters</div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {filters.moods?.slice(0, 10).map((m) => (
            <Link
              key={m.key}
              href={`/mood/${encodeURIComponent(m.key)}?country=${encodeURIComponent(country)}`}
              className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-sm hover:bg-neutral-100"
            >
              {m.label} <span className="text-neutral-500">({m.count})</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Core rails (excluding mood__ rails) */}
      {Object.entries(sections)
        .filter(([k]) => !k.startsWith("mood__"))
        .map(([k, items]) => (
          <Section key={k} title={k.replaceAll("_", " ")} items={items} />
        ))}

      {/* Mood rails injected into home */}
      {moodRails.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">Mood rails</h2>
          <div className="text-sm text-neutral-600">
            These are injected via <code>mood_rails=true</code> on <code>/feed/home</code>.
          </div>

          {moodRails.map((k) => {
            const mood = k.replace("mood__", "");
            return (
              <section key={k} className="mt-6">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">{mood}</h3>
                  <Link
                    className="text-sm underline"
                    href={`/mood/${encodeURIComponent(mood)}?country=${encodeURIComponent(country)}`}
                  >
                    open mood page
                  </Link>
                </div>
                <TrackRow items={sections[k] ?? []} />
              </section>
            );
          })}
        </section>
      )}
    </main>
  );
}