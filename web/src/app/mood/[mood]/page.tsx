import { getMoodFeed } from "@/lib/backend";

export default async function MoodPage({
  params,
  searchParams,
}: {
  params: { mood: string };
  searchParams: { country?: string };
}) {
  const country = searchParams.country ?? "Brazil";
  const mood = decodeURIComponent(params.mood);

  const d = await getMoodFeed({
    country,
    mood,
    n: 20,
    debug: false,
    fallback: true,
    include_tags: false,
  });

  const items = d.sections?.mood ?? [];

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <h1 className="text-2xl font-bold">Mood: {mood}</h1>
      <div className="mt-1 text-sm text-neutral-600">Country: {country}</div>

      <div className="mt-6 grid grid-cols-1 gap-3 md:grid-cols-2">
        {items.map((it) => (
          <div key={it.track_id} className="rounded-xl border border-neutral-200 bg-white p-4">
            <div className="text-sm text-neutral-500">{it.genre ?? "—"}</div>
            <div className="mt-1 font-semibold">{it.track_name ?? it.track_id}</div>
            <div className="text-sm text-neutral-600">{it.artist_name ?? "Unknown artist"}</div>
          </div>
        ))}
      </div>

      {items.length === 0 && (
        <div className="mt-8 rounded-xl border border-neutral-200 bg-white p-4 text-neutral-700">
          No results returned (fallback may be off or tag join is empty for this mood).
        </div>
      )}
    </main>
  );
}