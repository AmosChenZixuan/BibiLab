import { useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import type { SourceFacetsPatch } from "@/lib/types";

type Props = {
  seriesName: string | null | undefined;
  sequenceNumber: number | null | undefined;
  seasonNumber: number | null | undefined;
  editing: boolean;
  onSave: (patch: SourceFacetsPatch) => Promise<void>;
  onExitEdit: () => void;
};

function parseIntField(raw: string): number | null | undefined {
  const s = raw.trim();
  if (s === "") return null;
  if (!/^\d+$/.test(s)) return undefined;
  const n = Number(s);
  return n >= 1 ? n : undefined;
}

export function DigestFacets({
  seriesName,
  sequenceNumber,
  seasonNumber,
  editing,
  onSave,
  onExitEdit,
}: Props) {
  const { t } = useLanguage();
  const [series, setSeries] = useState(seriesName ?? "");
  const [num, setNum] = useState(sequenceNumber == null ? "" : String(sequenceNumber));
  const [season, setSeason] = useState(seasonNumber == null ? "" : String(seasonNumber));
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  if (editing) {
    const handleSave = async () => {
      const n = parseIntField(num);
      const s = parseIntField(season);
      if (n === undefined || s === undefined) {
        setError(t("lists.facets.invalidInt"));
        return;
      }
      setError(null);
      setSaving(true);
      try {
        await onSave({
          series_name: series.trim() || null,
          sequence_number: n,
          season_number: s,
        });
        onExitEdit();
      } finally {
        setSaving(false);
      }
    };
    return (
      <div className="flex flex-col gap-2 rounded-xl border border-dashed border-blue/40 bg-white/50 p-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-muted/70">
            {t("lists.facets.series")}
          </span>
          <input
            aria-label={t("lists.facets.series")}
            className="flex-1 rounded-md border border-border bg-white px-2 py-1 text-sm text-ink"
            value={series}
            onChange={(e) => setSeries(e.target.value)}
          />
        </label>
        <div className="flex gap-2">
          <label className="flex items-center gap-2 text-xs text-muted">
            <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-muted/70">
              {t("lists.facets.number")}
            </span>
            <input
              aria-label={t("lists.facets.number")}
              inputMode="numeric"
              className="w-20 rounded-md border border-border bg-white px-2 py-1 text-sm text-ink"
              value={num}
              onChange={(e) => setNum(e.target.value)}
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-muted">
            <span className="shrink-0 font-semibold uppercase tracking-wider text-muted/70">
              {t("lists.facets.season")}
            </span>
            <input
              aria-label={t("lists.facets.season")}
              inputMode="numeric"
              className="w-20 rounded-md border border-border bg-white px-2 py-1 text-sm text-ink"
              value={season}
              onChange={(e) => setSeason(e.target.value)}
            />
          </label>
        </div>
        {error && <p className="m-0 text-xs font-medium text-pink">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onExitEdit}
            className="rounded-md border border-border bg-white px-3 py-1 text-sm font-medium text-ink"
          >
            {t("lists.facets.cancel")}
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={handleSave}
            className="rounded-md border border-blue bg-blue px-3 py-1 text-sm font-semibold text-white disabled:opacity-50"
          >
            {t("lists.facets.save")}
          </button>
        </div>
      </div>
    );
  }

  const segs: string[] = [];
  if (seriesName != null) segs.push(seriesName);
  if (sequenceNumber != null) segs.push(`${t("lists.facets.numberPrefix")} ${sequenceNumber}`);
  if (seasonNumber != null) segs.push(`${t("lists.facets.seasonPrefix")} ${seasonNumber}`);
  if (segs.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-2 text-xs text-muted">
      {segs.map((s, i) => (
        <span key={s}>
          {i > 0 && <span className="px-1 text-muted/40">·</span>}
          <span className="font-semibold text-ink">{s}</span>
        </span>
      ))}
    </div>
  );
}
