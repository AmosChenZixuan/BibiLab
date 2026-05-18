import { useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { toErrorMessageWithT } from "@/lib/api";
import type { SourceFacetsPatch } from "@/lib/types";

export type Facets = {
  seriesName: string | null | undefined;
  sequenceNumber: number | null | undefined;
  seasonNumber: number | null | undefined;
};

type Props = {
  facets: Facets;
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

function Field({
  label,
  value,
  numeric,
  onChange,
}: {
  label: string;
  value: string;
  numeric?: boolean;
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      <span className="shrink-0 font-semibold uppercase tracking-wider text-muted/70">{label}</span>
      <input
        aria-label={label}
        inputMode={numeric ? "numeric" : undefined}
        className={`${numeric ? "w-20" : "flex-1"} rounded-md border border-border bg-white px-2 py-1 text-sm text-ink`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function DigestFacets({ facets, editing, onSave, onExitEdit }: Props) {
  const { t } = useLanguage();
  const [series, setSeries] = useState(facets.seriesName ?? "");
  const [num, setNum] = useState(facets.sequenceNumber == null ? "" : String(facets.sequenceNumber));
  const [season, setSeason] = useState(facets.seasonNumber == null ? "" : String(facets.seasonNumber));
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
        await onSave({ series_name: series.trim() || null, sequence_number: n, season_number: s });
        onExitEdit();
      } catch (err) {
        setError(toErrorMessageWithT(err, t));
      } finally {
        setSaving(false);
      }
    };
    return (
      <div
        className={`flex flex-col gap-2 rounded-xl border border-dashed border-blue/40 bg-white/50 p-3 ${
          saving ? "pointer-events-none opacity-40" : ""
        }`}
      >
        <Field label={t("lists.facets.series")} value={series} onChange={setSeries} />
        <div className="flex gap-2">
          <Field label={t("lists.facets.number")} value={num} numeric onChange={setNum} />
          <Field label={t("lists.facets.season")} value={season} numeric onChange={setSeason} />
        </div>
        {error && <p className="m-0 text-xs font-medium text-pink">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            disabled={saving}
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
  if (facets.seriesName != null) segs.push(facets.seriesName);
  if (facets.sequenceNumber != null) segs.push(`${t("lists.facets.numberPrefix")} ${facets.sequenceNumber}`);
  if (facets.seasonNumber != null) segs.push(`${t("lists.facets.seasonPrefix")} ${facets.seasonNumber}`);
  if (segs.length === 0) return null;

  return <p className="m-0 text-xs font-semibold text-ink">{segs.join(" · ")}</p>;
}
