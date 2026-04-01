import type { HealthResponse } from "../../lib/types";
import {
  appPanelClass,
  mutedTextClass,
  secondaryButtonClass,
  sectionTitleClass,
  statusChipClass,
} from "../../lib/ui";

type Props = {
  health: HealthResponse;
  refreshing: boolean;
  onRefresh: () => Promise<void>;
};

export function HealthPanel({ health, refreshing, onRefresh }: Props) {
  return (
    <section className={`${appPanelClass} grid gap-4`}>
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className={sectionTitleClass}>Health</h2>
          <p className={mutedTextClass}>Overall status: {health.overall}</p>
        </div>
        <button className={secondaryButtonClass} disabled={refreshing} onClick={() => void onRefresh()} type="button">
          {refreshing ? "Refreshing..." : "Refresh"}
        </button>
      </div>
      <div className="grid gap-3">
        {Object.entries(health.dependencies).map(([name, dependency]) => (
          <div className="grid gap-2 border-t border-[rgba(106,147,198,0.12)] pt-3" key={name}>
            <div className="flex flex-wrap items-center gap-3">
              <strong>{name}</strong>
              <span className={statusChipClass(dependency.status)}>{dependency.status}</span>
            </div>
            {dependency.message ? <p className={mutedTextClass}>{dependency.message}</p> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
