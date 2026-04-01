import type { HealthResponse } from "../../lib/types";

type Props = {
  health: HealthResponse;
  refreshing: boolean;
  onRefresh: () => Promise<void>;
};

export function HealthPanel({ health, refreshing, onRefresh }: Props) {
  return (
    <section>
      <div className="row">
        <div>
          <h2 className="list-card__title">Health</h2>
          <p className="page-lede">Overall status: {health.overall}</p>
        </div>
        <button className="secondary-button" disabled={refreshing} onClick={() => void onRefresh()} type="button">
          {refreshing ? "Refreshing..." : "Refresh"}
        </button>
      </div>
      <div className="dependency-list">
        {Object.entries(health.dependencies).map(([name, dependency]) => (
          <div className="dependency-row" key={name}>
            <div className="row">
              <strong>{name}</strong>
              <span className={`status-chip ${dependency.status}`}>{dependency.status}</span>
            </div>
            {dependency.message ? <p className="page-lede">{dependency.message}</p> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
