import type { HealthDependency, HealthResponse } from "./types";

export type HealthTier = "operational" | "degraded" | "unavailable";

export const HEALTH_META: Record<HealthTier, { label: string; className: string }> = {
  operational: { label: "Operational", className: "bg-[#4ade80]" },
  degraded: { label: "Degraded", className: "bg-[#fbbf24]" },
  unavailable: { label: "Unavailable", className: "bg-[#f87171]" },
};

export function deriveOverallHealthTier(health: HealthResponse): HealthTier {
  if (health.overall === "error") {
    return "unavailable";
  }

  if (
    health.dependencies.cuda?.status !== "ok" ||
    health.dependencies.embedding_model?.status !== "ok"
  ) {
    return "degraded";
  }

  return "operational";
}

export function deriveDependencyHealthTier(
  dependencies: Record<string, HealthDependency>,
  dependencyKeys: string[],
): HealthTier {
  const statuses = dependencyKeys.map((key) => dependencies[key]?.status).filter(Boolean);

  if (statuses.includes("error")) {
    return "unavailable";
  }

  if (statuses.some((status) => status !== "ok")) {
    return "degraded";
  }

  return "operational";
}
