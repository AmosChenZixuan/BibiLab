import type { HealthDependency, HealthResponse } from "./types";

export type HealthTier = "healthy" | "throttled" | "unavailable";

export const HEALTH_META: Record<HealthTier, { label: string; className: string }> = {
  healthy: { label: "Healthy", className: "bg-sky-400" },
  throttled: { label: "Throttled", className: "bg-amber-400" },
  unavailable: { label: "Unavailable", className: "bg-rose-500" },
};

export function deriveOverallHealthTier(health: HealthResponse): HealthTier {
  if (health.overall === "error") {
    return "unavailable";
  }

  if (
    health.dependencies.cuda?.status !== "ok" ||
    health.dependencies.embedding_model?.status !== "ok"
  ) {
    return "throttled";
  }

  return "healthy";
}

export function deriveDependencyHealthTier(
  dependencies: Record<string, HealthDependency>,
  dependencyKeys: readonly string[],
): HealthTier {
  const statuses = dependencyKeys.map((key) => dependencies[key]?.status).filter(Boolean);

  if (statuses.includes("error")) {
    return "unavailable";
  }

  if (statuses.some((status) => status !== "ok")) {
    return "throttled";
  }

  return "healthy";
}
