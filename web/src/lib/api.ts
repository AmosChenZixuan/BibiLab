import type {
  HealthResponse,
  Job,
  BibilabConfig,
  BibilabList,
  BibilabListPatch,
  OverviewDownload,
  Source,
  SourceContent,
  WhisperDownloadResponse,
  WhisperModel,
} from "./types";

type ApiErrorDetail = string | { message?: string };

export const HEALTH_REFRESH_EVENT = "bibilab:health:refresh";

export class ApiError extends Error {
  detail: ApiErrorDetail;
  status: number;

  constructor(status: number, detail: ApiErrorDetail) {
    super(typeof detail === "string" ? detail : detail.message ?? "Request failed");
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit & { signal?: AbortSignal }): Promise<T | undefined> {
  const response = await fetch(`/api${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: ApiErrorDetail } | null;
    throw new ApiError(response.status, body?.detail ?? (response.statusText || "Request failed"));
  }

  if (response.status === 204) {
    return undefined;
  }

  return response.json() as Promise<T>;
}

export function toErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401 && typeof error.detail !== "string") {
      return "error.401";
    }
    return "error.apiError";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "error.requestFailed";
}

export function toErrorMessageWithT(error: unknown, t: (key: string) => string): string {
  const key = toErrorMessage(error);
  return t(key);
}

export function notifyHealthChanged(health: HealthResponse) {
  window.dispatchEvent(new CustomEvent<HealthResponse>(HEALTH_REFRESH_EVENT, { detail: health }));
}

export const api = {
  listLists: (opts?: { signal?: AbortSignal }) => request<BibilabList[]>("/lists", opts),
  createList: (name: string) =>
    request<BibilabList>("/lists", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  updateList: (listId: string, patch: BibilabListPatch) =>
    request<BibilabList>(`/lists/${listId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  deleteList: (listId: string) =>
    request<void>(`/lists/${listId}`, {
      method: "DELETE",
    }),
  listSources: (listId: string, opts?: { signal?: AbortSignal }) => request<Source[]>(`/lists/${listId}/sources`, opts),
  getSource: (sourceId: string) => request<SourceContent>(`/sources/${sourceId}`),
  deleteSource: (listId: string, sourceId: string) =>
    request<void>(`/lists/${listId}/sources/${sourceId}`, {
      method: "DELETE",
    }),
  ingestUrl: (listId: string, url: string) =>
    request<{ queued: string[]; skipped: string[] }>(
      "/ingest/url",
      {
        method: "POST",
        body: JSON.stringify({ list_id: listId, url }),
      },
    ),
  rerunDigest: (sourceId: string) =>
    request<SourceContent>(`/sources/${sourceId}/rerun`, { method: "POST" }),
  generateOverview: (listId: string) =>
    request<OverviewDownload>(`/lists/${listId}/overview`, {
      method: "POST",
    }),
  getConfig: (opts?: { signal?: AbortSignal }) => request<BibilabConfig>("/config", opts),
  putConfig: (patch: Partial<BibilabConfig>) =>
    request<BibilabConfig>("/config", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
  getHealth: (opts?: { signal?: AbortSignal }) => request<HealthResponse>("/health", opts),
  listJobs: (opts?: { signal?: AbortSignal }) => request<Job[]>("/jobs", opts),
  deleteJob: (jobId: string) =>
    request<void>(`/jobs/${jobId}`, {
      method: "DELETE",
    }),
  listWhisperModels: (opts?: { signal?: AbortSignal }) => request<WhisperModel[]>("/models/whisper", opts),
  downloadWhisperModel: (modelSize: string) =>
    request<WhisperDownloadResponse>("/models/whisper/download", {
      method: "POST",
      body: JSON.stringify({ model_size: modelSize }),
    }),
};
