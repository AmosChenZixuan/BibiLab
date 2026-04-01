import type {
  HealthResponse,
  Job,
  LocusConfig,
  LocusList,
  NoteContent,
  NoteTranscript,
  OverviewDownload,
  Source,
  WhisperDownloadResponse,
  WhisperModel,
} from "./types";

type ApiErrorDetail = string | { message?: string };

export const JOBS_REFRESH_EVENT = "locus:jobs:refresh";

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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: ApiErrorDetail };
    throw new ApiError(response.status, body.detail ?? "Request failed");
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function toErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401 && typeof error.detail !== "string") {
      return "Authentication required";
    }
    if (typeof error.detail === "string") {
      return error.detail;
    }
    return error.detail.message ?? "Request failed";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed";
}

export function notifyJobsChanged() {
  window.dispatchEvent(new Event(JOBS_REFRESH_EVENT));
}

export const api = {
  listLists: () => request<LocusList[]>("/lists"),
  createList: (name: string) =>
    request<LocusList>("/lists", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  updateList: (listId: string, name: string) =>
    request<LocusList>(`/lists/${listId}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),
  deleteList: (listId: string) =>
    request<void>(`/lists/${listId}`, {
      method: "DELETE",
    }),
  listSources: (listId: string) => request<Source[]>(`/lists/${listId}/sources`),
  deleteSource: (listId: string, videoId: string) =>
    request<void>(`/lists/${listId}/sources/${videoId}`, {
      method: "DELETE",
    }),
  ingestUrl: (listId: string, url: string, rerun = false) =>
    request<{ queued: string[]; skipped: string[] }>(
      `/ingest/url${rerun ? "?rerun=true" : ""}`,
      {
        method: "POST",
        body: JSON.stringify({ list_id: listId, url }),
      },
    ),
  getNoteContent: (videoId: string) => request<NoteContent>(`/notes/${videoId}/content`),
  getNoteTranscript: (videoId: string) => request<NoteTranscript>(`/notes/${videoId}/transcript`),
  generateOverview: (listId: string) =>
    request<OverviewDownload>(`/lists/${listId}/overview`, {
      method: "POST",
    }),
  getConfig: () => request<LocusConfig>("/config"),
  putConfig: (patch: Partial<LocusConfig>) =>
    request<LocusConfig>("/config", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
  getHealth: () => request<HealthResponse>("/health"),
  listJobs: () => request<Job[]>("/jobs"),
  deleteJob: (jobId: string) =>
    request<void>(`/jobs/${jobId}`, {
      method: "DELETE",
    }),
  listWhisperModels: () => request<WhisperModel[]>("/models/whisper"),
  downloadWhisperModel: (modelSize: string) =>
    request<WhisperDownloadResponse>("/models/whisper/download", {
      method: "POST",
      body: JSON.stringify({ model_size: modelSize }),
    }),
};
