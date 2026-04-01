import type {
  HealthResponse,
  LocusConfig,
  LocusList,
  Source,
  WhisperDownloadResponse,
  WhisperModel,
} from "./types";

type ApiErrorDetail = string | { message?: string };

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

export const api = {
  listLists: () => request<LocusList[]>("/lists"),
  createList: (name: string) =>
    request<LocusList>("/lists", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  deleteList: (listId: string) =>
    request<void>(`/lists/${listId}`, {
      method: "DELETE",
    }),
  listSources: (listId: string) => request<Source[]>(`/lists/${listId}/sources`),
  getConfig: () => request<LocusConfig>("/config"),
  putConfig: (patch: Partial<LocusConfig>) =>
    request<LocusConfig>("/config", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
  getHealth: () => request<HealthResponse>("/health"),
  listWhisperModels: () => request<WhisperModel[]>("/models/whisper"),
  downloadWhisperModel: (modelSize: string) =>
    request<WhisperDownloadResponse>("/models/whisper/download", {
      method: "POST",
      body: JSON.stringify({ model_size: modelSize }),
    }),
};
