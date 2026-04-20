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
  ArtifactType,
  Artifact,
  PreviewResponse,
  IngestVideoIn,
  IngestResult,
  VideoMetadataMap,
} from "./types";
import type { Lang } from "@/app/LanguageContext";

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

export function toErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401 && typeof error.detail !== "string") {
      return "errors.401";
    }
    if (typeof error.detail === "string" && error.detail.length > 0) {
      return error.detail;
    }
    if (typeof error.detail === "object" && error.detail?.message) {
      return error.detail.message;
    }
    return "errors.apiError";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "errors.requestFailed";
}

export function toErrorMessageWithT(error: unknown, t: (key: string) => string): string {
  const key = toErrorMessage(error);
  return t(key);
}

export type HealthRefreshDetail = HealthResponse & { device?: string };

export function notifyHealthChanged(health: HealthResponse, device?: string) {
  window.dispatchEvent(new CustomEvent<HealthRefreshDetail>(HEALTH_REFRESH_EVENT, { detail: { ...health, device } }));
}

export const BILIBILI_AUTH_REFRESH_EVENT = "bibilab:auth:bilibili:refresh";

export function notifyBilibiliAuthChanged() {
  window.dispatchEvent(new CustomEvent(BILIBILI_AUTH_REFRESH_EVENT));
}

// ─── Current UI language ──────────────────────────────────────────────────────

let _currentLang: Lang = (localStorage.getItem("bibilab-lang") as Lang | null) ?? "en";

export function setCurrentLang(lang: Lang) {
  _currentLang = lang;
}

// ─── Request helper ───────────────────────────────────────────────────────────

type RequestFn = <T>(
  baseUrl: string,
  path: string,
  init?: RequestInit & { signal?: AbortSignal; responseType?: "json" | "text" },
) => Promise<T | undefined>;

async function request<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit & { signal?: AbortSignal; responseType?: "json" | "text" },
): Promise<T | undefined> {
  const { responseType, ...fetchInit } = init ?? {};
  const response = await fetch(`${baseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      "X-UI-Lang": _currentLang,
      ...(init?.headers ?? {}),
    },
    ...fetchInit,
  });

  if (!response.ok) {
    let detail: ApiErrorDetail = response.statusText || "Request failed";
    try {
      const body = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = (body as { detail: ApiErrorDetail }).detail;
      }
    } catch {
      // non-JSON error body — use statusText fallback already set above
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined;
  }

  if (responseType === "text") {
    const text = await response.text();
    return text as unknown as T;
  }

  const data = await response.json();
  return data as T;
}

// ─── Focused client classes ───────────────────────────────────────────────────

export class ListsClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  listLists(opts?: { signal?: AbortSignal }) {
    return this.request<BibilabList[]>(this.baseUrl, "/lists", opts);
  }

  createList(name: string) {
    return this.request<BibilabList>(this.baseUrl, "/lists", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
  }

  updateList(listId: string, patch: BibilabListPatch) {
    return this.request<BibilabList>(this.baseUrl, `/lists/${listId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
  }

  deleteList(listId: string) {
    return this.request<void>(this.baseUrl, `/lists/${listId}`, { method: "DELETE" });
  }

  generateOverview(listId: string) {
    return this.request<OverviewDownload>(this.baseUrl, `/lists/${listId}/overview`, { method: "POST" });
  }

  createArtifact(listId: string, body: { type: ArtifactType; prompt: string; source_ids: string[] }): Promise<Job> {
    return this.request<Job>(this.baseUrl, `/lists/${listId}/artifacts`, {
      method: "POST",
      body: JSON.stringify(body),
    }) as Promise<Job>;
  }
}

export class SourcesClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  listSources(listId: string, opts?: { signal?: AbortSignal }) {
    return this.request<Source[]>(this.baseUrl, `/lists/${listId}/sources`, opts);
  }

  getSource(sourceId: string, opts?: { signal?: AbortSignal }) {
    return this.request<SourceContent>(this.baseUrl, `/sources/${sourceId}`, opts);
  }

  deleteSource(listId: string, sourceId: string) {
    return this.request<void>(this.baseUrl, `/lists/${listId}/sources/${sourceId}`, { method: "DELETE" });
  }

  rerunDigest(sourceId: string) {
    return this.request<SourceContent>(this.baseUrl, `/sources/${sourceId}/rerun`, { method: "POST" });
  }
}

export class IngestClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  previewPlaylist(listId: string, url: string): Promise<PreviewResponse | undefined> {
    return this.request<PreviewResponse>(this.baseUrl, "/ingest/preview", {
      method: "POST",
      body: JSON.stringify({ list_id: listId, url }),
    });
  }

  previewPlaylistMetadata(videoIds: string[]): Promise<VideoMetadataMap | undefined> {
    return this.request<VideoMetadataMap>(this.baseUrl, "/ingest/preview/metadata", {
      method: "POST",
      body: JSON.stringify({ video_ids: videoIds }),
    });
  }

  ingestUrl(listId: string, videos: IngestVideoIn[]): Promise<IngestResult | undefined> {
    return this.request<IngestResult>(this.baseUrl, "/ingest/url", {
      method: "POST",
      body: JSON.stringify({ list_id: listId, videos }),
    });
  }
}

export class ArtifactsClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  listArtifacts(listId: string, opts?: { signal?: AbortSignal }) {
    return this.request<Artifact[]>(this.baseUrl, `/lists/${listId}/artifacts`, opts);
  }

  async getArtifactContent(artifactId: string, opts?: { signal?: AbortSignal }): Promise<{ content: string } | undefined> {
    const text = await this.request<string>(this.baseUrl, `/artifacts/${artifactId}/content`, {
      method: "GET",
      responseType: "text",
      ...opts,
    });
    if (text === undefined) return undefined;
    return { content: text };
  }

  updateArtifact(artifactId: string, patch: { name?: string }) {
    return this.request<Artifact>(this.baseUrl, `/artifacts/${artifactId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
  }

  deleteArtifact(artifactId: string) {
    return this.request<void>(this.baseUrl, `/artifacts/${artifactId}`, { method: "DELETE" });
  }
}

export class ConfigClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  getConfig(opts?: { signal?: AbortSignal }) {
    return this.request<BibilabConfig>(this.baseUrl, "/config", opts);
  }

  putConfig(patch: Partial<BibilabConfig>) {
    return this.request<BibilabConfig>(this.baseUrl, "/config", {
      method: "PUT",
      body: JSON.stringify(patch),
    });
  }
}

export class HealthClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  getHealth(opts?: { signal?: AbortSignal }) {
    return this.request<HealthResponse>(this.baseUrl, "/health", opts);
  }
}

export class JobsClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  listJobs(opts?: { signal?: AbortSignal }) {
    return this.request<Job[]>(this.baseUrl, "/jobs", opts);
  }

  deleteJob(jobId: string) {
    return this.request<void>(this.baseUrl, `/jobs/${jobId}`, { method: "DELETE" });
  }
}

export type BilibiliQrStatus = "waiting" | "scanned" | "expired" | "success";

export class AuthClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  generateBilibiliQr() {
    return this.request<{ url: string; key: string }>(this.baseUrl, "/auth/bilibili/qr", {
      method: "POST",
    });
  }

  pollBilibiliQr(key: string) {
    return this.request<{ status: BilibiliQrStatus }>(this.baseUrl, `/auth/bilibili/qr/status?key=${key}`);
  }

  deleteBilibiliAuth() {
    return this.request<void>(this.baseUrl, "/auth/bilibili", { method: "DELETE" });
  }
}

export class ModelsClient {
  constructor(private readonly baseUrl: string, private readonly request: RequestFn) {}

  listWhisperModels(opts?: { signal?: AbortSignal }) {
    return this.request<WhisperModel[]>(this.baseUrl, "/models/whisper", opts);
  }

  downloadWhisperModel(modelSize: string) {
    return this.request<WhisperDownloadResponse>(this.baseUrl, "/models/whisper/download", {
      method: "POST",
      body: JSON.stringify({ model_size: modelSize }),
    });
  }
}

// ─── ApiClient interface ──────────────────────────────────────────────────────

export interface ApiClient {
  listLists(opts?: { signal?: AbortSignal }): Promise<BibilabList[] | undefined>;
  createList(name: string): Promise<BibilabList | undefined>;
  updateList(listId: string, patch: BibilabListPatch): Promise<BibilabList | undefined>;
  deleteList(listId: string): Promise<void | undefined>;
  generateOverview(listId: string): Promise<OverviewDownload | undefined>;
  createArtifact(listId: string, body: { type: ArtifactType; prompt: string; source_ids: string[] }): Promise<Job>;
  listSources(listId: string, opts?: { signal?: AbortSignal }): Promise<Source[] | undefined>;
  getSource(sourceId: string, opts?: { signal?: AbortSignal }): Promise<SourceContent | undefined>;
  deleteSource(listId: string, sourceId: string): Promise<void | undefined>;
  rerunDigest(sourceId: string): Promise<SourceContent | undefined>;
  previewPlaylist(listId: string, url: string): Promise<PreviewResponse | undefined>;
  previewPlaylistMetadata(videoIds: string[]): Promise<VideoMetadataMap | undefined>;
  ingestUrl(listId: string, videos: IngestVideoIn[]): Promise<IngestResult | undefined>;
  listArtifacts(listId: string, opts?: { signal?: AbortSignal }): Promise<Artifact[] | undefined>;
  getArtifactContent(artifactId: string, opts?: { signal?: AbortSignal }): Promise<{ content: string } | undefined>;
  updateArtifact(artifactId: string, patch: { name?: string }): Promise<Artifact | undefined>;
  deleteArtifact(artifactId: string): Promise<void | undefined>;
  getConfig(opts?: { signal?: AbortSignal }): Promise<BibilabConfig | undefined>;
  putConfig(patch: Partial<BibilabConfig>): Promise<BibilabConfig | undefined>;
  getHealth(opts?: { signal?: AbortSignal }): Promise<HealthResponse | undefined>;
  listJobs(opts?: { signal?: AbortSignal }): Promise<Job[] | undefined>;
  deleteJob(jobId: string): Promise<void | undefined>;
  listWhisperModels(opts?: { signal?: AbortSignal }): Promise<WhisperModel[] | undefined>;
  downloadWhisperModel(modelSize: string): Promise<WhisperDownloadResponse | undefined>;
  auth: {
    generateBilibiliQr(): Promise<{ url: string; key: string } | undefined>;
    pollBilibiliQr(key: string): Promise<{ status: BilibiliQrStatus } | undefined>;
    deleteBilibiliAuth(): Promise<void | undefined>;
  };
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/**
 * Creates an ApiClient instance.
 *
 * In the browser, baseUrl defaults to window.location.origin + '/api'.
 * Pass a custom baseUrl for testing or custom deployments.
 */
export function createApiClient(baseUrl?: string): ApiClient {
  const base = baseUrl ?? `${window.location.origin}/api`;

  const lists = new ListsClient(base, request);
  const sources = new SourcesClient(base, request);
  const artifacts = new ArtifactsClient(base, request);
  const config = new ConfigClient(base, request);
  const health = new HealthClient(base, request);
  const jobs = new JobsClient(base, request);
  const models = new ModelsClient(base, request);
  const ingest = new IngestClient(base, request);
  const auth = new AuthClient(base, request);

  // Flat ApiClient surface — delegates to focused clients
  return {
    listLists: (opts) => lists.listLists(opts),
    createList: (name) => lists.createList(name),
    updateList: (id, patch) => lists.updateList(id, patch),
    deleteList: (id) => lists.deleteList(id),
    generateOverview: (id) => lists.generateOverview(id),
    createArtifact: (id, body) => lists.createArtifact(id, body),
    listSources: (id, opts) => sources.listSources(id, opts),
    getSource: (id, opts) => sources.getSource(id, opts),
    deleteSource: (listId, sourceId) => sources.deleteSource(listId, sourceId),
    rerunDigest: (id) => sources.rerunDigest(id),
    previewPlaylist: (listId, url) => ingest.previewPlaylist(listId, url),
    previewPlaylistMetadata: (videoIds) => ingest.previewPlaylistMetadata(videoIds),
    ingestUrl: (listId, videos) => ingest.ingestUrl(listId, videos),
    listArtifacts: (id, opts) => artifacts.listArtifacts(id, opts),
    getArtifactContent: (id, opts) => artifacts.getArtifactContent(id, opts),
    updateArtifact: (id, patch) => artifacts.updateArtifact(id, patch),
    deleteArtifact: (id) => artifacts.deleteArtifact(id),
    getConfig: (opts) => config.getConfig(opts),
    putConfig: (patch) => config.putConfig(patch),
    getHealth: (opts) => health.getHealth(opts),
    listJobs: (opts) => jobs.listJobs(opts),
    deleteJob: (id) => jobs.deleteJob(id),
    listWhisperModels: (opts) => models.listWhisperModels(opts),
    downloadWhisperModel: (modelSize) => models.downloadWhisperModel(modelSize),
    auth,
  };
}

// ─── Default singleton (backward-compatible) ─────────────────────────────────

export const api: ApiClient = createApiClient();
