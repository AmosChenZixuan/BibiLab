export type OutputLanguage = "ui" | "en" | "zh";

export type BibilabList = {
  id: string;
  name: string;
  created_at: string;
  thumbnail_source_id: string | null;
  thumbnail_url: string | null;
  source_count: number;
  updated_at: string;
};

export type BibilabListPatch = {
  name?: string;
  thumbnail_source_id?: string | null;
};

export type Source = {
  id: string;
  video_id: string;
  platform: string;
  title: string;
  summary: string;
  keywords: string[];
  cover_url: string | null;
  source_url: string;
  duration_seconds: number;
  uploader: string;
  language: string | null;
  processed_at: string;
};

export type SourceContent = {
  id: string;
  video_id: string;
  platform: string;
  title: string;
  source_url: string;
  duration_seconds: number;
  uploader: string;
  language: string | null;
  processed_at: string;
  summary: string;
  keywords: string[];
  cover_url: string | null;
  transcript: string;
  settings_snapshot: Record<string, unknown>;
  series_name?: string | null;
  sequence_number?: number | null;
  season_number?: number | null;
};

export type SourceFacetsPatch = {
  series_name?: string | null;
  sequence_number?: number | null;
  season_number?: number | null;
};

type JobStatus =
  | "queued"
  | "downloading"
  | "transcribing"
  | "processing"
  | "done"
  | "failed"
  | "needs_auth";

type BaseJob = {
  id: string;
  status: JobStatus;
  progress: number;
  error: string | null;
  created_at: string;
  updated_at: string;
};

type IngestMeta = {
  list_id?: string;
  source_url?: string;
  platform?: string;
  video_id?: string;
  title?: string;
  cover_url?: string;
  duration_seconds?: number;
  uploader?: string;
};

type ModelDownloadMeta = {
  model_name?: string;
};

export type IngestJob = BaseJob & {
  type: "ingest";
  meta: IngestMeta;
};

export type ModelDownloadJob = BaseJob & {
  type: "model_download";
  meta: ModelDownloadMeta;
};

export type ArtifactJob = {
  id: string;
  type: "artifact";
  status: ArtifactStatus | JobStatus;
  progress: number;
  error: string | null;
  created_at: string;
  updated_at: string;
  meta: {
    list_id?: string;
    artifact_id?: string;
    artifact_type?: string;
  };
};

type DigestMeta = {
  source_id: string;
  list_id?: string;
  source_title?: string;
  ui_lang?: string;
};

export type DigestJob = BaseJob & {
  type: "digest";
  meta: DigestMeta;
};

export type Job = IngestJob | ModelDownloadJob | ArtifactJob | DigestJob;

export type ArtifactType = "brief" | "study_guide" | "blog_post" | "custom_report" | (string & {});

export type ArtifactStatus = "generating" | "completed" | "failed";

export type Artifact = {
  id: string;
  name: string;
  type: ArtifactType;
  prompt: string;
  source_ids: string[];
  status: ArtifactStatus;
  created_at: string;
  error?: string;
};

export type BibilabConfig = {
  accounts: {
    bilibili: {
      cookie: string;
      username: string;
      avatar_url: string;
    };
  };
  ai: {
    protocol: string;
    model: string;
    api_key: string;
    base_url: string;
    output_language?: OutputLanguage;
  };
  transcription: {
    model: string;
    device: string;
    language: string;
  };
  backend: {
    port: number;
    max_concurrent_jobs: number;
    cors_origins: string[];
  };
  rag: {
    max_distance: number;
    reranking_enabled: boolean;
    hybrid_enabled: boolean;
    debug_prompts: boolean;
  };
};

type HealthStatus = "ok" | "error" | "unavailable";

export type HealthDependency = {
  status: HealthStatus;
  message: string;
};

export type HealthResponse = {
  overall: string;
  dependencies: Record<string, HealthDependency>;
};

export type ModelKind = "transcription" | "diarization" | "vad" | "punctuation" | "embedding" | "reranker";

type ModelStatus = "present" | "missing";

export type ModelInfo = {
  id: string;
  display_name: string;
  kind: ModelKind;
  size_mb: number;
  status: ModelStatus;
  required_by_config: boolean;
  path: string | null;
};

export type ModelDownloadResponse = {
  job_id: string;
  status: "queued";
  spec_id: string;
};

export type SyncResponse = {
  job_ids: string[];
  synced: string[];
  skipped: string[];
};

export type VideoStatus = "new" | "processed" | "in_progress" | "needs_auth";

export interface PreviewVideo {
  video_id: string;
  title: string;
  cover_url: string;
  duration_seconds: number;
  uploader: string;
  platform: string;
  source_url: string;
  part_label: string | null;
  status: VideoStatus;
}

export interface PreviewResponse {
  videos: PreviewVideo[];
}

export type IngestVideoIn = Omit<PreviewVideo, "status" | "part_label">;

export interface IngestResult {
  queued: string[];
  skipped: string[];
}

export type Conversation = {
  id: string;
  list_id: string;
  summary: string | null;
  created_at: string;
  updated_at: string;
  active_stream_message_id: string | null;
};

export type Message = {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
  status?: string;
  error?: string | null;
};

export interface VideoMetadataMap {
  videos: Record<string, {
    title: string;
    cover_url: string;
    duration_seconds: number;
    uploader: string;
    source_url: string;
    part_label: string | null;
  }>;
  expanded: Record<string, string[]>;
}
