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
  video_id: string;
  platform: string;
  title: string;
  note_path: string;
  processed_at: string;
};

export type JobStatus =
  | "queued"
  | "downloading"
  | "transcribing"
  | "extracting"
  | "writing"
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

export type IngestMeta = {
  list_id?: string;
  source_url?: string;
  platform?: string;
  video_id?: string;
  title?: string;
  cover_url?: string;
  duration_seconds?: number;
  uploader?: string;
  rerun?: boolean;
};

export type ModelDownloadMeta = {
  model_family?: string;
  model_size?: string;
};

export type IngestJob = BaseJob & {
  type: "ingest";
  meta: IngestMeta;
};

export type ModelDownloadJob = BaseJob & {
  type: "model_download";
  meta: ModelDownloadMeta;
};

export type Job = IngestJob | ModelDownloadJob;

export type BibilabConfig = {
  accounts: {
    bilibili: {
      cookie: string;
      last_verified: string;
    };
  };
  ai: {
    provider: string;
    model: string;
    api_key: string;
    base_url: string;
    output_language?: OutputLanguage;
  };
  transcription: {
    engine: string;
    model_size: string;
    device: string;
    language: string;
  };
  vision: {
    enabled: boolean;
    frame_sample_rate: number;
    model: string | null;
  };
  backend: {
    port: number;
    worker_concurrency: number;
  };
};

export type HealthStatus = "ok" | "error" | "unavailable";

export type HealthDependency = {
  status: HealthStatus;
  message: string;
};

export type HealthResponse = {
  overall: string;
  dependencies: Record<string, HealthDependency>;
};

export type WhisperModel = {
  name: string;
  installed: boolean;
  path: string | null;
  selected: boolean;
};

export type WhisperDownloadResponse = {
  job_id: string;
  status: string;
  model_family: string;
  model_size: string;
};

export type NoteContent = {
  video_id: string;
  title: string;
  markdown: string;
};

export type NoteTranscript = {
  video_id: string;
  text: string;
};

export type OverviewDownload = {
  filename: string;
  content: string;
};
