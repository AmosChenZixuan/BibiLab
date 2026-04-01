export type LocusList = {
  id: string;
  name: string;
  created_at: string;
};

export type Source = {
  video_id: string;
  platform: string;
  title: string;
  note_path: string;
  processed_at: string;
};

export type Job = {
  id: string;
  type: string;
  source_url: string;
  platform: string;
  status: string;
  progress: number;
  error: string | null;
  created_at: string;
  updated_at: string;
  meta: Record<string, unknown>;
};

export type LocusConfig = {
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
    base_url: string | null;
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
