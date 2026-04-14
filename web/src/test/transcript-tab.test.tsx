import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { TranscriptTab } from "@/components/settings/TranscriptTab";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";

vi.mock("../lib/api", () => {
  const mockApi = {
    listWhisperModels: vi.fn().mockResolvedValue([
      { name: "base", installed: true, path: "/models/base", selected: true },
      { name: "large-v3", installed: false, path: null, selected: false },
    ]),
    downloadWhisperModel: vi.fn(),
    listJobs: vi.fn().mockResolvedValue([]),
  };
  return {
    createApiClient: () => mockApi,
    api: mockApi,
    setCurrentLang: vi.fn(),
  };
});

import { api } from "@/lib/api";

const baseConfig: BibilabConfig = {
  accounts: { bilibili: { cookie: "", last_verified: "" } },
  ai: { provider: "openai", model: "", api_key: "", base_url: "" },
  transcription: {
    engine: "faster-whisper",
    model_size: "base",
    device: "cpu",
    language: "auto",
  },
  vision: { enabled: false, model: "", frame_sample_rate: 60 },
  backend: { port: 8765, worker_concurrency: 2 },
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const healthDeps: Record<string, HealthDependency> = {
  cuda: { status: "unavailable", message: "CUDA not available; CPU will be used" },
};

function renderTab() {
  return render(
    <JobActivityProvider>
      <LanguageProvider>
        <TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />
      </LanguageProvider>
    </JobActivityProvider>,
  );
}

describe("transcript tab", () => {
  test("renders transcription device dropdown", () => {
    renderTab();

    expect(screen.getByLabelText(/device/i)).toBeInTheDocument();
  });

  test("shows installed path for downloaded model", async () => {
    renderTab();

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "base" })).toBeInTheDocument();
    expect(await screen.findByText(/\/models\/base/i)).toBeInTheDocument();
  });

  test("shows download button for missing model", async () => {
    renderTab();

    expect(await screen.findByRole("button", { name: /^download$/i })).toBeInTheDocument();
  });

  test("shows impact messaging for cuda and missing whisper models", async () => {
    renderTab();

    expect(screen.getByText(/cuda not available; cpu will be used/i)).toBeInTheDocument();
    expect(await screen.findByText(/transcription cannot start until a model is downloaded/i)).toBeInTheDocument();
  });

  test("model size dropdown only lists downloaded models", async () => {
    renderTab();

    expect(await screen.findByRole("option", { name: "base" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "large-v3" })).not.toBeInTheDocument();
  });

  test("download table does not render status chips", async () => {
    renderTab();

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.queryByText(/selected/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^installed$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^missing$/i)).not.toBeInTheDocument();
  });

  test("disables cuda when health reports it unavailable", () => {
    renderTab();

    expect(screen.getByRole("option", { name: "CUDA" })).toBeDisabled();
  });

  test("shows inline progress for a tracked whisper download job", async () => {
    vi.mocked(api.downloadWhisperModel).mockResolvedValue({
      job_id: "job-download",
      status: "queued",
      model_family: "whisper",
      model_size: "large-v3",
    });
    vi.mocked(api.listJobs)
      .mockResolvedValueOnce([])
      .mockResolvedValue([
        {
          id: "job-download",
          type: "model_download",
          status: "downloading",
          progress: 32,
          error: null,
          created_at: "2026-03-31T20:00:00Z",
          updated_at: "2026-03-31T20:01:00Z",
          meta: { model_family: "whisper", model_size: "large-v3" },
        },
      ]);

    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: /^download$/i }));

    await waitFor(() => {
      expect(screen.getByRole("status", { name: /downloading large-v3/i })).toBeInTheDocument();
    });
  });
});
