import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { TranscriptTab } from "@/components/settings/TranscriptTab";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";

vi.mock("../lib/api", () => {
  const mockApi = {
    listAsrModels: vi.fn().mockResolvedValue([
      { name: "medium", engine: "whisper", installed: true, path: "/models/whisper/medium", selected: false },
      { name: "large-v3", engine: "whisper", installed: true, path: "/models/whisper/large-v3", selected: true },
      { name: "small", engine: "sensevoice", installed: false, path: null, selected: false },
      { name: "cam++", engine: "diarization", installed: false, path: null, selected: false },
    ]),
    downloadAsrModel: vi.fn(),
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
  accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
  ai: { protocol: "openai", model: "", api_key: "", base_url: "" },
  transcription: {
    engine: "whisper",
    model_size: "medium",
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
    expect(await screen.findByRole("option", { name: "medium" })).toBeInTheDocument();
    expect(await screen.findByText(/\/models\/whisper\/medium/i)).toBeInTheDocument();
  });

  test("shows download button for missing model", async () => {
    renderTab();

    const buttons = await screen.findAllByRole("button", { name: /^download$/i });
    expect(buttons.length).toBeGreaterThan(0);
  });

  test("shows impact messaging for cuda and missing whisper models", async () => {
    renderTab();

    expect(screen.getByText(/cuda is unavailable, so transcription will run on cpu/i)).toBeInTheDocument();
    expect(await screen.findByText(/transcription cannot start until a model is downloaded/i)).toBeInTheDocument();
  });

  test("model size dropdown only lists downloaded models", async () => {
    renderTab();

    expect(await screen.findByRole("option", { name: "medium" })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "large-v3" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "small" })).not.toBeInTheDocument();
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

  test("shows inline progress for a tracked model download job", async () => {
    vi.mocked(api.downloadAsrModel).mockResolvedValue({
      job_id: "job-download",
      status: "queued",
      engine: "sensevoice",
      model_size: "small",
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
          meta: { engine: "sensevoice", model_size: "small" },
        },
      ]);

    renderTab();

    const downloadButtons = await screen.findAllByRole("button", { name: /^download$/i });
    await userEvent.click(downloadButtons[0]);

    await waitFor(() => {
      expect(screen.getByRole("status", { name: /downloading small/i })).toBeInTheDocument();
    });
  });
});
