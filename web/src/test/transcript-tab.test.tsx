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
      { name: "large-v3", kind: "transcription", installed: true, path: "/models/asr/large-v3/large-v3.pt", selected: true, size_mb: 3000 },
      { name: "sensevoice-small", kind: "transcription", installed: false, path: null, selected: false, size_mb: 936 },
      { name: "cam++", kind: "diarization", installed: false, path: null, selected: false, size_mb: 28 },
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
    model: "large-v3",
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
    expect(await screen.findByRole("option", { name: "large-v3" })).toBeInTheDocument();
    expect(await screen.findByText(/\/models\/asr\/large-v3\/large-v3\.pt/i)).toBeInTheDocument();
  });

  test("shows download button for missing model", async () => {
    renderTab();

    const buttons = await screen.findAllByRole("button", { name: /^download$/i });
    expect(buttons.length).toBeGreaterThan(0);
  });

  test("shows impact messaging for cuda and missing models", async () => {
    renderTab();

    expect(screen.getByText(/cuda is unavailable, so transcription will run on cpu/i)).toBeInTheDocument();
    expect(
      await screen.findByText(/at least one transcription model must be downloaded/i),
    ).toBeInTheDocument();
  });

  test("model dropdown only lists installed transcription models", async () => {
    renderTab();

    expect(await screen.findByRole("option", { name: "large-v3" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /sensevoice-small/i })).not.toBeInTheDocument();
  });

  test("diarization row shows auto-install hint instead of download button", async () => {
    renderTab();

    expect(await screen.findByText(/auto-installs on first ingest/i)).toBeInTheDocument();
    const camButtons = screen.queryAllByRole("button", { name: /download/i });
    expect(camButtons.length).toBe(1);
  });

  test("renders bundle sizes in download table", async () => {
    renderTab();

    expect(await screen.findByText("3.0 GB")).toBeInTheDocument();
    expect(await screen.findByText("936 MB")).toBeInTheDocument();
    expect(await screen.findByText("28 MB")).toBeInTheDocument();
  });

  test("disables cuda when health reports it unavailable", () => {
    renderTab();

    expect(screen.getByRole("option", { name: "CUDA" })).toBeDisabled();
  });

  test("shows inline progress for a tracked model download job", async () => {
    vi.mocked(api.downloadAsrModel).mockResolvedValue({
      job_id: "job-download",
      status: "queued",
      model_name: "sensevoice-small",
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
          meta: { model_name: "sensevoice-small" },
        },
      ]);

    renderTab();

    const downloadButtons = await screen.findAllByRole("button", { name: /^download$/i });
    await userEvent.click(downloadButtons[0]);

    await waitFor(() => {
      expect(screen.getByRole("status", { name: /downloading sensevoice-small/i })).toBeInTheDocument();
    });
  });
});
