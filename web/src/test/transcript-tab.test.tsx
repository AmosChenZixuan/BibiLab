import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import { LanguageProvider } from "@/app/LanguageContext";
import { TranscriptTab } from "@/components/settings/TranscriptTab";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";

vi.mock("../lib/api", () => {
  const mockApi = {
    listAsrModels: vi.fn().mockResolvedValue([
      { name: "large-v3", display_name: "Faster Whisper large-v3", kind: "transcription", installed: true, path: "/models/asr/large-v3/large-v3.pt", selected: true, size_mb: 3000 },
      { name: "sensevoice-small", display_name: "SenseVoice Small", kind: "transcription", installed: false, path: null, selected: false, size_mb: 936 },
      { name: "cam++", display_name: "CAM++ (Speaker Diarization)", kind: "diarization", installed: false, path: null, selected: false, size_mb: 28 },
    ]),
    listJobs: vi.fn().mockResolvedValue([]),
  };
  return {
    createApiClient: () => mockApi,
    api: mockApi,
    setCurrentLang: vi.fn(),
  };
});

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

function renderTab(config: BibilabConfig = baseConfig) {
  return render(
    <MemoryRouter>
      <JobActivityProvider>
        <LanguageProvider>
          <TranscriptTab config={config} dependencies={healthDeps} onBlur={() => {}} />
        </LanguageProvider>
      </JobActivityProvider>
    </MemoryRouter>,
  );
}

describe("transcript tab", () => {
  test("renders transcription device dropdown", () => {
    renderTab();

    expect(screen.getByLabelText(/device/i)).toBeInTheDocument();
  });

  test("model dropdown only lists installed transcription models", async () => {
    renderTab();

    expect(await screen.findByRole("option", { name: /Faster Whisper large-v3/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /sensevoice-small/i })).not.toBeInTheDocument();
  });

  test("does not surface diarization row (moved to Local Models tab)", () => {
    renderTab();

    expect(screen.queryByText(/speaker diarization/i)).not.toBeInTheDocument();
  });

  test("disables cuda when health reports it unavailable", () => {
    renderTab();

    expect(screen.getByRole("option", { name: "CUDA" })).toBeDisabled();
  });

  test("shows impact messaging for cuda", () => {
    renderTab();

    expect(screen.getByText(/cuda is unavailable, so transcription will run on cpu/i)).toBeInTheDocument();
  });

  test("renames language label to transcription language", () => {
    renderTab();

    expect(screen.getByLabelText(/transcription language/i)).toBeInTheDocument();
  });

  test("does not render an inline model download table", async () => {
    renderTab();

    expect(await screen.findByRole("option", { name: /Faster Whisper large-v3/i })).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^download$/i })).not.toBeInTheDocument();
  });
});
