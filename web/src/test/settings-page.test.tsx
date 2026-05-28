import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { SettingsPage } from "@/pages/SettingsPage";

const MOCK_ERROR_MESSAGE = "Request failed";

vi.mock("../lib/api", () => {
  const mockApi = {
    getConfig: vi.fn().mockResolvedValue({
      accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "" },
      transcription: {
        engine: "faster-whisper",
        model_size: "base",
        device: "cpu",
        language: "auto",
      },
      vision: { enabled: false, model: "", frame_sample_rate: 60 },
      backend: { port: 8765, worker_concurrency: 2 },
    }),
    putConfig: vi.fn().mockResolvedValue({}),
    getHealth: vi.fn().mockResolvedValue({
      overall: "ok",
      dependencies: {
        backend: { status: "ok", message: "" },
        llm: { status: "ok", message: "" },
        asr_model: { status: "ok", message: "" },
        ffmpeg: { status: "ok", message: "" },
        cuda: { status: "unavailable", message: "CPU only" },
        embedding_model: { status: "ok", message: "" },
      },
    }),
    listWhisperModels: vi.fn().mockResolvedValue([]),
    downloadWhisperModel: vi.fn(),
  };
  return {
    createApiClient: () => mockApi,
    api: mockApi,
    notifyHealthChanged: vi.fn(),
    toErrorMessageWithT: (error: unknown) => (error instanceof Error ? error.message : MOCK_ERROR_MESSAGE),
    setCurrentLang: vi.fn(),
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderPage() {
  return render(
    <JobActivityProvider>
      <LanguageProvider>
        <MemoryRouter>
          <SettingsPage />
        </MemoryRouter>
      </LanguageProvider>
    </JobActivityProvider>,
  );
}

function renderPageAt(entry: string) {
  function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
  }

  return render(
    <JobActivityProvider>
      <LanguageProvider>
        <MemoryRouter initialEntries={[entry]}>
          <SettingsPage />
          <LocationProbe />
        </MemoryRouter>
      </LanguageProvider>
    </JobActivityProvider>,
  );
}

describe("settings page", () => {
  test("renders three tabs", async () => {
    renderPage();

    expect(await screen.findByRole("tab", { name: /llm/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /transcript/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /other/i })).toBeInTheDocument();
  });

  test("LLM tab is active by default and shows provider dropdown", async () => {
    renderPage();

    expect(await screen.findByLabelText(/provider/i)).toBeInTheDocument();
  });

  test("clicking Transcript tab shows device dropdown", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("tab", { name: /transcript/i }));
    expect(await screen.findByLabelText(/device/i)).toBeInTheDocument();
  });

  test("clicking Other tab shows backend api section", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("tab", { name: /other/i }));
    expect(await screen.findByText(/backend api/i)).toBeInTheDocument();
  });

  test("clicking a tab updates the url query", async () => {
    renderPageAt("/settings");

    fireEvent.click(await screen.findByRole("tab", { name: /other/i }));

    expect(screen.getByTestId("location")).toHaveTextContent("/settings?tab=other");
  });

  test("reads active tab from the url query", async () => {
    renderPageAt("/settings?tab=transcript");

    expect(await screen.findByLabelText(/device/i)).toBeInTheDocument();
    expect(screen.getByTestId("location")).toHaveTextContent("/settings?tab=transcript");
  });

  test("tab health indicators use the same tiers as the navbar health model", async () => {
    renderPage();

    expect(await screen.findByRole("tab", { name: /llm/i })).toHaveAttribute("title", "Healthy");
    expect(screen.getByRole("tab", { name: /transcript/i })).toHaveAttribute("title", "Healthy");
    expect(screen.getByRole("tab", { name: /other/i })).toHaveAttribute("title", "Healthy");
  });

  test("refreshes health after saving config", async () => {
    vi.mocked(api.getHealth)
      .mockResolvedValueOnce({
        overall: "ok",
        dependencies: {
          backend: { status: "ok", message: "" },
          llm: { status: "ok", message: "" },
          asr_model: { status: "ok", message: "" },
          ffmpeg: { status: "ok", message: "" },
          cuda: { status: "unavailable", message: "CPU only" },
          embedding_model: { status: "ok", message: "" },
        },
      })
      .mockResolvedValueOnce({
        overall: "ok",
        dependencies: {
          backend: { status: "ok", message: "" },
          llm: { status: "error", message: "base_url not configured" },
          asr_model: { status: "ok", message: "" },
          ffmpeg: { status: "ok", message: "" },
          cuda: { status: "unavailable", message: "CPU only" },
          embedding_model: { status: "ok", message: "" },
        },
      });
    vi.mocked(api.putConfig).mockResolvedValueOnce({
      accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "" },
      transcription: {
        engine: "faster-whisper",
        model_size: "base",
        device: "cpu",
        language: "auto",
      },
      vision: { enabled: false, model: "", frame_sample_rate: 60 },
      backend: { port: 8765, worker_concurrency: 2 },
    });

    renderPage();

    const modelInput = await screen.findByLabelText(/model/i);
    fireEvent.change(modelInput, { target: { value: "gpt-4.1" } });
    fireEvent.blur(modelInput);

    expect(await screen.findByRole("tab", { name: /llm/i })).toHaveAttribute("title", "Unavailable");
    expect(api.getHealth).toHaveBeenCalledTimes(2);
  });

  test("does not save when the config value did not change", async () => {
    renderPage();

    const modelInput = await screen.findByLabelText(/model/i);
    fireEvent.blur(modelInput);

    expect(api.putConfig).not.toHaveBeenCalled();
    expect(api.getHealth).toHaveBeenCalledTimes(1);
  });

  test("does not refresh health for non-health-affecting config changes", async () => {
    vi.mocked(api.putConfig).mockResolvedValueOnce({
      accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "" },
      transcription: {
        engine: "faster-whisper",
        model_size: "base",
        device: "cpu",
        language: "auto",
      },
      vision: { enabled: false, model: "", frame_sample_rate: 60 },
      backend: { port: 8765, worker_concurrency: 3 },
    });

    renderPageAt("/settings?tab=other");

    const workerInput = await screen.findByLabelText(/worker concurrency/i);
    fireEvent.change(workerInput, { target: { value: "3" } });
    fireEvent.blur(workerInput);

    expect(api.putConfig).toHaveBeenCalledTimes(1);
    expect(api.getHealth).toHaveBeenCalledTimes(1);
  });
});
