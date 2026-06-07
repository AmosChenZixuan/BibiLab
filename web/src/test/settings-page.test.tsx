import { cleanup, fireEvent, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { SettingsPage } from "@/pages/SettingsPage";
import { renderWithProviders } from "@/test/utils";

const MOCK_ERROR_MESSAGE = "Request failed";

vi.mock("../lib/api", () => {
  const mockApi = {
    getConfig: vi.fn().mockResolvedValue({
      accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
      transcription: {
        model: "large-v3",
        device: "cpu",
        language: "auto",
      },
      backend: { port: 8765, max_concurrent_jobs: 2, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
      rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
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
    listModels: vi.fn().mockResolvedValue([]),
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
  return renderWithProviders(
    <MemoryRouter>
      <SettingsPage />
    </MemoryRouter>,
    { providers: [JobActivityProvider, LanguageProvider] },
  );
}

function renderPageAt(entry: string) {
  function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
  }

  return renderWithProviders(
    <MemoryRouter initialEntries={[entry]}>
      <SettingsPage />
      <LocationProbe />
    </MemoryRouter>,
    { providers: [JobActivityProvider, LanguageProvider] },
  );
}

describe("settings page", () => {
  test("renders four tabs", async () => {
    renderPage();

    expect(await screen.findByRole("tab", { name: /ai service/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /transcript/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /local models/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /system/i })).toBeInTheDocument();
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

  test("clicking System tab shows backend api section", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("tab", { name: /system/i }));
    expect(await screen.findByText(/backend api/i)).toBeInTheDocument();
  });

  test("clicking a tab updates the url query", async () => {
    renderPageAt("/settings");

    fireEvent.click(await screen.findByRole("tab", { name: /system/i }));

    expect(screen.getByTestId("location")).toHaveTextContent("/settings?tab=system");
  });

  test("reads active tab from the url query", async () => {
    renderPageAt("/settings?tab=transcript");

    expect(await screen.findByLabelText(/device/i)).toBeInTheDocument();
    expect(screen.getByTestId("location")).toHaveTextContent("/settings?tab=transcript");
  });

  test("tab health indicators use the same tiers as the navbar health model", async () => {
    renderPage();

    expect(await screen.findByRole("tab", { name: /ai service/i })).toHaveAttribute("title", "Healthy");
    expect(screen.getByRole("tab", { name: /transcript/i })).toHaveAttribute("title", "Healthy");
    expect(screen.getByRole("tab", { name: /local models/i })).toHaveAttribute("title", "Healthy");
    expect(screen.getByRole("tab", { name: /system/i })).toHaveAttribute("title", "Healthy");
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
      accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
      transcription: {
        model: "large-v3",
        device: "cpu",
        language: "auto",
      },
      backend: { port: 8765, max_concurrent_jobs: 2, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
      rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
    });

    renderPage();

    const modelInput = await screen.findByLabelText("Model");
    fireEvent.change(modelInput, { target: { value: "gpt-4.1" } });
    fireEvent.blur(modelInput);

    expect(await screen.findByRole("tab", { name: /ai service/i })).toHaveAttribute("title", "Unavailable");
    expect(api.getHealth).toHaveBeenCalledTimes(2);
  });

  test("does not save when the config value did not change", async () => {
    renderPage();

    const modelInput = await screen.findByLabelText("Model");
    fireEvent.blur(modelInput);

    expect(api.putConfig).not.toHaveBeenCalled();
    expect(api.getHealth).toHaveBeenCalledTimes(1);
  });

  test("does not refresh health for non-health-affecting config changes", async () => {
    vi.mocked(api.putConfig).mockResolvedValueOnce({
      accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
      ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
      transcription: {
        model: "large-v3",
        device: "cpu",
        language: "auto",
      },
      backend: { port: 8765, max_concurrent_jobs: 3, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
      rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
    });

    renderPageAt("/settings?tab=system");

    const workerInput = await screen.findByLabelText(/max concurrent jobs/i);
    fireEvent.change(workerInput, { target: { value: "3" } });
    fireEvent.blur(workerInput);

    expect(api.putConfig).toHaveBeenCalledTimes(1);
    expect(api.getHealth).toHaveBeenCalledTimes(1);
  });
});
