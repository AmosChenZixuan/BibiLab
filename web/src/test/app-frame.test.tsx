import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { AppFrame } from "@/components/layout/AppFrame";
import type { BibilabConfig, HealthResponse } from "@/lib/types";
import { renderWithProviders } from "@/test/utils";

vi.mock("../lib/api", () => {
  const mockApi = {
    getHealth: vi.fn(),
    getConfig: vi.fn(),
    listJobs: vi.fn().mockResolvedValue([]),
    auth: {
      generateBilibiliQr: vi.fn(),
      pollBilibiliQr: vi.fn(),
      deleteBilibiliAuth: vi.fn(),
    },
  };
  return {
    HEALTH_REFRESH_EVENT: "bibilab:health:refresh",
    BILIBILI_AUTH_REFRESH_EVENT: "bibilab:auth:bilibili:refresh",
    JOBS_REFRESH_EVENT: "bibilab:jobs:refresh",
    notifyBilibiliAuthChanged: vi.fn(),
    createApiClient: () => mockApi,
    api: mockApi,
  };
});

import { api } from "@/lib/api";
import { HEALTH_REFRESH_EVENT } from "@/lib/api";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderFrame(healthPayload: HealthResponse, configPayload?: BibilabConfig) {
  vi.mocked(api.getHealth).mockResolvedValue(healthPayload);
  vi.mocked(api.getConfig).mockResolvedValue(
    configPayload ?? ({
      accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
      ai: { protocol: "", model: "", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
      transcription: { model: "large-v3", device: "cuda", language: "" },
      backend: { port: 8765, max_concurrent_jobs: 1, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
      rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
    }),
  );

  return renderWithProviders(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route element={<AppFrame />}>
          <Route index element={<div>Home</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
    { providers: [LanguageProvider] },
  );
}

describe("app frame", () => {
  test("shows operational badge when all deps ok", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByTitle("Healthy")).toBeInTheDocument();
  });

  test("shows healthy badge when cuda unavailable (no GPU hardware)", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "unavailable", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByTitle("Healthy")).toBeInTheDocument();
  });

  test("shows degraded badge when cuda available but device is cpu", async () => {
    renderFrame(
      {
        overall: "ok",
        dependencies: {
          cuda: { status: "ok", message: "" },
          embedding_model: { status: "ok", message: "" },
        },
      },
      {
        accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
        ai: { protocol: "", model: "", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
        transcription: { model: "large-v3", device: "cpu", language: "" },
        backend: { port: 8765, max_concurrent_jobs: 1, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
        rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
      },
    );

    expect(await screen.findByTitle("Throttled")).toBeInTheDocument();
  });

  test("shows unavailable badge when overall is error", async () => {
    renderFrame({
      overall: "error",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByTitle("Unavailable")).toBeInTheDocument();
  });

  test("does not render a navbar language toggle", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByLabelText(/settings/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/^language$/i)).not.toBeInTheDocument();
  });

  test("does not reserve navbar space for a jobs control", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByLabelText(/settings/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /jobs/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /active job/i })).not.toBeInTheDocument();
  });

  test("updates navbar health when settings broadcasts refreshed health", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByTitle("Healthy")).toBeInTheDocument();

    window.dispatchEvent(
      new CustomEvent(HEALTH_REFRESH_EVENT, {
        detail: {
          overall: "error",
          dependencies: {
            cuda: { status: "ok", message: "" },
            embedding_model: { status: "ok", message: "" },
          },
        } satisfies HealthResponse,
      }),
    );

    await waitFor(() => {
      expect(screen.getByTitle("Unavailable")).toBeInTheDocument();
    });
  });

  test("renders identity panel outside the navbar stacking context", async () => {
    const { container } = renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    await userEvent.click(await screen.findByRole("button", { name: "Identity" }));

    const nav = container.querySelector("nav");
    const menu = screen.getByRole("menu", { name: "Identity" });

    expect(nav).not.toContainElement(menu);
  });

  test("opens identity panel with signed-in state when config has username and avatar_url", async () => {
    renderFrame(
      {
        overall: "ok",
        dependencies: {
          cuda: { status: "ok", message: "" },
          embedding_model: { status: "ok", message: "" },
        },
      },
      {
        accounts: {
          bilibili: {
            cookie: "SESSDATA=abc",
            username: "test_user",
            avatar_url: "https://i0.hdslb.com/bfs/face/abc.jpg",
          },
        },
        ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "", context_window: 128000, max_output_tokens: 16384 },
        transcription: { model: "large-v3", device: "cpu", language: "auto" },
        backend: { port: 8765, max_concurrent_jobs: 1, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
        rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
      },
    );

    await userEvent.click(await screen.findByRole("button", { name: /identity/i }));

    const menu = screen.getByRole("menu", { name: "Identity" });
    expect(menu).toBeInTheDocument();
    expect(screen.getByText("test_user")).toBeInTheDocument();
    expect(screen.getByLabelText("Sign out")).toBeInTheDocument();
  });

  test("renders the BrandMark inside the navbar home link", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    const homeLink = await screen.findByRole("link", { name: "Home" });
    const svg = homeLink.querySelector("svg");
    expect(svg).toBeInTheDocument();
    // The mark is decorative — the link's accessible name comes from the
    // parent's aria-label="Home", not the SVG.
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(homeLink.className).toContain("text-muted");
  });
});
