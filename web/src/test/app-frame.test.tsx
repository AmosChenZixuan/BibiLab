import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { AppFrame } from "@/components/layout/AppFrame";
import type { BibilabConfig, HealthResponse } from "@/lib/types";

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
    setCurrentLang: vi.fn(),
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
      accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
      ai: { protocol: "", model: "", api_key: "", base_url: "" },
      transcription: { model: "large-v3", device: "cuda", language: "" },
      vision: { enabled: false, frame_sample_rate: 0, model: null },
      backend: { port: 8765, max_concurrent_jobs: 1 },
    }),
  );

  return render(
    <LanguageProvider>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<AppFrame />}>
            <Route index element={<div>Home</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </LanguageProvider>,
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
        accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
        ai: { protocol: "", model: "", api_key: "", base_url: "" },
        transcription: { model: "large-v3", device: "cpu", language: "" },
        vision: { enabled: false, frame_sample_rate: 0, model: null },
        backend: { port: 8765, max_concurrent_jobs: 1 },
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
            last_verified: "2025-01-01T00:00:00Z",
            username: "test_user",
            avatar_url: "https://i0.hdslb.com/bfs/face/abc.jpg",
          },
        },
        ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "" },
        transcription: { model: "large-v3", device: "cpu", language: "auto" },
        vision: { enabled: false, frame_sample_rate: 30, model: null },
        backend: { port: 8765, max_concurrent_jobs: 1 },
      },
    );

    await userEvent.click(await screen.findByRole("button", { name: /identity/i }));

    const menu = screen.getByRole("menu", { name: "Identity" });
    expect(menu).toBeInTheDocument();
    expect(screen.getByText("test_user")).toBeInTheDocument();
    expect(screen.getByLabelText("Sign out")).toBeInTheDocument();
  });
});
