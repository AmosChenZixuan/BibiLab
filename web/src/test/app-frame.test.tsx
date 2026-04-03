import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "../app/LanguageContext";
import { AppFrame } from "../components/layout/AppFrame";
import type { HealthResponse } from "../lib/types";

vi.mock("../lib/api", () => ({
  HEALTH_REFRESH_EVENT: "locus:health:refresh",
  JOBS_REFRESH_EVENT: "locus:jobs:refresh",
  api: {
    getHealth: vi.fn(),
    listJobs: vi.fn().mockResolvedValue([]),
  },
}));

import { api } from "../lib/api";
import { HEALTH_REFRESH_EVENT } from "../lib/api";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderFrame(healthPayload: HealthResponse) {
  vi.mocked(api.getHealth).mockResolvedValue(healthPayload);

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

    expect(await screen.findByTitle("Operational")).toBeInTheDocument();
  });

  test("shows degraded badge when cuda unavailable", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "unavailable", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByTitle("Degraded")).toBeInTheDocument();
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

  test("shows EN/ZH language toggle button", async () => {
    renderFrame({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    });

    expect(await screen.findByLabelText(/language/i)).toBeInTheDocument();
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

    expect(await screen.findByTitle("Operational")).toBeInTheDocument();

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
});
