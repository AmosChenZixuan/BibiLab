import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "../app/LanguageContext";
import { AppFrame } from "../components/layout/AppFrame";
import type { HealthResponse } from "../lib/types";

vi.mock("../lib/api", () => ({
  JOBS_REFRESH_EVENT: "locus:jobs:refresh",
  api: {
    getHealth: vi.fn(),
    listJobs: vi.fn().mockResolvedValue([]),
  },
}));

import { api } from "../lib/api";

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
});
