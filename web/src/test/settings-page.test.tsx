import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

function installFetchMock(handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("settings page", () => {
  test("loads config and health, saves a config patch, and queues a whisper model download", async () => {
    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/config") && method === "GET") {
        return Response.json({
          accounts: { bilibili: { cookie: "***", last_verified: "" } },
          ai: { provider: "openai", model: "gpt-4o", api_key: "***", base_url: null },
          transcription: {
            engine: "faster-whisper",
            model_size: "large-v3",
            device: "cuda",
            language: "auto",
          },
          vision: { enabled: false, frame_sample_rate: 30, model: null },
          backend: { port: 8765, worker_concurrency: 1 },
        });
      }

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
            llm: { status: "ok", message: "" },
            whisper_model: { status: "error", message: "missing" },
            ffmpeg: { status: "ok", message: "" },
            cuda: { status: "unavailable", message: "cpu" },
            bilibili_session: { status: "error", message: "cookie missing" },
            embedding_model: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/models/whisper") && method === "GET") {
        return Response.json([
          { name: "small", installed: false, path: null, selected: false },
          { name: "large-v3", installed: true, path: "/tmp/large-v3", selected: true },
        ]);
      }

      if (url.endsWith("/api/config") && method === "PUT") {
        const body = JSON.parse(String(init?.body));
        return Response.json({
          accounts: { bilibili: { cookie: "***", last_verified: "" } },
          ai: {
            provider: "openai",
            model: body.ai?.model ?? "gpt-4o",
            api_key: "***",
            base_url: null,
          },
          transcription: {
            engine: "faster-whisper",
            model_size: "large-v3",
            device: "cuda",
            language: "auto",
          },
          vision: { enabled: false, frame_sample_rate: 30, model: null },
          backend: { port: 8765, worker_concurrency: 1 },
        });
      }

      if (url.endsWith("/api/models/whisper/download") && method === "POST") {
        return Response.json(
          {
            job_id: "job-1",
            status: "queued",
            model_family: "whisper",
            model_size: "small",
          },
          { status: 202 },
        );
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/settings"] });

    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: /settings/i })).toBeInTheDocument();
    expect(screen.getByDisplayValue("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText(/overall status: ok/i)).toBeInTheDocument();
    expect(screen.getByText(/^large-v3$/)).toBeInTheDocument();

    const modelInput = screen.getByLabelText(/ai model/i);
    await userEvent.clear(modelInput);
    await userEvent.type(modelInput, "gpt-4.1");
    await userEvent.click(screen.getByRole("button", { name: /save settings/i }));
    expect(await screen.findByText(/settings saved/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /download small/i }));
    expect(await screen.findByText(/queued whisper model download/i)).toBeInTheDocument();
  });

  test("shows initial settings load errors inline", async () => {
    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/config") && method === "GET") {
        return Response.json({ detail: "Settings unavailable" }, { status: 503 });
      }

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/models/whisper") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/settings"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: /settings/i })).toBeInTheDocument();
    expect(screen.getByText("Settings unavailable")).toBeInTheDocument();
  });
});
