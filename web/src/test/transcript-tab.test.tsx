import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { TranscriptTab } from "../components/settings/TranscriptTab";
import type { HealthDependency, LocusConfig } from "../lib/types";

vi.mock("../lib/api", () => ({
  api: {
    listWhisperModels: vi.fn().mockResolvedValue([
      { name: "base", installed: true, path: "/models/base", selected: true },
      { name: "large-v3", installed: false, path: null, selected: false },
    ]),
    downloadWhisperModel: vi.fn(),
  },
}));

const baseConfig: LocusConfig = {
  accounts: { bilibili: { cookie: "", last_verified: "" } },
  ai: { provider: "openai", model: "", api_key: "", base_url: "" },
  transcription: {
    engine: "faster-whisper",
    model_size: "base",
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

describe("transcript tab", () => {
  test("renders transcription device dropdown", () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(screen.getByLabelText(/device/i)).toBeInTheDocument();
  });

  test("shows installed path for downloaded model", async () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "base" })).toBeInTheDocument();
    expect(await screen.findByText(/\/models\/base/i)).toBeInTheDocument();
  });

  test("shows download button for missing model", async () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(await screen.findByRole("button", { name: /download large-v3/i })).toBeInTheDocument();
  });

  test("shows impact messaging for cuda and missing whisper models", async () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(screen.getByText(/cuda not available; cpu will be used/i)).toBeInTheDocument();
    expect(await screen.findByText(/transcription cannot start until a model is downloaded/i)).toBeInTheDocument();
  });

  test("model size dropdown only lists downloaded models", async () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(await screen.findByRole("option", { name: "base" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "large-v3" })).not.toBeInTheDocument();
  });

  test("download table does not render status chips", async () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.queryByText(/selected/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^installed$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^missing$/i)).not.toBeInTheDocument();
  });

  test("disables cuda when health reports it unavailable", () => {
    render(<TranscriptTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} />);

    expect(screen.getByRole("option", { name: "CUDA" })).toBeDisabled();
  });
});
