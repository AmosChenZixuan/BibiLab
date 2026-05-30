import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { OtherTab } from "@/components/settings/OtherTab";
import type { HealthDependency, BibilabConfig } from "@/lib/types";

const baseConfig: BibilabConfig = {
  accounts: { bilibili: { cookie: "", last_verified: "", username: "", avatar_url: "" } },
  ai: { protocol: "openai", model: "", api_key: "", base_url: "" },
  transcription: {
    model: "large-v3",
    device: "cpu",
    language: "auto",
  },
  vision: { enabled: false, model: "", frame_sample_rate: 60 },
  backend: { port: 8765, max_concurrent_jobs: 2 },
};

const healthDeps: Record<string, HealthDependency> = {
  backend: { status: "ok", message: "" },
  ffmpeg: { status: "ok", message: "/usr/bin/ffmpeg" },
  embedding_model: {
    status: "ok",
    message: "/home/test/.bibilab/chroma/onnx/model.onnx",
  },
};

afterEach(() => {
  cleanup();
});

describe("other tab", () => {
  const renderTab = (props = {}) =>
    render(
      <LanguageProvider>
        <OtherTab config={baseConfig} dependencies={healthDeps} onBlur={() => {}} {...props} />
      </LanguageProvider>,
    );

  test("shows backend connected status and max concurrent jobs together", () => {
    renderTab();

    expect(screen.getByText(/backend api/i)).toBeInTheDocument();
    expect(screen.getByText(/localhost:8765/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max concurrent jobs/i)).toBeInTheDocument();
  });

  test("shows ffmpeg label and install path when installed", () => {
    renderTab();

    expect(screen.getByText(/^ffmpeg$/i)).toBeInTheDocument();
    expect(screen.getByText(/\/usr\/bin\/ffmpeg/i)).toBeInTheDocument();
  });

  test("does not surface embedding or reranker rows (moved to Models tab)", () => {
    renderTab();

    expect(screen.queryByText(/^Embedding Model$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Reranker Model$/i)).not.toBeInTheDocument();
  });

  test("shows impact messaging for backend and ffmpeg", () => {
    renderTab();

    expect(screen.getByText(/backend is offline, the web app cannot load or save configuration/i)).toBeInTheDocument();
    expect(screen.getByText(/without ffmpeg, media audio cannot be extracted/i)).toBeInTheDocument();
  });

  test("renders interface language selector", () => {
    renderTab();

    expect(screen.getByLabelText(/interface language/i)).toBeInTheDocument();
  });

  test("does not render a session cookie field", () => {
    renderTab();

    expect(screen.queryByLabelText(/session cookie/i)).not.toBeInTheDocument();
  });

  test("does not render vision settings", () => {
    renderTab();

    expect(screen.queryByText(/^vision$/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/vision model/i)).not.toBeInTheDocument();
  });
});
