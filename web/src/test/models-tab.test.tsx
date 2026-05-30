import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ModelsTab } from "@/components/settings/ModelsTab";

vi.mock("../lib/api", () => {
  const mockApi = {
    listModels: vi.fn().mockResolvedValue([
      {
        id: "sensevoice-small",
        display_name: "SenseVoice Small",
        kind: "transcription",
        size_mb: 936,
        status: "present",
        required_by_config: true,
        path: "/models/asr/sensevoice-small",
      },
      {
        id: "ct-punc",
        display_name: "CT-Transformer Punctuation (zh-en)",
        kind: "punctuation",
        size_mb: 1050,
        status: "missing",
        required_by_config: true,
        path: null,
      },
    ]),
    listJobs: vi.fn().mockResolvedValue([]),
  };
  return {
    createApiClient: () => mockApi,
    api: mockApi,
    setCurrentLang: vi.fn(),
    notifyHealthChanged: vi.fn(),
    toErrorMessageWithT: (error: unknown) => (error instanceof Error ? error.message : "error"),
  };
});

afterEach(cleanup);

describe("ModelsTab", () => {
  test("renders punctuation-kind models (ct-punc) in the transcription group", async () => {
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <ModelsTab />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    // Regression: punctuation kind was filtered out of every render group,
    // so ct-punc never appeared in the settings model tab.
    expect(await screen.findByText("CT-Transformer Punctuation (zh-en)")).toBeInTheDocument();
  });
});
