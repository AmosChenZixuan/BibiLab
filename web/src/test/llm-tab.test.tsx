import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { LlmTab } from "@/components/settings/LlmTab";
import type { BibilabConfig } from "@/lib/types";

const baseConfig: BibilabConfig = {
  accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
  ai: { protocol: "openai", model: "gpt-4o", api_key: "", base_url: "" },
  transcription: {
    model: "large-v3",
    device: "cpu",
    language: "auto",
  },
  backend: { port: 8765, max_concurrent_jobs: 2, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
  rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
};

afterEach(() => {
  cleanup();
});

describe("llm tab", () => {
  test("renders provider dropdown", () => {
    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={() => {}} />
      </LanguageProvider>,
    );

    expect(screen.getByLabelText(/provider/i)).toBeInTheDocument();
  });

  test("calls onBlur with updated config when model changes", () => {
    const onBlur = vi.fn();

    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={onBlur} />
      </LanguageProvider>,
    );

    fireEvent.change(screen.getByLabelText(/model/i), {
      target: { value: "gpt-4-turbo" },
    });
    fireEvent.blur(screen.getByLabelText(/model/i));

    expect(onBlur).toHaveBeenCalledWith(
      expect.objectContaining({
        ai: expect.objectContaining({ model: "gpt-4-turbo" }),
      }),
    );
  });

  test("treats base url as required text, not nullable", () => {
    const onBlur = vi.fn();

    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={onBlur} />
      </LanguageProvider>,
    );

    fireEvent.change(screen.getByLabelText(/base url/i), {
      target: { value: "https://api.openai.com/v1" },
    });
    fireEvent.blur(screen.getByLabelText(/base url/i));

    expect(onBlur).toHaveBeenCalledWith(
      expect.objectContaining({
        ai: expect.objectContaining({ base_url: "https://api.openai.com/v1" }),
      }),
    );
  });

  test("updates base url hint and placeholder by provider", () => {
    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={() => {}} />
      </LanguageProvider>,
    );

    expect(screen.getByText(/Required\. OpenAI, DeepSeek, GLM, Ollama, and other OpenAI-compatible providers/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/base url/i)).toHaveAttribute("placeholder", "https://api.openai.com/v1");

    fireEvent.change(screen.getByLabelText(/provider/i), {
      target: { value: "anthropic" },
    });

    expect(screen.getByText(/Anthropic API base URL/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/base url/i)).toHaveAttribute("placeholder", "https://api.anthropic.com/v1");
  });

  test("api_key input type toggles between password and text", () => {
    const config = { ...baseConfig, ai: { ...baseConfig.ai, api_key: "sk-test123" } };
    render(
      <LanguageProvider>
        <LlmTab config={config} onBlur={() => {}} />
      </LanguageProvider>,
    );

    const apiKeyInput = screen.getByLabelText("API Key", { selector: "input" });
    expect(apiKeyInput).toHaveAttribute("type", "password");

    const toggleBtn = screen.getByRole("button", { name: /reveal/i });
    fireEvent.click(toggleBtn);
    expect(apiKeyInput).toHaveAttribute("type", "text");

    const hideBtn = screen.getByRole("button", { name: /hide/i });
    fireEvent.click(hideBtn);
    expect(apiKeyInput).toHaveAttribute("type", "password");
  });
});
