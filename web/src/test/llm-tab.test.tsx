import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { LlmTab } from "@/components/settings/LlmTab";
import type { BibilabConfig } from "@/lib/types";

const baseConfig: BibilabConfig = {
  accounts: { bilibili: { cookie: "", username: "", avatar_url: "" } },
  ai: {
    protocol: "openai",
    model: "gpt-4o",
    api_key: "",
    base_url: "",
    context_window: 128000,
    max_output_tokens: 16384,
  },
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

    // Use exact label "Model" (not the regex /model/i) — the new
    // "Model context window" SlotSlider label also contains the substring
    // "model" and would match a loose query.
    const modelInput = screen.getByLabelText("Model");
    fireEvent.change(modelInput, { target: { value: "gpt-4-turbo" } });
    fireEvent.blur(modelInput);

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

    const baseUrlInput = screen.getByLabelText("Base URL");
    fireEvent.change(baseUrlInput, { target: { value: "https://api.openai.com/v1" } });
    fireEvent.blur(baseUrlInput);

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
    expect(screen.getByLabelText("Base URL")).toHaveAttribute("placeholder", "https://api.openai.com/v1");

    fireEvent.change(screen.getByLabelText("Provider"), {
      target: { value: "anthropic" },
    });

    expect(screen.getByText(/Anthropic API base URL/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Base URL")).toHaveAttribute("placeholder", "https://api.anthropic.com/v1");
  });

  test("calls onBlur with the selected context window preset as a number", () => {
    const onBlur = vi.fn();

    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={onBlur} />
      </LanguageProvider>,
    );

    // SlotSlider renders a radiogroup; scope to the context_window group first
    // because both sliders share a 64K slot.
    const ctxGroup = screen.getByRole("radiogroup", { name: /context window/i });
    const slot = ctxGroup.querySelector('[role="radio"][aria-label="200K"]') as HTMLElement;
    fireEvent.click(slot);

    expect(onBlur).toHaveBeenCalledWith(
      expect.objectContaining({
        ai: expect.objectContaining({ context_window: 200000 }),
      }),
    );
  });

  test("calls onBlur with the selected max output tokens preset as a number", () => {
    const onBlur = vi.fn();

    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={onBlur} />
      </LanguageProvider>,
    );

    const maxGroup = screen.getByRole("radiogroup", { name: /maximum output/i });
    const slot = maxGroup.querySelector('[role="radio"][aria-label="64K"]') as HTMLElement;
    fireEvent.click(slot);

    expect(onBlur).toHaveBeenCalledWith(
      expect.objectContaining({
        ai: expect.objectContaining({ max_output_tokens: 65536 }),
      }),
    );
  });

  test("does not commit a selection that makes the budget invalid", () => {
    const onBlur = vi.fn();
    // 64K window — selecting a 100K output budget would exceed it, which the
    // backend validator rejects. The slider must surface the inline error and
    // hold off the commit rather than fire a doomed PUT.
    const config = {
      ...baseConfig,
      ai: { ...baseConfig.ai, context_window: 64000, max_output_tokens: 16384 },
    };
    render(
      <LanguageProvider>
        <LlmTab config={config} onBlur={onBlur} />
      </LanguageProvider>,
    );

    const maxGroup = screen.getByRole("radiogroup", { name: /maximum output/i });
    const slot = maxGroup.querySelector('[role="radio"][aria-label="100K"]') as HTMLElement;
    fireEvent.click(slot);

    expect(onBlur).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/smaller than the context window/i);
  });

  test("shows inline error when max output tokens >= context window", () => {
    const config = {
      ...baseConfig,
      ai: { ...baseConfig.ai, context_window: 64000, max_output_tokens: 64000 },
    };
    render(
      <LanguageProvider>
        <LlmTab config={config} onBlur={() => {}} />
      </LanguageProvider>,
    );

    // The cross-field validator surfaces the issue as a paragraph with role=alert.
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/smaller than the context window/i);
  });

  test("no inline error when max output tokens < context window", () => {
    render(
      <LanguageProvider>
        <LlmTab config={baseConfig} onBlur={() => {}} />
      </LanguageProvider>,
    );

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
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
