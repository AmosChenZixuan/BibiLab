import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LlmTab } from "../components/settings/LlmTab";
import type { LocusConfig } from "../lib/types";

const baseConfig: LocusConfig = {
  accounts: { bilibili: { cookie: "", last_verified: "" } },
  ai: { provider: "openai", model: "gpt-4o", api_key: "", base_url: "" },
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
});

describe("llm tab", () => {
  test("renders provider dropdown", () => {
    render(<LlmTab config={baseConfig} onBlur={() => {}} />);

    expect(screen.getByLabelText(/provider/i)).toBeInTheDocument();
  });

  test("calls onBlur with updated config when model changes", () => {
    const onBlur = vi.fn();

    render(<LlmTab config={baseConfig} onBlur={onBlur} />);

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

    render(<LlmTab config={baseConfig} onBlur={onBlur} />);

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
    render(<LlmTab config={baseConfig} onBlur={() => {}} />);

    expect(screen.getByText(/official openai v1 base url/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/base url/i)).toHaveAttribute("placeholder", "https://api.openai.com/v1");

    fireEvent.change(screen.getByLabelText(/provider/i), {
      target: { value: "ollama" },
    });

    expect(screen.getByText(/ollama server root/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/base url/i)).toHaveAttribute("placeholder", "http://localhost:11434");
  });
});
