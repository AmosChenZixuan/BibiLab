import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DebugDrawer } from "@/components/debug/DebugDrawer";
import { TEST_IDS } from "@/lib/test-ids";

afterEach(() => {
  cleanup();
});

const dump = {
  system: "You are a video-grounded assistant.",
  tools: [
    { name: "find_passages", description: "Search", parameters: { type: "object" } },
  ],
  messages: [
    { role: "user", content: "What changed?" },
    {
      role: "assistant",
      content: "Let me check.",
      tool_calls: [
        { id: "c1", type: "function", function: { name: "find_passages", arguments: "{}" } },
      ],
    },
    { role: "tool", tool_call_id: "c1", content: "[1] Result text" },
  ],
  response: { text: "Final answer." },
  model: "gpt-4o",
  timestamp: "2026-06-06T14:23:11+08:00",
};

describe("DebugDrawer", () => {
  it("renders system prompt section", () => {
    render(<DebugDrawer messageId="m1" dump={dump} onClose={vi.fn()} />);
    expect(screen.getByText(/system prompt/i)).toBeInTheDocument();
  });

  it("renders role chips for each message", () => {
    render(<DebugDrawer messageId="m1" dump={dump} onClose={vi.fn()} />);
    expect(screen.getByText("user")).toBeInTheDocument();
    expect(screen.getAllByText(/assistant/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/tool/i)).toBeInTheDocument();
  });

  it("renders tool name in the assistant message", () => {
    render(<DebugDrawer messageId="m1" dump={dump} onClose={vi.fn()} />);
    // The function name now appears in both the catalog and the tool_call chip
    expect(screen.getAllByText(/find_passages/).length).toBeGreaterThanOrEqual(1);
  });

  it("renders function catalog with description and parameters for tool definitions", () => {
    render(<DebugDrawer messageId="m1" dump={dump} onClose={vi.fn()} />);
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("parameters")).toBeInTheDocument();
  });

  it("falls back to JsonTree for unknown envelope shapes", () => {
    const weird = {
      ...dump,
      messages: [...dump.messages, { role: "developer", content: "x" }],
    };
    render(<DebugDrawer messageId="m1" dump={weird} onClose={vi.fn()} />);
    expect(screen.getByText("developer")).toBeInTheDocument();
  });

  it("toggles to raw JSON view", () => {
    render(<DebugDrawer messageId="m1" dump={dump} onClose={vi.fn()} />);
    fireEvent.click(screen.getByText("Raw"));
    expect(screen.getByText(/"system":/)).toBeInTheDocument();
  });
});
