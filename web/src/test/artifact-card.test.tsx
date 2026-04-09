import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ArtifactCard } from "@/components/lists/lab/ArtifactCard";
import type { Artifact } from "@/lib/types";

function renderCard(artifact: Artifact, onDismiss?: (id: string) => void) {
  return render(
    <LanguageProvider>
      <ArtifactCard artifact={artifact} onDismiss={onDismiss} />
    </LanguageProvider>,
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ArtifactCard", () => {
  describe("generating state", () => {
    test("shows pulsing skeleton with type label as name", () => {
      const artifact: Artifact = {
        id: "gen-brief-1",
        name: "",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "generating",
        created_at: "2026-04-08T12:00:00Z",
      };

      renderCard(artifact);

      // Should show type label as name (uppercase)
      expect(screen.getByText("BRIEF")).toBeInTheDocument();
      // Should have animate-pulse class (skeleton)
      expect(screen.getByTestId("artifact-skeleton")).toBeInTheDocument();
      // Should NOT have context menu
      expect(screen.queryByRole("button", { name: /options/i })).not.toBeInTheDocument();
    });

    test("generating study_guide shows STUDY_GUIDE label", () => {
      const artifact: Artifact = {
        id: "gen-study-1",
        name: "",
        type: "study_guide",
        prompt: "Generate a study guide",
        source_ids: ["source-1", "source-2"],
        status: "generating",
        created_at: "2026-04-08T12:00:00Z",
      };

      renderCard(artifact);
      expect(screen.getByText("STUDY_GUIDE")).toBeInTheDocument();
    });
  });

  describe("done state", () => {
    test("shows name, metadata line, and context menu", () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1", "source-2"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };

      renderCard(artifact);

      // Name should be bold
      expect(screen.getByText("My Artifact")).toBeInTheDocument();
      // Metadata line shows type · source count · date
      expect(screen.getByText(/^brief · 2 sources · /)).toBeInTheDocument();
      // Context menu should be present
      expect(screen.getByRole("button", { name: /artifact options/i })).toBeInTheDocument();
    });
  });

  describe("error state", () => {
    test("shows pink error card with alert icon, error message, and dismiss button", async () => {
      const artifact: Artifact = {
        id: "error-blog-1",
        name: "Failed Artifact",
        type: "blog_post",
        prompt: "Generate a blog post",
        source_ids: ["source-1"],
        status: "error",
        created_at: "2026-04-08T12:00:00Z",
        error: "Generation failed: model timeout",
      };

      const onDismiss = vi.fn();
      renderCard(artifact, onDismiss);

      // AlertCircle icon should be present
      expect(screen.getByTestId("alert-icon")).toBeInTheDocument();
      // Error message should be shown
      expect(screen.getByText(/model timeout/i)).toBeInTheDocument();
      // Dismiss button should be present
      const dismissBtn = screen.getByRole("button", { name: /dismiss/i });
      expect(dismissBtn).toBeInTheDocument();
      // Clicking dismiss calls onDismiss with artifact id
      await userEvent.click(dismissBtn);
      await waitFor(() => {
        expect(onDismiss).toHaveBeenCalledWith("error-blog-1");
      });
    });
  });
});
