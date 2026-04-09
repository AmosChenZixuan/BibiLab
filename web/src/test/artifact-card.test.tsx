import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ArtifactCard } from "@/components/lists/lab/ArtifactCard";
import type { Artifact } from "@/lib/types";

function renderCard(
  artifact: Artifact,
  onDismiss?: (id: string) => void,
  onDownload?: (id: string) => void,
  onRename?: (id: string, name: string) => void,
  onViewPrompt?: (id: string) => void,
  onDelete?: (id: string) => void,
) {
  return render(
    <LanguageProvider>
      <ArtifactCard
        artifact={artifact}
        onDismiss={onDismiss}
        onDownload={onDownload}
        onRename={onRename}
        onViewPrompt={onViewPrompt}
        onDelete={onDelete}
      />
    </LanguageProvider>,
  );
}

afterEach(() => {
  cleanup();
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

    test("done state shows context menu with rename, download, view prompt, and delete items", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief about AI",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onDownload = vi.fn();
      const onRename = vi.fn();
      const onViewPrompt = vi.fn();
      const onDelete = vi.fn();

      renderCard(artifact, undefined, onDownload, onRename, onViewPrompt, onDelete);

      // Open the context menu
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);

      // Context menu items should be visible
      expect(screen.getByText(/rename/i)).toBeInTheDocument();
      expect(screen.getByText(/download/i)).toBeInTheDocument();
      expect(screen.getByText(/view prompt/i)).toBeInTheDocument();
      expect(screen.getByText(/delete/i)).toBeInTheDocument();
    });

    test("clicking download calls onDownload with artifact id", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onDownload = vi.fn();

      renderCard(artifact, undefined, onDownload);

      // Open context menu and click download
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^download$/i));

      expect(onDownload).toHaveBeenCalledWith("done-brief-1");
    });

    test("clicking view prompt calls onViewPrompt with artifact id", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief about AI",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onViewPrompt = vi.fn();

      renderCard(artifact, undefined, undefined, undefined, onViewPrompt);

      // Open context menu and click view prompt
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^view prompt$/i));

      expect(onViewPrompt).toHaveBeenCalledWith("done-brief-1");
    });

    test("clicking delete calls onDelete with artifact id", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onDelete = vi.fn();

      renderCard(artifact, undefined, undefined, undefined, undefined, onDelete);

      // Open context menu and click delete
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^delete$/i));

      expect(onDelete).toHaveBeenCalledWith("done-brief-1");
    });

    test("clicking rename shows input field with current name", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onRename = vi.fn();

      renderCard(artifact, undefined, undefined, onRename);

      // Open context menu and click rename
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^rename$/i));

      // Name should now be an input field
      const input = screen.getByRole("textbox");
      expect(input).toBeInTheDocument();
      expect(input).toHaveValue("My Artifact");
    });

    test("rename input submits on enter and calls onRename", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onRename = vi.fn();

      renderCard(artifact, undefined, undefined, onRename);

      // Open context menu and click rename
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^rename$/i));

      // Change name and press enter
      const input = screen.getByRole("textbox");
      await userEvent.clear(input);
      await userEvent.type(input, "New Artifact Name{Enter}");

      expect(onRename).toHaveBeenCalledWith("done-brief-1", "New Artifact Name");
    });

    test("rename input submits on blur and calls onRename", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onRename = vi.fn();

      renderCard(artifact, undefined, undefined, onRename);

      // Open context menu and click rename
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^rename$/i));

      // Change name and blur (tab away)
      const input = screen.getByRole("textbox");
      await userEvent.clear(input);
      await userEvent.type(input, "Blurred Name");
      await userEvent.tab();

      expect(onRename).toHaveBeenCalledWith("done-brief-1", "Blurred Name");
    });

    test("rename input cancels on escape and reverts to original name", async () => {
      const artifact: Artifact = {
        id: "done-brief-1",
        name: "My Artifact",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      };
      const onRename = vi.fn();

      renderCard(artifact, undefined, undefined, onRename);

      // Open context menu and click rename
      const menuBtn = screen.getByRole("button", { name: /artifact options/i });
      await userEvent.click(menuBtn);
      await userEvent.click(screen.getByText(/^rename$/i));

      // Type new name but press escape
      const input = screen.getByRole("textbox");
      await userEvent.clear(input);
      await userEvent.type(input, "Cancelled Name{Escape}");

      // onRename should NOT be called
      expect(onRename).not.toHaveBeenCalled();
      // Original name should be displayed
      expect(screen.getByText("My Artifact")).toBeInTheDocument();
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
