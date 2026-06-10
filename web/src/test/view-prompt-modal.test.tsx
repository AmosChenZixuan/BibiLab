import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ViewPromptModal } from "@/components/lists/lab/ViewPromptModal";
import { renderWithProviders } from "@/test/utils";

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  return {
    api: createMockApi(),
  };
});

function renderModal(props?: Partial<React.ComponentProps<typeof ViewPromptModal>>) {
  return renderWithProviders(
    <ViewPromptModal
      open={true}
      onClose={vi.fn()}
      prompt="Generate a study guide for these videos."
      {...props}
    />,
    { providers: [LanguageProvider] },
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ViewPromptModal", () => {
  test("renders prompt text in a pre block", () => {
    renderModal();
    expect(screen.getByText("Generate a study guide for these videos.")).toBeInTheDocument();
  });

  test("close button calls onClose", async () => {
    const onClose = vi.fn();
    renderModal({ onClose });

    const buttons = screen.getAllByRole("button");
    const closeButton = buttons.find((btn) => btn.textContent?.match(/close/i));
    expect(closeButton).toBeTruthy();
    await userEvent.click(closeButton!);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("copy button copies prompt to clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    renderModal();

    const buttons = screen.getAllByRole("button");
    const copyButton = buttons.find((btn) => btn.textContent?.match(/copy/i));
    expect(copyButton).toBeTruthy();
    await userEvent.click(copyButton!);
    expect(writeText).toHaveBeenCalledWith("Generate a study guide for these videos.");
  });

  test("does not render when closed", () => {
    renderModal({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
