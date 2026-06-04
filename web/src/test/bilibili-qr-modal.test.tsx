import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { BilibiliQrModal } from "@/components/auth/BilibiliQrModal";
import { renderWithProviders } from "@/test/utils";

const { mockGenerateBilibiliQr, mockPollBilibiliQr, mockDeleteBilibiliAuth } = vi.hoisted(() => ({
  mockGenerateBilibiliQr: vi.fn(),
  mockPollBilibiliQr: vi.fn(),
  mockDeleteBilibiliAuth: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  const { createMockApi } = await import("@/test/utils");
  return {
    ...actual,
    api: createMockApi({
      auth: {
        generateBilibiliQr: mockGenerateBilibiliQr,
        pollBilibiliQr: mockPollBilibiliQr,
        deleteBilibiliAuth: mockDeleteBilibiliAuth,
      },
    }),
  };
});

function renderModal(props?: Partial<React.ComponentProps<typeof BilibiliQrModal>>) {
  return renderWithProviders(
    <BilibiliQrModal
      open={true}
      onClose={vi.fn()}
      onSuccess={vi.fn()}
      {...props}
    />,
    { providers: [LanguageProvider] },
  );
}

describe("BilibiliQrModal", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockGenerateBilibiliQr.mockResolvedValue({ url: "https://example.com/qr", key: "test-key" });
    mockPollBilibiliQr.mockResolvedValue({ status: "waiting" });
    mockDeleteBilibiliAuth.mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
  });

  test("renders modal with title when open", async () => {
    renderModal({ open: true });
    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
    expect(screen.getByText("Sign in to Bilibili")).toBeInTheDocument();
  });

  test("does not render when open is false", () => {
    renderModal({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("shows loading state initially", async () => {
    mockGenerateBilibiliQr.mockReturnValue(new Promise(() => {}));
    renderModal({ open: true });
    await waitFor(() => {
      expect(screen.getByText("Creating...")).toBeInTheDocument();
    });
  });

  test("renders QR code after generation", async () => {
    renderModal({ open: true });
    await waitFor(() => {
      expect(screen.queryByText("Creating...")).not.toBeInTheDocument();
    });
    expect(screen.getByRole("img")).toBeInTheDocument();
  });

  test("shows waiting status message after QR generation", async () => {
    renderModal({ open: true });
    await waitFor(() => {
      expect(screen.queryByText("Creating...")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Scan the QR code with the Bilibili app")).toBeInTheDocument();
  });
});
