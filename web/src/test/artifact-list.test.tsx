import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ArtifactList } from "@/components/lists/lab/ArtifactList";
import { renderWithProviders } from "@/test/utils";

const mockArtifacts = vi.hoisted(() => [
  {
    id: "artifact-1",
    name: "Brief One",
    type: "brief" as const,
    prompt: "Generate a brief",
    source_ids: ["source-1"],
    status: "completed" as const,
    created_at: "2026-04-08T12:00:00Z",
  },
  {
    id: "artifact-2",
    name: "",
    type: "study_guide" as const,
    prompt: "Generate a study guide",
    source_ids: ["source-1", "source-2"],
    status: "generating" as const,
    created_at: "2026-04-08T12:01:00Z",
  },
]);

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  const mockApi = createMockApi({
    listArtifacts: vi.fn().mockResolvedValue(mockArtifacts),
    listJobs: vi.fn().mockResolvedValue([]),
    deleteJob: vi.fn(),
  });
  return {
    api: mockApi,
    createApiClient: () => mockApi,
  };
});

import { api } from "@/lib/api";

function renderArtifactList(props?: Partial<React.ComponentProps<typeof ArtifactList>>) {
  return renderWithProviders(
    <ArtifactList
      listId="list-1"
      artifacts={mockArtifacts}
      onArtifactsChange={vi.fn()}
      {...props}
    />,
    { providers: [LanguageProvider, JobActivityProvider] },
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ArtifactList", () => {
  test("renders all artifacts from props", async () => {
    renderArtifactList();

    await waitFor(() => {
      expect(screen.getByText("Brief One")).toBeInTheDocument();
    });
    expect(screen.getByText("Study Guide")).toBeInTheDocument();
  });
});
