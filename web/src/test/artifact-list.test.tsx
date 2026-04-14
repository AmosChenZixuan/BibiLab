import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ArtifactList } from "@/components/lists/lab/ArtifactList";

const mockArtifacts = [
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
];

vi.mock("@/lib/api", () => {
  return {
    api: {
      listArtifacts: vi.fn().mockResolvedValue([
        {
          id: "artifact-1",
          name: "Brief One",
          type: "brief",
          prompt: "Generate a brief",
          source_ids: ["source-1"],
          status: "done",
          created_at: "2026-04-08T12:00:00Z",
        },
        {
          id: "artifact-2",
          name: "",
          type: "study_guide",
          prompt: "Generate a study guide",
          source_ids: ["source-1", "source-2"],
          status: "generating",
          created_at: "2026-04-08T12:01:00Z",
        },
      ]),
      listJobs: vi.fn().mockResolvedValue([]),
      deleteJob: vi.fn(),
    },
    createApiClient: () => ({
      listArtifacts: vi.fn().mockResolvedValue([
        {
          id: "artifact-1",
          name: "Brief One",
          type: "brief",
          prompt: "Generate a brief",
          source_ids: ["source-1"],
          status: "done",
          created_at: "2026-04-08T12:00:00Z",
        },
        {
          id: "artifact-2",
          name: "",
          type: "study_guide",
          prompt: "Generate a study guide",
          source_ids: ["source-1", "source-2"],
          status: "generating",
          created_at: "2026-04-08T12:01:00Z",
        },
      ]),
      listJobs: vi.fn().mockResolvedValue([]),
      deleteJob: vi.fn(),
    }),
    setCurrentLang: vi.fn(),
  };
});

import { api } from "@/lib/api";

function renderArtifactList(props?: Partial<React.ComponentProps<typeof ArtifactList>>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ArtifactList
          listId="list-1"
          artifacts={mockArtifacts}
          onArtifactsChange={vi.fn()}
          {...props}
        />
      </JobActivityProvider>
    </LanguageProvider>,
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
