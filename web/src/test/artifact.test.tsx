import { describe, expect, test } from "vitest";

import type { Artifact, ArtifactStatus, ArtifactType } from "@/lib/types";

describe("Artifact types", () => {
  test("ArtifactStatus is union of generating, done, error", () => {
    const statuses: ArtifactStatus[] = ["generating", "done", "error"];
    expect(statuses).toBeDefined();
  });

  test("ArtifactType is union of brief, study_guide, blog_post, custom_report", () => {
    const types: ArtifactType[] = ["brief", "study_guide", "blog_post", "custom_report"];
    expect(types).toBeDefined();
  });

  test("Artifact has required fields", () => {
    const artifact: Artifact = {
      id: "artifact-1",
      name: "My Artifact",
      type: "brief",
      prompt: "Generate a brief",
      source_ids: ["source-1", "source-2"],
      status: "done",
      created_at: "2026-04-08T12:00:00Z",
    };
    expect(artifact.id).toBe("artifact-1");
    expect(artifact.name).toBe("My Artifact");
    expect(artifact.type).toBe("brief");
    expect(artifact.status).toBe("done");
    expect(artifact.error).toBeUndefined();
  });

  test("Artifact can have error field when status is error", () => {
    const artifact: Artifact = {
      id: "artifact-2",
      name: "Failed Artifact",
      type: "study_guide",
      prompt: "Generate a study guide",
      source_ids: ["source-1"],
      status: "error",
      created_at: "2026-04-08T12:00:00Z",
      error: "Generation failed: model timeout",
    };
    expect(artifact.status).toBe("error");
    expect(artifact.error).toBe("Generation failed: model timeout");
  });
});
