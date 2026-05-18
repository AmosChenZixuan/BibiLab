import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestFacets } from "@/components/lists/DigestFacets";
import { LanguageProvider } from "@/app/LanguageContext";

afterEach(() => {
  cleanup();
});

function setup(props: Partial<React.ComponentProps<typeof DigestFacets>> = {}) {
  const onSave = props.onSave ?? vi.fn().mockResolvedValue(undefined);
  const onExitEdit = props.onExitEdit ?? vi.fn();
  render(
    <LanguageProvider>
      <DigestFacets
        facets={props.facets ?? { seriesName: "罗翔说刑法", sequenceNumber: 8, seasonNumber: null }}
        editing={props.editing ?? false}
        onSave={onSave}
        onExitEdit={onExitEdit}
      />
    </LanguageProvider>,
  );
  return { onSave, onExitEdit };
}

describe("DigestFacets read", () => {
  test("renders series · season · number order, no kind", () => {
    setup({ facets: { seriesName: "罗翔说刑法", sequenceNumber: 8, seasonNumber: 2 } });
    expect(screen.getByText("罗翔说刑法 · S 2 · No. 8")).toBeInTheDocument();
  });

  test("all null renders nothing", () => {
    const { container } = render(
      <LanguageProvider>
        <DigestFacets
          facets={{ seriesName: null, sequenceNumber: null, seasonNumber: null }}
          editing={false}
          onSave={vi.fn()}
          onExitEdit={vi.fn()}
        />
      </LanguageProvider>,
    );
    expect(container.textContent).toBe("");
  });
});

describe("DigestFacets edit", () => {
  test.each(["0", "abc", "-1", "1.5"])("invalid number %s blocks save", async (bad) => {
    const { onSave } = setup({ editing: true });
    fireEvent.change(screen.getByLabelText("No.") as HTMLInputElement, { target: { value: bad } });
    fireEvent.click(screen.getByText("Save"));
    expect(await screen.findByText("Must be a whole number ≥ 1")).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  test("valid save sends patch and exits edit", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const { onExitEdit } = setup({
      editing: true,
      facets: { seriesName: "A", sequenceNumber: 1, seasonNumber: null },
      onSave,
    });
    fireEvent.change(screen.getByLabelText("Series"), { target: { value: "B" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({ series_name: "B", sequence_number: 1, season_number: null }),
    );
    expect(onExitEdit).toHaveBeenCalled();
  });

  test("failed save shows localized error and stays in edit", async () => {
    const onSave = vi.fn().mockRejectedValue(new Error("boom"));
    const { onExitEdit } = setup({
      editing: true,
      facets: { seriesName: "A", sequenceNumber: 1, seasonNumber: null },
      onSave,
    });
    fireEvent.click(screen.getByText("Save"));
    expect(await screen.findByText("boom")).toBeInTheDocument();
    expect(onExitEdit).not.toHaveBeenCalled();
  });
});
