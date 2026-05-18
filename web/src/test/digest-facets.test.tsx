import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestFacets } from "@/components/lists/DigestFacets";
import { LanguageProvider } from "@/app/LanguageContext";

afterEach(() => {
  cleanup();
});

function setup(props: Partial<React.ComponentProps<typeof DigestFacets>> = {}) {
  const onSave = props.onSave ?? vi.fn().mockResolvedValue(undefined);
  render(
    <LanguageProvider>
      <DigestFacets
        seriesName={props.seriesName ?? "罗翔说刑法"}
        sequenceNumber={props.sequenceNumber ?? 8}
        seasonNumber={props.seasonNumber ?? null}
        editing={props.editing ?? false}
        onSave={onSave}
        onExitEdit={props.onExitEdit ?? vi.fn()}
      />
    </LanguageProvider>,
  );
  return { onSave };
}

describe("DigestFacets read", () => {
  test("renders only non-null segments, no kind", () => {
    setup({ seriesName: "罗翔说刑法", sequenceNumber: 8, seasonNumber: 2 });
    expect(screen.getByText(/罗翔说刑法/)).toBeInTheDocument();
    expect(screen.getByText(/No\.\s*8/)).toBeInTheDocument();
    expect(screen.getByText(/S\s*2/)).toBeInTheDocument();
  });

  test("all null renders nothing", () => {
    const { container } = render(
      <LanguageProvider>
        <DigestFacets
          seriesName={null}
          sequenceNumber={null}
          seasonNumber={null}
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
  test("invalid number blocks save", async () => {
    const { onSave } = setup({ editing: true, sequenceNumber: 8 });
    const num = screen.getByLabelText("No.") as HTMLInputElement;
    fireEvent.change(num, { target: { value: "0" } });
    fireEvent.click(screen.getByText("Save"));
    expect(await screen.findByText("Must be a whole number ≥ 1")).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  test("valid save sends patch and exits edit", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const onExitEdit = vi.fn();
    render(
      <LanguageProvider>
        <DigestFacets
          seriesName="A"
          sequenceNumber={1}
          seasonNumber={null}
          editing
          onSave={onSave}
          onExitEdit={onExitEdit}
        />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText("Series"), { target: { value: "B" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({ series_name: "B", sequence_number: 1, season_number: null }),
    );
    expect(onExitEdit).toHaveBeenCalled();
  });
});
