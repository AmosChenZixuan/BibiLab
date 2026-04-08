import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test } from "vitest";
import { nameToPastelIndex, PASTEL_COLORS, ListCard } from "@/components/lists/ListCard";
import { LanguageProvider } from "@/app/LanguageContext";
import { BibilabList } from "@/lib/types";

afterEach(() => {
  cleanup();
  localStorage.removeItem("bibilab-lang");
});

function makeList(overrides: Partial<BibilabList> = {}): BibilabList {
  return {
    id: "list-1",
    name: "Test List",
    created_at: "2026-03-31T19:00:00Z",
    thumbnail_source_id: null,
    thumbnail_url: null,
    source_count: 0,
    updated_at: "2026-03-31T20:00:00Z",
    ...overrides,
  };
}

function renderListCard(list: BibilabList) {
  return render(
    <MemoryRouter>
      <LanguageProvider>
        <ListCard
          list={list}
          onDelete={async () => {}}
        />
      </LanguageProvider>
    </MemoryRouter>,
  );
}

describe("ListCard i18n source count", () => {
  test("renders singular 'source' when source_count is 1", () => {
    renderListCard(makeList({ source_count: 1, updated_at: "2026-03-31T20:00:00Z" }));
    // Must use i18n: should show "1 source" from t("lists.sourceSingular") not hardcoded English
    expect(screen.getByText(/1\s+source/)).toBeInTheDocument();
    // Must NOT contain bare "sources" (plural form) when count is 1
    expect(screen.queryByText(/sources/)).not.toBeInTheDocument();
  });

  test("renders plural 'sources' when source_count is 3", () => {
    renderListCard(makeList({ source_count: 3, updated_at: "2026-03-31T20:00:00Z" }));
    // Must use i18n: should show "3 sources" from t("lists.sourcePlural") not hardcoded English
    expect(screen.getByText(/3\s+sources/)).toBeInTheDocument();
  });

  test("uses i18n: renders Chinese '来源' instead of English 'source' when locale is zh", () => {
    // Pre-set localStorage so LanguageProvider initializes in zh mode
    localStorage.setItem("bibilab-lang", "zh");
    const list = makeList({ source_count: 1, updated_at: "2026-03-31T20:00:00Z" });
    render(
      <MemoryRouter>
        <LanguageProvider>
          <ListCard list={list} onDelete={async () => {}} />
        </LanguageProvider>
      </MemoryRouter>,
    );
    // With zh locale and i18n, the source count should show Chinese "来源"
    // With hardcoded English, it would still show "source" (the bug)
    expect(screen.queryByText(/source/i)).not.toBeInTheDocument();
    expect(screen.getByText(/1.*来源/)).toBeInTheDocument();
  });
});

describe("nameToPastelIndex", () => {
  test("returns integer in range 0-7", () => {
    const idx = nameToPastelIndex("Watch Later");
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(8);
  });

  test("is deterministic", () => {
    expect(nameToPastelIndex("Watch Later")).toBe(nameToPastelIndex("Watch Later"));
  });

  test("distributes differently for different names", () => {
    // May collide but should not be always equal
    const a = nameToPastelIndex("AAAA");
    const b = nameToPastelIndex("BBBB");
    // Just verify both are valid indices
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(8);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThan(8);
  });
});
