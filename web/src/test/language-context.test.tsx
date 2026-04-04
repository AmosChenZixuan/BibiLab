import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { LanguageProvider, useLanguage } from "@/app/LanguageContext";

function Toggle() {
  const { lang, setLang } = useLanguage();

  return (
    <button type="button" onClick={() => setLang(lang === "en" ? "zh" : "en")}>
      {lang}
    </button>
  );
}

describe("language context", () => {
  test("defaults to en and toggles to zh", () => {
    render(
      <LanguageProvider>
        <Toggle />
      </LanguageProvider>,
    );

    expect(screen.getByText("en")).toBeInTheDocument();
    fireEvent.click(screen.getByText("en"));
    expect(screen.getByText("zh")).toBeInTheDocument();
  });
});
