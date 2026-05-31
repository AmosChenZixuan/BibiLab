import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Thumbnail } from "../components/ui/Thumbnail";

afterEach(() => {
  cleanup();
});

describe("Thumbnail", () => {
  it("renders_placeholder_when_no_src", () => {
    render(<Thumbnail />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
    expect(screen.getByTestId("thumbnail-placeholder")).toBeInTheDocument();
  });

  it("renders_img_with_src", () => {
    render(<Thumbnail src="/api/sources/abc/cover" alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    expect(img).toHaveAttribute("src", "/api/sources/abc/cover");
    expect(img).toHaveClass("opacity-0");
  });

  it("transitions_to_opacity_100_on_load", () => {
    render(<Thumbnail src="/api/sources/abc/cover" alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.load(img);
    expect(img).toHaveClass("opacity-100");
  });

  it("falls_back_to_fallbackSrc_on_primary_error", () => {
    render(<Thumbnail src="/api/sources/abc/cover" fallbackSrc="/api/proxy/cover?url=remote" alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.error(img);
    expect(img).toHaveAttribute("src", "/api/proxy/cover?url=remote");
  });

  it("stays_on_placeholder_when_no_fallback_and_primary_errors", () => {
    render(<Thumbnail src="/api/sources/abc/cover" />);
    fireEvent.error(screen.getByTestId("thumbnail-img"));
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
    expect(screen.getByTestId("thumbnail-placeholder")).toBeInTheDocument();
  });

  it("shows_placeholder_after_fallback_also_errors", () => {
    render(<Thumbnail src="/api/sources/abc/cover" fallbackSrc="/api/proxy/cover?url=remote" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.error(img); // primary fails -> switch to fallback
    fireEvent.error(screen.getByTestId("thumbnail-img")); // fallback fails -> placeholder
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
    expect(screen.getByTestId("thumbnail-placeholder")).toBeInTheDocument();
  });

  it("empty_or_null_src_renders_placeholder_only", () => {
    const { rerender } = render(<Thumbnail src="" />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();

    rerender(<Thumbnail src={null} />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
  });

  it("className_applies_to_wrapper_div_not_img", () => {
    render(<Thumbnail src="/api/proxy/cover?url=remote" className="custom-wrapper" alt="test" />);
    const wrapper = screen.getByTestId("thumbnail-wrapper");
    expect(wrapper).toHaveClass("custom-wrapper");
    expect(screen.getByTestId("thumbnail-img")).not.toHaveClass("custom-wrapper");
  });

  it("caller_onError_still_fires_alongside_internal_handling", () => {
    const callerOnError = vi.fn();
    render(<Thumbnail src="/api/proxy/cover?url=remote" onError={callerOnError} alt="test" />);
    fireEvent.error(screen.getByTestId("thumbnail-img"));
    expect(callerOnError).toHaveBeenCalled();
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
  });
});
