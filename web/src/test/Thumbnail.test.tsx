import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Thumbnail } from "../components/ui/Thumbnail";

afterEach(() => {
  cleanup();
});

describe("Thumbnail", () => {
  it("renders_placeholder_when_no_source_and_no_remote_url", () => {
    render(<Thumbnail />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
    expect(screen.getByTestId("thumbnail-placeholder")).toBeInTheDocument();
  });

  it("renders_img_with_local_source_url_when_source_provided", () => {
    render(<Thumbnail source={{ id: "abc", cover_url: "https://example.com/c.jpg" }} alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    expect(img).toHaveAttribute("src", "/api/sources/abc/cover");
    expect(img).toHaveClass("opacity-0");
  });

  it("transitions_to_opacity_100_on_load", () => {
    render(<Thumbnail source={{ id: "abc", cover_url: "https://example.com/c.jpg" }} alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.load(img);
    expect(img).toHaveClass("opacity-100");
  });

  it("falls_back_to_remote_proxy_on_local_error", () => {
    render(<Thumbnail source={{ id: "abc", cover_url: "https://example.com/c.jpg" }} alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.error(img);
    expect(img).toHaveAttribute(
      "src",
      "/api/proxy/cover?url=https%3A%2F%2Fexample.com%2Fc.jpg"
    );
  });

  it("stays_on_placeholder_when_source_has_null_cover_url_and_local_errors", () => {
    render(<Thumbnail source={{ id: "abc", cover_url: null }} />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
  });

  it("remote_only_mode_uses_proxy_url", () => {
    render(<Thumbnail remoteUrl="https://example.com/r.jpg" alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    expect(img).toHaveAttribute(
      "src",
      "/api/proxy/cover?url=https%3A%2F%2Fexample.com%2Fr.jpg"
    );
  });

  it("remote_only_error_stays_on_placeholder", () => {
    render(<Thumbnail remoteUrl="https://example.com/r.jpg" alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.error(img);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
    expect(screen.getByTestId("thumbnail-placeholder")).toBeInTheDocument();
  });

  it("empty_remote_url_renders_placeholder_only", () => {
    const { rerender } = render(<Thumbnail remoteUrl="" />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();

    rerender(<Thumbnail remoteUrl={null} />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();

    rerender(<Thumbnail remoteUrl="   " />);
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
  });

  it("className_applies_to_wrapper_div_not_img", () => {
    render(<Thumbnail remoteUrl="r.jpg" className="custom-wrapper" alt="test" />);
    const wrapper = screen.getByTestId("thumbnail-wrapper");
    expect(wrapper).toHaveClass("custom-wrapper");
    const img = screen.getByTestId("thumbnail-img");
    expect(img).not.toHaveClass("custom-wrapper");
  });

  it("caller_cannot_override_internal_onError", () => {
    const callerOnError = vi.fn();
    render(<Thumbnail remoteUrl="r.jpg" onError={callerOnError} alt="test" />);
    const img = screen.getByTestId("thumbnail-img");
    fireEvent.error(img);
    expect(callerOnError).toHaveBeenCalled();
    expect(screen.queryByTestId("thumbnail-img")).not.toBeInTheDocument();
  });
});
