import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { JsonTree } from "@/components/debug/JsonTree";

afterEach(() => {
  cleanup();
});

describe("JsonTree", () => {
  it("renders strings as plain text", () => {
    render(<JsonTree value="hello" />);
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders numbers and booleans plainly", () => {
    render(<JsonTree value={42} />);
    expect(screen.getByText("42")).toBeInTheDocument();
    render(<JsonTree value={true} />);
    expect(screen.getByText("true")).toBeInTheDocument();
  });

  it("renders null as '—'", () => {
    render(<JsonTree value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders arrays as bullet lists", () => {
    render(<JsonTree value={["a", "b"]} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("renders objects as key-value pairs", () => {
    render(<JsonTree value={{ foo: 1, bar: "x" }} />);
    expect(screen.getByText("foo")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("collapses deep objects by default; click to expand", () => {
    render(<JsonTree value={{ a: { b: { c: "deep" } } }} />);
    expect(screen.queryByText("deep")).toBeNull();
    fireEvent.click(screen.getByText("b"));
    expect(screen.getByText("deep")).toBeInTheDocument();
  });
});
