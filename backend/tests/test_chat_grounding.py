"""Tests for build_grounding_prompt content — four-section structure."""

from bibilab.routers.chat import build_grounding_prompt


class TestBuildGroundingPrompt:
    def test_four_section_headers_in_order(self):
        prompt = build_grounding_prompt("English")
        lines = prompt.split("\n")
        headers = [line for line in lines if line.startswith("## ")]
        assert headers == ["## Workflow", "## Grounding", "## Citation", "## Style"], (
            f"Expected four ## headers in order, got: {headers}"
        )

    def test_line_count_under_20(self):
        prompt = build_grounding_prompt("English")
        assert len(prompt.split("\n")) <= 20, f"Prompt has {len(prompt.split(chr(10)))} lines, must be <= 20"

    def test_no_forbidden_substrings(self):
        prompt = build_grounding_prompt("English")
        forbidden = [
            '["1","3"]',
            "CRITICAL CONTEXT",
            "CRITICAL RULES",
            "Rule ",
            "1-3 sentences",
        ]
        for s in forbidden:
            assert s not in prompt, f"Prompt contains forbidden substring: {s!r}"

    def test_contains_verbatim(self):
        assert "verbatim" in build_grounding_prompt("English")

    def test_contains_citation_marker(self):
        assert "[N]" in build_grounding_prompt("English")

    def test_contains_fictional(self):
        assert "fictional" in build_grounding_prompt("English")

    def test_response_language_interpolated_at_least_twice(self):
        prompt = build_grounding_prompt("English")
        assert prompt.count("English") >= 2, (
            f"response_language interpolated {prompt.count('English')} times, expected >= 2"
        )

    def test_citation_section_has_same_line_directive(self):
        """AC7/D7 — `## Citation` instructs same-line placement, no own-line citation."""
        prompt = build_grounding_prompt("English")
        citation_section = prompt.split("## Citation\n", 1)[1].split("\n## ", 1)[0]
        lowered = citation_section.lower()
        assert "same line" in lowered, citation_section
        assert "own line" in lowered, citation_section
