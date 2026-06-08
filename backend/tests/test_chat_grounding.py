"""Tests for build_grounding_prompt content — four-section structure."""

from bibilab.routers.chat import build_grounding_prompt


class TestBuildGroundingPrompt:
    def test_four_section_headers_in_order(self):
        prompt = build_grounding_prompt("en")
        lines = prompt.split("\n")
        headers = [line for line in lines if line.startswith("## ")]
        assert headers == ["## Workflow", "## Grounding", "## Citation", "## Style"], (
            f"Expected four ## headers in order, got: {headers}"
        )

    def test_line_count_under_30(self):
        prompt = build_grounding_prompt("en")
        assert len(prompt.split("\n")) <= 30, f"Prompt has {len(prompt.split(chr(10)))} lines, must be <= 30"

    def test_no_forbidden_substrings(self):
        prompt = build_grounding_prompt("en")
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
        assert "verbatim" in build_grounding_prompt("en")

    def test_contains_citation_marker(self):
        assert "[N]" in build_grounding_prompt("en")

    def test_contains_fictional(self):
        assert "fictional" in build_grounding_prompt("en")

    def test_response_language_single_directive_at_tail(self):
        # code "en" maps to readable "English"; the directive appears
        # exactly once, as the literal last line (recency, no repetition).
        prompt = build_grounding_prompt("en")
        assert prompt.count("English") == 1, (
            f"response_language interpolated {prompt.count('English')} times, expected exactly 1"
        )
        assert prompt.rstrip().endswith("Respond in English."), prompt[-60:]

    def test_citation_section_has_same_line_directive(self):
        """AC7/D7 — `## Citation` instructs same-line placement."""
        prompt = build_grounding_prompt("en")
        citation_section = prompt.split("## Citation\n", 1)[1].split("\n## ", 1)[0]
        lowered = citation_section.lower()
        assert "same line" in lowered, citation_section

    def test_grounding_section_has_fence_pointer(self):
        """Grounding rule points at the per-source fence structure."""
        prompt = build_grounding_prompt("en")
        grounding = prompt.split("## Grounding\n", 1)[1].split("\n## ", 1)[0]
        assert "===== Source [N] =====" in grounding, grounding
        assert "across a fence" in grounding.lower(), grounding

    def test_grounding_prompt_two_tool_surface(self):
        p = build_grounding_prompt("en")
        assert "find_passages" in p and "read_source" in p
        # deleted tools must NOT appear
        for dead in ("survey", "retrieve_scoped", "query_list_metadata", "generate_report"):
            assert dead not in p
        # escalation rule present
        assert "read_source" in p and "in full" in p.lower()

    def test_grounding_prompt_no_coverage_rule_in_prompt(self):
        """No-coverage behaviour lives in the static prompt now (the backend
        _NO_COVERAGE_NOTE injection is deleted) — the prompt must instruct refusal
        on an empty find_passages result."""
        p = build_grounding_prompt("en").lower()
        assert "no excerpts" in p or "no content" in p
        assert "outside knowledge" in p

    def test_grounding_prompt_degraded_scope_rules_in_prompt(self):
        """The DIRECTIVE half of the fact/directive split lives here:
        the no-match and empty-transcript behaviours are prompt rules; the tool emits
        only the matching fact."""
        p = build_grounding_prompt("en").lower()
        # facet no-match → state degraded scope
        assert "matched no source" in p
        # empty transcript → can't answer, don't infer from metadata
        assert "no transcript available" in p
        assert "not available yet" in p

    def test_grounding_prompt_has_no_tool_and_history_rules(self):
        p = build_grounding_prompt("en")
        # trivial-message no-tool anchor
        assert "without calling" in p.lower() or "no tool" in p.lower()
        # answerable-from-history → no tool
        assert "conversation history" in p.lower() or "already answered" in p.lower()

    def test_find_passages_description_has_no_coverage_example(self):
        """The find_passages tool must NOT demonstrate a coverage question
        ('第N集讲了什么') — that class routes to read_source (live-usage bug fix)."""
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        assert "讲了什么" not in FIND_PASSAGES_TOOL.description

    def test_grounding_prompt_routes_coverage_to_read_source(self):
        """A named-episode coverage question must route to read_source, not
        find_passages-with-facet."""
        p = build_grounding_prompt("en")
        # the coverage→read_source directive is present and names the pattern
        assert "讲了什么" in p
        # the weak "only when needed" escalation wording is gone
        assert "only when needed" not in p
