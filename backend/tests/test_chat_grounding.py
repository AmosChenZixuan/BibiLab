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

    def test_line_count_bounded(self):
        # The Workflow section is an agentic-RAG operating manual (Plan→Act→Reflect +
        # a 10-shape playbook + stopping rules), so it is line-structured rather than a
        # terse blurb. Still bounded to catch unbounded prompt bloat.
        prompt = build_grounding_prompt("en")
        assert len(prompt.split("\n")) <= 45, f"Prompt has {len(prompt.split(chr(10)))} lines, must be <= 45"

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
        """Grounding rule points at the per-section fence structure."""
        prompt = build_grounding_prompt("en")
        grounding = prompt.split("## Grounding\n", 1)[1].split("\n## ", 1)[0]
        assert '===== [N] "title" · Section S =====' in grounding, grounding
        assert "across a fence" in grounding.lower(), grounding

    def test_grounding_prompt_two_tool_surface(self):
        p = build_grounding_prompt("en")
        assert "find_passages" in p and "read_section" in p
        # read_source is GONE — section granularity replaced it
        assert "read_source" not in p
        # deleted tools must NOT appear
        for dead in ("survey", "retrieve_scoped", "query_list_metadata", "generate_report"):
            assert dead not in p
        # escalation rule present
        assert "read_section" in p and "in full" in p.lower()

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
        ('第N集讲了什么') as a find_passages example — coverage routes to an
        OUTLINE retrieve (find_passages with the matching facet)."""
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        assert "讲了什么" not in FIND_PASSAGES_TOOL.description

    def test_grounding_prompt_routes_coverage_to_outline(self):
        """A named-episode coverage question must be answered from a fresh
        OUTLINE retrieve (find_passages with the matching facet), not from
        conversation history or a per-section read."""
        p = build_grounding_prompt("en")
        # the coverage→outline directive names the pattern
        assert "讲了什么" in p
        # the #396 exemption is present (outline retrieve is the only path)
        assert "outline" in p.lower()
        # the weak "only when needed" escalation wording is gone
        assert "only when needed" not in p

    def test_grounding_prompt_has_396_coverage_exemption(self):
        """#396 coverage-confabulation: a coverage question MUST NOT be
        answered from conversation history — always retrieve the outline."""
        p = build_grounding_prompt("en").lower()
        # the exemption text names both the symptom and the rule
        assert "coverage question" in p
        # explicit prohibition: history is never the source for coverage
        assert "never answer a coverage question" in p
        assert "from conversation history" in p or "from history" in p

    def test_workflow_encodes_react_loop(self):
        """The Workflow is a Plan→Act→Reflect operating manual (the #523 rewrite)."""
        workflow = build_grounding_prompt("en").split("## Grounding", 1)[0]
        for phase in ("PLAN", "ACT", "REFLECT"):
            assert phase in workflow, workflow
        assert "Plan → Act → Reflect" in workflow

    def test_workflow_has_playbook_with_failure_shapes(self):
        """The need-shape → strategy playbook names the failure-prone shapes the
        rewrite must handle (comparison, multi-hop, enumeration, causal-absent)."""
        workflow = build_grounding_prompt("en").split("## Grounding", 1)[0]
        assert "Playbook" in workflow
        lowered = workflow.lower()
        for shape in ("comparison", "multi-hop", "enumeration", "causal"):
            assert shape in lowered, shape

    def test_workflow_has_stopping_discipline(self):
        """The three stopping rules that fix thrash + facet drop + redundant read."""
        workflow = build_grounding_prompt("en").split("## Grounding", 1)[0]
        lowered = workflow.lower()
        # one-retrieval-per-need (anti-thrash); no reword-and-retry of the same need
        assert "one retrieval per need" in lowered
        assert "not allowed" in lowered
        # keep the facet across the turn (anti facet-drop)
        assert "keep the same sequence_number" in lowered
        # answer from read_section, don't re-search (anti redundant-read)
        assert "after read_section, answer from it" in lowered

    def test_style_forbids_narrating_plan_and_reflection(self):
        """`## Style` contract (silent-plan): the reply is the answer only — the
        model must NOT narrate its plan / tool choice / reflection into the output.
        The Plan→Act→Reflect reasoning stays in the model's own thinking, so
        streaming it as visible preamble is disallowed."""
        prompt = build_grounding_prompt("en")
        style = prompt.split("## Style\n", 1)[1].lower()
        assert "do not narrate" in style
        assert "plan" in style and "reflection" in style
        # the direct-answer discipline is retained
        assert "direct" in style
        # and the Workflow PLAN step marks planning as internal, not output
        workflow = prompt.split("## Grounding", 1)[0].lower()
        assert "do not write the plan into your reply" in workflow

    def test_grounding_prompt_citation_distinguishes_verbatim_from_outline(self):
        """The Citation section must instruct: cite [N] ONLY for sections whose
        verbatim you saw (find_passages fragments or read_section). Outline
        summaries are orientation, not citable evidence."""
        p = build_grounding_prompt("en")
        citation = p.split("## Citation\n", 1)[1].split("\n## ", 1)[0]
        # the verbatim-required rule is named
        assert "verbatim" in citation.lower()
        # outline summaries are explicitly NOT evidence
        assert "outline" in citation.lower()
        assert "orientation" in citation.lower() or "not evidence" in citation.lower()
