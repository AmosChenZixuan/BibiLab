from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static, TextArea
from textual.binding import Binding
from textual.screen import ModalScreen

from eval.storage import (
    list_runs as _list_runs,
    load_eval_run,
    load_eval_set,
    load_graded_run,
    save_eval_set,
)


class _QuitConfirm(ModalScreen[bool]):
    CSS = """
    #quit-dialog { width: 40; padding: 1 2; background: $surface; }
    """

    BINDINGS = [
        Binding("y", "quit_yes", "Yes, quit"),
        Binding("n,escape", "quit_no", "No, stay"),
    ]

    def compose(self):
        yield Vertical(
            Static("⚠ Unsaved changes. Quit without saving?"),
            Static("[y] Yes  [n] No"),
            id="quit-dialog",
        )

    def action_quit_yes(self):
        self.dismiss(True)

    def action_quit_no(self):
        self.dismiss(False)


_FIELDS = ("question", "expected_answer_draft")
_FIELD_LABELS_REVIEW = {
    "question": "Question",
    "expected_answer_draft": "Expected Answer",
}


class ReviewApp(App):
    """Eval case review. left/right switch case, up/down select field, enter edit."""

    CSS = """
    #main { padding: 1 2; }
    #status-bar { height: 1; background: $surface; padding: 0 1; }
    .row { padding: 0 1; }
    .row-selected { background: $boost; }
    .field-body { padding: 0 1 1 4; }
    TextArea { margin: 0 1 1 4; height: auto; max-height: 20; }
    """

    BINDINGS = [
        Binding("left,h", "prev_case", "Prev case"),
        Binding("right,l", "next_case", "Next case"),
        Binding("down,j", "next_field", "Next field"),
        Binding("up,k", "prev_field", "Prev field"),
        Binding("space", "toggle_lock", "Lock/unlock"),
        Binding("enter,e", "edit", "Edit"),
        Binding("ctrl+s", "save", "Save"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, eval_set_id: str):
        super().__init__()
        self.eval_set_id = eval_set_id
        self.eval_set = load_eval_set(eval_set_id)
        self.case_index = 0
        self.field_index = 0
        self._dirty = False
        self._editing_field: str | None = None
        self._edit_backup: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            VerticalScroll(id="main"),
            Static("", id="status-bar"),
        )
        yield Footer()

    def on_mount(self):
        for i, case in enumerate(self.eval_set.cases):
            if not case.locked:
                self.case_index = i
                break
        self._refresh()

    def _refresh(self):
        main = self.query_one("#main")
        main.remove_children()
        status = self.query_one("#status-bar")

        if not self.eval_set.cases:
            main.mount(Static("No cases."))
            status.update("")
            return

        self.case_index = max(0, min(self.case_index, len(self.eval_set.cases) - 1))
        case = self.eval_set.cases[self.case_index]
        lock_str = "🔒 LOCKED" if case.locked else "🔓 UNLOCKED"
        dirty_str = "   *unsaved*" if self._dirty else ""
        status.update(
            f"Case {self.case_index + 1}/{len(self.eval_set.cases)} — {case.category}   "
            f"{lock_str}   (←/→ case · ↑/↓ field · enter edit · space lock · ctrl+s save){dirty_str}"
        )

        for i, field in enumerate(_FIELDS):
            sel = "→" if i == self.field_index else " "
            cls = "row row-selected" if i == self.field_index else "row"
            main.mount(Static(f"{sel} {_FIELD_LABELS_REVIEW[field]}:", classes=cls))
            value = getattr(case, field, "")
            if self._editing_field == field:
                ta = TextArea(value, id=f"edit-{field}")
                main.mount(ta)
                ta.focus()
            else:
                main.mount(Static(value or "(empty)", classes="field-body"))


    def action_next_case(self):
        if self._editing_field is not None:
            return
        if self.case_index < len(self.eval_set.cases) - 1:
            self.case_index += 1
            self._refresh()

    def action_prev_case(self):
        if self._editing_field is not None:
            return
        if self.case_index > 0:
            self.case_index -= 1
            self._refresh()

    def action_next_field(self):
        if self._editing_field is not None:
            return
        if self.field_index < len(_FIELDS) - 1:
            self.field_index += 1
            self._refresh()

    def action_prev_field(self):
        if self._editing_field is not None:
            return
        if self.field_index > 0:
            self.field_index -= 1
            self._refresh()

    def action_toggle_lock(self):
        if self._editing_field is not None:
            return
        if self.eval_set.cases:
            case = self.eval_set.cases[self.case_index]
            case.locked = not case.locked
            self._dirty = True
            self._refresh()

    def action_edit(self):
        if not self.eval_set.cases:
            return
        if self._editing_field is not None:
            self._commit_edit()
            return
        field = _FIELDS[self.field_index]
        case = self.eval_set.cases[self.case_index]
        self._edit_backup = getattr(case, field, "")
        self._editing_field = field
        self._refresh()

    def _commit_edit(self):
        if self._editing_field is None:
            return
        case = self.eval_set.cases[self.case_index]
        field = self._editing_field
        try:
            new_val = self.query_one(f"#edit-{field}", TextArea).text
        except Exception as e:
            self.notify(f"Edit lost ({e}); restoring previous value.", severity="error", timeout=5)
            new_val = self._edit_backup
        if field == "question":
            new_val = new_val.strip()
            if not new_val:
                new_val = self._edit_backup
        if getattr(case, field, "") != new_val:
            setattr(case, field, new_val)
            self._dirty = True
        self._editing_field = None
        self._refresh()

    def _cancel_edit(self):
        self._editing_field = None
        self._refresh()

    def action_save(self):
        if self._editing_field is not None:
            self._commit_edit()
        save_eval_set(self.eval_set)
        self._dirty = False
        self._refresh()
        self.notify("Saved!", timeout=2)

    def action_quit_app(self):
        if self._editing_field is not None:
            self._cancel_edit()
        if self._dirty:
            def handle(result: bool):
                if result:
                    self.exit()
            self.push_screen(_QuitConfirm(), handle)
        else:
            self.exit()

    def on_key(self, event):
        if self._editing_field is not None:
            if event.key == "escape":
                self._cancel_edit()
                event.prevent_default()
            elif event.key == "ctrl+s":
                self._commit_edit()
                event.prevent_default()


def run_review_tui(eval_set_id: str):
    app = ReviewApp(eval_set_id)
    app.run()


# -- Config TUI --

_CUSTOM_FIELDS = ("protocol", "model", "base_url", "api_key")
_FIELD_LABELS = {
    "protocol": "Protocol",
    "model": "Model",
    "base_url": "Base URL",
    "api_key": "API Key",
}


@dataclass
class _Row:
    kind: str           # "language" | "profile" | "field"
    profile: str = ""   # for profile/field rows
    field: str = ""     # for field rows only


class ConfigApp(App):
    """Single-screen eval config. j/k navigate, space toggle, enter/e edit, ctrl+s save."""

    CSS = """
    #main { padding: 1 2; }
    #status-bar { height: 1; background: $surface; padding: 0 1; }
    .row { padding: 0 1; }
    .row-selected { background: $boost; }
    .section { color: $accent; padding: 1 0 0 0; }
    .field { padding: 0 1 0 4; }
    Input { margin: 0 1 0 4; }
    """

    BINDINGS = [
        Binding("down,j", "next", "Next"),
        Binding("up,k", "prev", "Prev"),
        Binding("space", "toggle", "Toggle"),
        Binding("enter,e", "edit", "Edit"),
        Binding("ctrl+s", "save", "Save"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        from eval.config import PROFILE_NAMES, load_eval_config

        # Edit as a plain dict (mode="json" so Language enum → "zh"/"en");
        # validate back into EvalConfig on save.
        self.cfg: dict[str, Any] = load_eval_config().model_dump(mode="json")
        self.profile_names = PROFILE_NAMES
        self.rows: list[_Row] = []
        self.cursor = 0
        self._dirty = False
        self._editing_field: tuple[str, str] | None = None  # (profile, field)
        self._edit_backup: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            VerticalScroll(id="main"),
            Static("", id="status-bar"),
        )
        yield Footer()

    def on_mount(self):
        self._rebuild_rows()
        self._refresh()

    def _rebuild_rows(self):
        rows: list[_Row] = [_Row("language"), _Row("backend_url")]
        for name in self.profile_names:
            rows.append(_Row("profile", profile=name))
            entry = self.cfg["profiles"].get(name)
            if isinstance(entry, dict):
                for f in _CUSTOM_FIELDS:
                    rows.append(_Row("field", profile=name, field=f))
        self.rows = rows
        self.cursor = max(0, min(self.cursor, len(self.rows) - 1))

    def _refresh(self):
        main = self.query_one("#main")
        main.remove_children()
        status = self.query_one("#status-bar")
        status.update(
            f"Profile: {self._cursor_profile() or '-'}   "
            f"(space: toggle, enter: edit field, ctrl+s: save){'   *unsaved*' if self._dirty else ''}"
        )

        lang = self.cfg.get("language", "zh")

        profiles_header_mounted = False
        for i, row in enumerate(self.rows):
            sel = "→" if i == self.cursor else " "
            cls = "row row-selected" if i == self.cursor else "row"

            if row.kind == "language":
                main.mount(Static(f"{sel} Language: {lang}    (space: zh↔en)", classes=cls))
            elif row.kind == "backend_url":
                if self._editing_field == ("", "backend_url"):
                    main.mount(Static(f"{sel} Backend URL:", classes=cls))
                    inp = Input(value=self.cfg.get("backend_url", ""), id="field-edit")
                    main.mount(inp)
                    inp.focus()
                else:
                    main.mount(Static(f"{sel} Backend URL: {self.cfg.get('backend_url', '')}    (enter: edit)", classes=cls))
            elif row.kind == "profile":
                if not profiles_header_mounted:
                    main.mount(Static("Profiles", classes="section"))
                    profiles_header_mounted = True
                entry = self.cfg["profiles"].get(row.profile)
                if entry is None:
                    summary = "(backend)"
                else:
                    summary = f"{entry.get('protocol', '')} · {entry.get('model', '')} · {entry.get('base_url', '')}"
                main.mount(Static(f"{sel} {row.profile:<10} {summary}", classes=cls))
            elif row.kind == "field":
                entry = self.cfg["profiles"].get(row.profile) or {}
                if self._editing_field == (row.profile, row.field):
                    main.mount(Static(f"{sel} {_FIELD_LABELS[row.field]}:", classes=cls))
                    is_secret = row.field == "api_key"
                    inp = Input(
                        value=entry.get(row.field, ""),
                        password=is_secret,
                        id="field-edit",
                    )
                    main.mount(inp)
                    inp.focus()
                else:
                    val = entry.get(row.field, "")
                    if row.field == "api_key" and val:
                        display = "***"
                    else:
                        display = val or "(empty)"
                    main.mount(Static(f"{sel}   {_FIELD_LABELS[row.field]}: {display}", classes=cls))

    def _cursor_profile(self) -> str:
        if not self.rows:
            return ""
        row = self.rows[self.cursor]
        return row.profile if row.kind in ("profile", "field") else ""

    def action_next(self):
        if self._editing_field is not None:
            return
        if self.cursor < len(self.rows) - 1:
            self.cursor += 1
            self._refresh()

    def action_prev(self):
        if self._editing_field is not None:
            return
        if self.cursor > 0:
            self.cursor -= 1
            self._refresh()

    def action_toggle(self):
        if self._editing_field is not None:
            return
        row = self.rows[self.cursor]
        if row.kind == "language":
            self.cfg["language"] = "en" if self.cfg.get("language") == "zh" else "zh"
            self._dirty = True
            self._refresh()
        elif row.kind == "profile":
            entry = self.cfg["profiles"].get(row.profile)
            if entry is None:
                self.cfg["profiles"][row.profile] = self._backend_seed()
            else:
                self.cfg["profiles"][row.profile] = None
            self._dirty = True
            self._rebuild_rows()
            self._refresh()

    def _backend_seed(self) -> dict:
        try:
            from eval import api
            ai = api.get_backend_ai()
            return {
                "protocol": ai.get("protocol", ""),
                "model": ai.get("model", ""),
                "base_url": ai.get("base_url") or "",
                # The config endpoint masks api_key. Leave it empty: an empty
                # override field is treated as unset, so calls inherit the
                # backend's real key.
                "api_key": "",
            }
        except Exception as e:
            self.notify(f"Backend config unavailable: {e}", severity="warning", timeout=5)
            # All-empty seed = all-unset: if saved untouched, the profile sends
            # no override instead of guessing "openai" at the backend.
            return {"protocol": "", "model": "", "base_url": "", "api_key": ""}

    def action_edit(self):
        if self._editing_field is not None:
            self._commit_field()
            return
        row = self.rows[self.cursor]
        if row.kind == "backend_url":
            self._edit_backup = self.cfg.get("backend_url", "")
            self._editing_field = ("", "backend_url")
            self._refresh()
            return
        if row.kind != "field":
            return
        entry = self.cfg["profiles"].get(row.profile) or {}
        self._edit_backup = entry.get(row.field, "")
        self._editing_field = (row.profile, row.field)
        self._refresh()

    def _commit_field(self):
        if self._editing_field is None:
            return
        try:
            inp = self.query_one("#field-edit", Input)
            new_val = inp.value
        except Exception as e:
            self.notify(f"Edit lost ({e}); restoring previous value.", severity="error", timeout=5)
            new_val = self._edit_backup
        profile, field = self._editing_field
        if not profile and field == "backend_url":
            if self.cfg.get("backend_url", "") != new_val:
                self.cfg["backend_url"] = new_val
                self._dirty = True
        else:
            entry = self.cfg["profiles"].get(profile)
            if isinstance(entry, dict):
                if entry.get(field, "") != new_val:
                    entry[field] = new_val
                    self._dirty = True
        self._editing_field = None
        self._refresh()

    def _cancel_field(self):
        self._editing_field = None
        self._refresh()

    def action_save(self):
        from eval.config import EvalConfig, save_eval_config

        if self._editing_field is not None:
            self._commit_field()
        try:
            validated = EvalConfig.model_validate(self.cfg)
        except Exception as e:
            self.notify(f"Config invalid: {e}", severity="error", timeout=8)
            return
        save_eval_config(validated)
        self._dirty = False
        self.notify("Saved!", timeout=2)

    def action_quit_app(self):
        if self._editing_field is not None:
            self._cancel_field()
        if self._dirty:
            def handle(result: bool):
                if result:
                    self.exit()
            self.push_screen(_QuitConfirm(), handle)
        else:
            self.exit()

    def on_key(self, event):
        if self._editing_field is not None:
            if event.key == "escape":
                self._cancel_field()
                event.prevent_default()
            elif event.key == "enter":
                self._commit_field()
                event.prevent_default()


def run_config_tui():
    app = ConfigApp()
    app.run()


# -- Report TUI --

from eval.reporter import aggregate_scores  # noqa: E402


def _fmt_diff(dv: float) -> str:
    if dv > 0:
        return f"(+{dv})"
    if dv < 0:
        return f"({dv})"
    return "(=)"


class _CompareModal(ModalScreen[str | None]):
    CSS = """
    #compare-dialog { width: 80; padding: 1 2; background: $surface; }
    .crow { padding: 0 1; }
    .crow-selected { background: $boost; }
    """

    BINDINGS = [
        Binding("down,j", "next", "Next"),
        Binding("up,k", "prev", "Prev"),
        Binding("enter", "pick", "Pick"),
        Binding("c", "clear", "Clear"),
        Binding("escape,q", "cancel", "Cancel"),
    ]

    def __init__(self, runs: list[tuple[str, str, str]]):
        super().__init__()
        self.runs = runs
        self.cursor = 0

    def compose(self):
        self._box = Vertical(id="compare-dialog")
        yield self._box

    def on_mount(self):
        self._refresh()

    def _refresh(self):
        self._box.remove_children()
        self._box.mount(Static("Compare against: (enter pick · c clear · esc cancel)"))
        if not self.runs:
            self._box.mount(Static("(no other graded runs)", classes="crow"))
            return
        for i, (rid, ts, model) in enumerate(self.runs):
            sel = "→" if i == self.cursor else " "
            cls = "crow crow-selected" if i == self.cursor else "crow"
            self._box.mount(Static(f"{sel} {rid[:8]}  {ts}  {model}", classes=cls))

    def action_next(self):
        if self.runs and self.cursor < len(self.runs) - 1:
            self.cursor += 1
            self._refresh()

    def action_prev(self):
        if self.runs and self.cursor > 0:
            self.cursor -= 1
            self._refresh()

    def action_pick(self):
        if self.runs:
            self.dismiss(self.runs[self.cursor][0])

    def action_clear(self):
        self.dismiss("")  # sentinel: clear compare

    def action_cancel(self):
        self.dismiss(None)


class ReportApp(App):
    """Eval report. Root: per-category aggregate; enter dives into case-by-case."""

    CSS = """
    #main { padding: 1 2; }
    #status-bar { height: 1; background: $surface; padding: 0 1; }
    .row { padding: 0 1; }
    .row-selected { background: $boost; }
    .header { color: $accent; padding: 0 0 1 0; }
    .judge { padding: 0 1 1 2; }
    """

    BINDINGS = [
        Binding("down,j", "next", "Next"),
        Binding("up,k", "prev", "Prev"),
        Binding("right,l", "next_case", "Next case"),
        Binding("left,h", "prev_case", "Prev case"),
        Binding("enter", "enter_cat", "Open"),
        Binding("c", "compare", "Compare"),
        Binding("escape,backspace", "back", "Back"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, run_id: str, compare_run_id: str | None = None):
        super().__init__()
        self.run_id = run_id
        self.compare_run_id = compare_run_id
        self.view = "cats"        # "cats" | "cases"
        self.cat_cursor = 0
        self.case_cursor = 0
        self.active_cat = ""
        self._load()

    def _load(self):
        self.gr = load_graded_run(self.run_id)
        self.eval_run = load_eval_run(self.run_id)
        self.eval_set = load_eval_set(self.eval_run.eval_set_id)
        self.case_map = {c.id: c for c in self.eval_set.cases}
        self.cat_map = {c.id: c.category for c in self.eval_set.cases}
        self.agg = aggregate_scores(self.gr.grades, self.cat_map)
        self.grade_by_id = {g.case_id: g for g in self.gr.grades}
        self.answer_by_id = {c.case_id: c for c in self.eval_run.cases}
        self.cats = sorted(c for c in self.agg if c != "overall")
        self.cases_by_cat: dict[str, list[str]] = {}
        for g in self.gr.grades:
            cat = self.cat_map.get(g.case_id, "?")
            self.cases_by_cat.setdefault(cat, []).append(g.case_id)

        self.prev_agg: dict | None = None
        self.prev_grade_by_id: dict = {}
        self.compare_model: str | None = None
        self._compare_missing: str | None = None
        if self.compare_run_id:
            try:
                prev_gr = load_graded_run(self.compare_run_id)
                prev_run = load_eval_run(self.compare_run_id)
                prev_es = load_eval_set(prev_run.eval_set_id)
                prev_cat_map = {c.id: c.category for c in prev_es.cases}
                self.prev_agg = aggregate_scores(prev_gr.grades, prev_cat_map)
                self.prev_grade_by_id = {g.case_id: g for g in prev_gr.grades}
                self.compare_model = prev_run.test_profile.model
            except FileNotFoundError:
                missing = self.compare_run_id
                self.compare_run_id = None
                # Defer notify until after mount — App may not be running yet during __init__._load
                self._compare_missing = missing

    def _agg_diff(self, cat: str, dim: str) -> str:
        if not self.prev_agg or cat not in self.prev_agg:
            return ""
        cur = self.agg.get(cat, {}).get(dim, 0.0)
        prev = self.prev_agg.get(cat, {}).get(dim, 0.0)
        return " " + _fmt_diff(round(cur - prev, 1))

    def _case_diff(self, cid: str, attr: str) -> str:
        prev = self.prev_grade_by_id.get(cid)
        if not prev:
            return ""
        cur_v = getattr(self.grade_by_id[cid], attr, None)
        prev_v = getattr(prev, attr, None)
        if cur_v is None or prev_v is None:
            return ""
        return " " + _fmt_diff(cur_v - prev_v)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            VerticalScroll(id="main"),
            Static("", id="status-bar"),
        )
        yield Footer()

    def on_mount(self):
        self._refresh()
        if self._compare_missing:
            self.notify(f"Compare run not found: {self._compare_missing}", severity="warning", timeout=5)
            self._compare_missing = None

    def _header_text(self) -> str:
        test_model = self.eval_run.test_profile.model
        grade_model = self.gr.grade_profile.model
        overall = self.agg.get("overall", {})
        cmp_line = f"   vs {self.compare_model} ({self.compare_run_id[:8]})" if self.compare_model else ""
        return (
            f"Eval Set: {self.eval_set.id[:8]}   Cases: {len(self.gr.grades)}   "
            f"Test: {test_model}   Grade: {grade_model}{cmp_line}\n"
            f"OVERALL  CR={overall.get('context_relevance', 0.0)}{self._agg_diff('overall', 'context_relevance')}  "
            f"G={overall.get('groundedness', 0.0)}{self._agg_diff('overall', 'groundedness')}  "
            f"AR={overall.get('answer_relevance', 0.0)}{self._agg_diff('overall', 'answer_relevance')}"
        )

    def _refresh(self):
        main = self.query_one("#main")
        main.remove_children()
        status = self.query_one("#status-bar")

        main.mount(Static(self._header_text(), classes="header"))

        if self.view == "cats":
            main.mount(Static(f"{'category':<14} {'n':>3}  CR     G      AR", classes="row"))
            for i, cat in enumerate(self.cats):
                s = self.agg[cat]
                n = len(self.cases_by_cat.get(cat, []))
                sel = "→" if i == self.cat_cursor else " "
                cls = "row row-selected" if i == self.cat_cursor else "row"
                main.mount(Static(
                    f"{sel}{cat:<14} {n:>3}  "
                    f"CR={s.get('context_relevance', 0.0)}{self._agg_diff(cat, 'context_relevance')}  "
                    f"G={s.get('groundedness', 0.0)}{self._agg_diff(cat, 'groundedness')}  "
                    f"AR={s.get('answer_relevance', 0.0)}{self._agg_diff(cat, 'answer_relevance')}",
                    classes=cls,
                ))
            status.update(
                f"Category {self.cat_cursor + 1}/{len(self.cats)}   "
                f"(↑/↓ select · enter open · c compare · q quit)"
            )
            return

        # cases view
        cids = self.cases_by_cat.get(self.active_cat, [])
        if not cids:
            main.mount(Static(f"(no cases in {self.active_cat})", classes="row"))
            status.update("(esc back · q quit)")
            return
        self.case_cursor = max(0, min(self.case_cursor, len(cids) - 1))
        cid = cids[self.case_cursor]
        case = self.case_map.get(cid)
        g = self.grade_by_id[cid]
        rc = self.answer_by_id.get(cid)
        answer = (rc.answer or "(no answer)") if rc else "(missing)"
        if rc and rc.error:
            answer = f"ERROR: {rc.error}"
        q = case.question if case else cid
        expected = case.expected_answer_draft if case else ""

        main.mount(Static(
            f"[{self.active_cat}] case {self.case_cursor + 1}/{len(cids)}  "
            f"CR={g.context_relevance}{self._case_diff(cid, 'context_relevance')}  "
            f"G={g.groundedness}{self._case_diff(cid, 'groundedness')}  "
            f"AR={g.answer_relevance}{self._case_diff(cid, 'answer_relevance')}",
            classes="row",
        ))
        main.mount(Static(""))
        main.mount(Static(f"Q: {q}", classes="row"))
        main.mount(Static(""))
        main.mount(Static(f"A: {answer}", classes="row"))
        main.mount(Static(""))
        if expected:
            main.mount(Static(f"Expected: {expected}", classes="row"))
            main.mount(Static(""))
        main.mount(Static("Judge:", classes="row"))
        main.mount(Static(f"  CR({g.context_relevance}/5): {g.context_relevance_reasoning}", classes="judge"))
        main.mount(Static(f"  G ({g.groundedness}/5): {g.groundedness_reasoning}", classes="judge"))
        main.mount(Static(f"  AR({g.answer_relevance}/5): {g.answer_relevance_reasoning}", classes="judge"))

        status.update(
            f"[{self.active_cat}] case {self.case_cursor + 1}/{len(cids)}   "
            f"(←/→ case · esc back · q quit)"
        )

    def action_next(self):
        if self.view == "cats":
            if self.cat_cursor < len(self.cats) - 1:
                self.cat_cursor += 1
                self._refresh()

    def action_prev(self):
        if self.view == "cats":
            if self.cat_cursor > 0:
                self.cat_cursor -= 1
                self._refresh()

    def action_next_case(self):
        if self.view != "cases":
            return
        cids = self.cases_by_cat.get(self.active_cat, [])
        if self.case_cursor < len(cids) - 1:
            self.case_cursor += 1
            self._refresh()

    def action_prev_case(self):
        if self.view != "cases":
            return
        if self.case_cursor > 0:
            self.case_cursor -= 1
            self._refresh()

    def action_enter_cat(self):
        if self.view != "cats" or not self.cats:
            return
        self.active_cat = self.cats[self.cat_cursor]
        self.case_cursor = 0
        self.view = "cases"
        self._refresh()

    def action_back(self):
        if self.view == "cases":
            self.view = "cats"
            self._refresh()

    def action_compare(self):
        candidates: list[tuple[str, str, str]] = []
        for r in _list_runs(self.eval_set.id):
            if r.id == self.run_id:
                continue
            try:
                load_graded_run(r.id)
            except FileNotFoundError:
                continue
            candidates.append((r.id, r.timestamp, r.test_profile.model))

        def handle(result: str | None):
            if result is None:
                return
            self.compare_run_id = result or None
            self._load()
            self._refresh()

        self.push_screen(_CompareModal(candidates), handle)

    def action_quit_app(self):
        self.exit()


def run_report_tui(run_id: str, compare_run_id: str | None = None):
    app = ReportApp(run_id, compare_run_id)
    app.run()
