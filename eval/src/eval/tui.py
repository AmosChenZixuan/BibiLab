from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static, TextArea
from textual.binding import Binding
from textual.screen import ModalScreen

from eval.storage import load_eval_set, save_eval_set


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

        if case.expected_sources:
            main.mount(Static(f"  Expected Sources: {', '.join(case.expected_sources)}", classes="field-body"))

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
        except Exception:
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

        self.cfg: dict[str, Any] = load_eval_config()
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
        rows: list[_Row] = [_Row("language")]
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

        for i, row in enumerate(self.rows):
            sel = "→" if i == self.cursor else " "
            cls = "row row-selected" if i == self.cursor else "row"

            if row.kind == "language":
                main.mount(Static(f"{sel} Language: {lang}    (space: zh↔en)", classes=cls))
                main.mount(Static("Profiles", classes="section"))
            elif row.kind == "profile":
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
            from bibilab.config import load_config
            ai = load_config().ai
            return {
                "protocol": ai.protocol,
                "model": ai.model,
                "base_url": ai.base_url,
                "api_key": ai.api_key,
            }
        except Exception:
            return {"protocol": "openai", "model": "", "base_url": "", "api_key": ""}

    def action_edit(self):
        if self._editing_field is not None:
            self._commit_field()
            return
        row = self.rows[self.cursor]
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
        except Exception:
            new_val = self._edit_backup
        profile, field = self._editing_field
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
        from eval.config import save_eval_config

        if self._editing_field is not None:
            self._commit_field()
        save_eval_config(self.cfg)
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
