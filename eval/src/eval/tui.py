from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Header, Footer, TextArea
from textual.binding import Binding
from textual.screen import ModalScreen

from eval.storage import load_eval_set, save_eval_set


class ReviewApp(App):
    """Single-panel eval case review. j/k navigate, space toggles lock, e edits inline."""

    CSS = """
    #main { padding: 1 2; }
    #status-bar { height: 3; background: $surface; padding: 0 1; }
    TextArea { margin-bottom: 1; }
    """

    BINDINGS = [
        Binding("down,j", "next", "Next"),
        Binding("up,k", "prev", "Prev"),
        Binding("space", "toggle_lock", "Lock/unlock"),
        Binding("e", "edit", "Edit"),
        Binding("ctrl+s", "save", "Save"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, eval_set_id: str):
        super().__init__()
        self.eval_set_id = eval_set_id
        self.eval_set = load_eval_set(eval_set_id)
        self.case_index = 0
        self._dirty = False
        self._editing = False
        self._edit_backup: tuple[str, str] = ("", "")

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
            return

        self.case_index = max(0, min(self.case_index, len(self.eval_set.cases) - 1))
        case = self.eval_set.cases[self.case_index]

        sb = [f"Case {self.case_index + 1}/{len(self.eval_set.cases)} — {case.category}"]
        status.update("".join(sb))

        if case.locked:
            main.mount(Static("🔒 LOCKED — included in eval run"))
        else:
            main.mount(Static("🔓 UNLOCKED — skipped during eval run"))

        if self._editing:
            main.mount(Static("Question (edit, Ctrl+S save, Esc cancel):"))
            main.mount(TextArea(case.question, id="edit-question"))
            main.mount(Static("Expected answer:"))
            main.mount(TextArea(case.expected_answer_draft, id="edit-answer"))
        else:
            main.mount(Static(f"Question:\n{case.question}"))
            if case.expected_answer_draft:
                main.mount(Static(f"Expected Answer:\n{case.expected_answer_draft}"))
            if case.expected_sources:
                main.mount(Static(f"Expected Sources: {', '.join(case.expected_sources)}"))

    def action_next(self):
        if self._editing:
            return
        if self.case_index < len(self.eval_set.cases) - 1:
            self.case_index += 1
            self._refresh()

    def action_prev(self):
        if self._editing:
            return
        if self.case_index > 0:
            self.case_index -= 1
            self._refresh()

    def action_toggle_lock(self):
        if self._editing:
            return
        if self.eval_set.cases:
            case = self.eval_set.cases[self.case_index]
            case.locked = not case.locked
            self._dirty = True
            self._refresh()

    def action_edit(self):
        if not self.eval_set.cases:
            return
        case = self.eval_set.cases[self.case_index]
        if self._editing:
            self._commit_edit()
        else:
            self._edit_backup = (case.question, case.expected_answer_draft)
            self._editing = True
            self._refresh()

    def _commit_edit(self):
        case = self.eval_set.cases[self.case_index]
        try:
            q = self.query_one("#edit-question").text
            a = self.query_one("#edit-answer").text
            if q.strip():
                case.question = q.strip()
            case.expected_answer_draft = a.strip()
            self._dirty = True
        except Exception:
            pass
        self._editing = False
        self._refresh()

    def _cancel_edit(self):
        case = self.eval_set.cases[self.case_index]
        case.question, case.expected_answer_draft = self._edit_backup
        self._editing = False
        self._refresh()

    def action_save(self):
        if self._editing:
            self._commit_edit()
        save_eval_set(self.eval_set)
        self._dirty = False
        self.notify("Saved!", timeout=2)

    def action_quit_app(self):
        if self._editing:
            self._cancel_edit()
        if self._dirty:
            self._confirm_quit()
        else:
            self.exit()

    def _confirm_quit(self):
        class QuitScreen(ModalScreen[bool]):
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

        def handle(result: bool):
            if result:
                self.exit()

        self.push_screen(QuitScreen(), handle)

    def on_key(self, event):
        if self._editing and event.key == "escape":
            self._cancel_edit()
            event.prevent_default()
        elif self._editing and event.key == "ctrl+s":
            self._commit_edit()
            event.prevent_default()


def run_review_tui(eval_set_id: str):
    app = ReviewApp(eval_set_id)
    app.run()


# -- Config TUI --

def run_config_tui():
    from eval.config import load_eval_config, save_eval_config

    class ConfigApp(App):
        CSS = """
        #sidebar { width: 30; border: solid $primary; padding: 1; }
        #detail { width: 1fr; padding: 1; }
        """

        BINDINGS = [
            Binding("j", "next_profile", "Next"),
            Binding("k", "prev_profile", "Prev"),
            Binding("t", "toggle_source", "Toggle inherit/custom"),
            Binding("ctrl+s", "save_config", "Save"),
            Binding("q", "quit_app", "Quit"),
        ]

        def __init__(self):
            super().__init__()
            self.cfg = load_eval_config()
            self.profiles = ["generate", "test", "grade"]
            self.selected = 0

        def compose(self):
            yield Header()
            yield Horizontal(
                VerticalScroll(id="sidebar"),
                VerticalScroll(Static("", id="detail")),
            )
            yield Footer()

        def on_mount(self):
            self._refresh()

        def _refresh(self):
            sidebar = self.query_one("#sidebar")
            sidebar.remove_children()
            for i, name in enumerate(self.profiles):
                entry = self.cfg[name]
                source = entry.get("source", "inherit")
                model = entry.get("model", "(from backend)")
                marker = "→" if i == self.selected else " "
                sidebar.mount(Static(
                    f"{marker} [{name}]\n  Model: {model}\n  Source: {source}"
                ))

            detail = self.query_one("#detail")
            name = self.profiles[self.selected]
            entry = self.cfg[name]
            lines = [f"Profile: {name}", f"Source: {entry.get('source', 'inherit')}"]
            if entry.get("source") == "custom":
                lines.append(f"Protocol: {entry.get('protocol', '')}")
                lines.append(f"Model: {entry.get('model', '')}")
                lines.append(f"Base URL: {entry.get('base_url', '')}")
                lines.append(f"API Key: {'***' if entry.get('api_key') else '(not set)'}")
            else:
                lines.append("(reads from ~/.bibilab/config.json ai section)")

            detail.update("\n".join(lines))

        def action_next_profile(self):
            self.selected = (self.selected + 1) % len(self.profiles)
            self._refresh()

        def action_prev_profile(self):
            self.selected = (self.selected - 1) % len(self.profiles)
            self._refresh()

        def action_toggle_source(self):
            name = self.profiles[self.selected]
            entry = self.cfg[name]
            if entry.get("source") == "custom":
                self.cfg[name] = {"source": "inherit"}
            else:
                try:
                    from bibilab.config import load_config
                    backend = load_config()
                    self.cfg[name] = {
                        "source": "custom",
                        "protocol": backend.ai.protocol,
                        "model": backend.ai.model,
                        "api_key": backend.ai.api_key,
                        "base_url": backend.ai.base_url,
                    }
                except Exception:
                    self.cfg[name] = {
                        "source": "custom",
                        "protocol": "openai",
                        "model": "",
                        "api_key": "",
                        "base_url": "",
                    }
            self._refresh()

        def action_save_config(self):
            save_eval_config(self.cfg)
            self.notify("Config saved!", timeout=2)

        def action_quit_app(self):
            self.exit()

    app = ConfigApp()
    app.run()
