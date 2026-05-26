from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


@dataclass
class _Task:
    tid: str
    label: str
    state: str = "queued"      # queued | running | done | error
    status: str = ""           # sub-status (e.g. "retrieving \"X\"")
    started_at: float | None = None
    ended_at: float | None = None
    error: str = ""


class TaskDashboard:
    """Live-updating Rich dashboard for batch async work.

    Use as a context manager. Call start/update/done from any thread.
    """

    def __init__(
        self,
        title: str,
        task_ids: Iterable[tuple[str, str]],
        banner: str | None = None,
        refresh_per_second: float = 8.0,
    ):
        self.title = title
        self.banner = banner
        self._tasks: dict[str, _Task] = {
            tid: _Task(tid=tid, label=label) for tid, label in task_ids
        }
        self._order = list(self._tasks.keys())
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._console = Console(stderr=False)
        self._spinner = Spinner("dots", style="yellow")
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )

    def __rich__(self):
        with self._lock:
            return self._render()

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._live.refresh()
        self._live.__exit__(exc_type, exc, tb)

    # ---- thread-safe mutators ---------------------------------------------

    def start(self, tid: str, status: str = "starting"):
        with self._lock:
            t = self._tasks.get(tid)
            if t is None:
                return
            t.state = "running"
            t.status = status
            t.started_at = time.monotonic()

    def update(self, tid: str, status: str):
        with self._lock:
            t = self._tasks.get(tid)
            if t is None or t.state != "running":
                return
            t.status = status

    def done(self, tid: str, ok: bool, status: str = "", error: str = ""):
        with self._lock:
            t = self._tasks.get(tid)
            if t is None:
                return
            t.state = "done" if ok else "error"
            t.status = status or ("ok" if ok else "failed")
            t.error = error
            t.ended_at = time.monotonic()

    def set_banner(self, banner: str | None):
        with self._lock:
            self.banner = banner

    # ---- rendering --------------------------------------------------------

    def _render(self) -> Group:
        counts = {"queued": 0, "running": 0, "done": 0, "error": 0}
        for t in self._tasks.values():
            counts[t.state] = counts.get(t.state, 0) + 1
        elapsed = int(time.monotonic() - self._started_at)

        total = len(self._tasks)
        header = Text(
            f"{self.title} — {total} tasks · "
            f"✓ {counts['done']}  ✗ {counts['error']}  "
            f"⠋ {counts['running']}  ○ {counts['queued']}      "
            f"{elapsed}s elapsed",
            style="bold",
        )

        table = Table.grid(padding=(0, 1), expand=False)
        table.add_column(justify="left", width=2)
        table.add_column(justify="left", overflow="fold")
        table.add_column(justify="left", overflow="fold")
        table.add_column(justify="right", width=8)

        for tid in self._order:
            t = self._tasks[tid]
            if t.state == "queued":
                icon = Text("○", style="dim")
                dur = "-"
            elif t.state == "running":
                icon = self._spinner
                dur = f"{int(time.monotonic() - (t.started_at or time.monotonic()))}s"
            elif t.state == "done":
                icon = Text("✓", style="green")
                dur = f"{(t.ended_at or 0) - (t.started_at or 0):.1f}s"
            else:
                icon = Text("✗", style="red")
                dur = f"{(t.ended_at or 0) - (t.started_at or 0):.1f}s"

            label = Text(t.label, style="cyan" if t.state == "running" else "")
            status_text = t.status if not t.error else f"{t.status}: {t.error[:60]}"
            status = Text(status_text, style="red" if t.state == "error" else "dim")

            table.add_row(icon, label, status, Text(dur, style="dim"))

        parts = [header]
        if self.banner:
            parts.append(Text(self.banner, style="dim italic"))
        parts.append(table)
        return Group(*parts)
