from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from tkinter import ttk

from ..components.pump_panel import PumpPanel
from ..components.recognition_panel import RecognitionPanel
from ..components.status_panel import StatusPanel


class StatusPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._poll_job = None
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=16, pady=8)
        ttk.Button(top, text="监控页面", command=lambda: self.app.show_page("monitor")).pack(side="left")

        row = ttk.Frame(self)
        row.pack(fill="x", padx=16, pady=8)
        self.status_panel = StatusPanel(row)
        self.status_panel.pack(side="left", fill="both", expand=True, padx=6)
        self.recognition_panel = RecognitionPanel(row)
        self.recognition_panel.pack(side="left", fill="both", expand=True, padx=6)
        self.pump_panel = PumpPanel(row)
        self.pump_panel.pack(side="left", fill="both", expand=True, padx=6)

        raw_box = ttk.LabelFrame(self, text="SystemSnapshot 原始视图")
        raw_box.pack(fill="both", expand=True, padx=16, pady=8)
        self.text = ttk.Treeview(raw_box, columns=("value",), show="tree headings")
        self.text.heading("#0", text="字段")
        self.text.heading("value", text="值")
        self.text.column("#0", width=220, anchor="w")
        self.text.column("value", width=900, anchor="w")
        self.text.pack(fill="both", expand=True, padx=8, pady=8)

    def on_show(self) -> None:
        self._start_poll()

    def on_hide(self) -> None:
        self._stop_poll()

    def _start_poll(self) -> None:
        self._stop_poll()
        self._poll_once()

    def _stop_poll(self) -> None:
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _to_jsonable(self, obj):
        if obj is None:
            return None
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "value"):
            return getattr(obj, "value")
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return obj

    def _poll_once(self) -> None:
        snap = self.app.orchestrator.get_snapshot()
        self.status_panel.update_snapshot(snap)
        self.recognition_panel.update_snapshot(snap)
        self.pump_panel.update_snapshot(snap)

        raw = self._to_jsonable(snap)
        text = json.dumps(raw, ensure_ascii=False, indent=2)
        self.text.delete(*self.text.get_children())
        for line in text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                self.text.insert("", "end", text=k.strip(), values=(v.strip(),))
            else:
                self.text.insert("", "end", text=line.strip(), values=("",))

        self._poll_job = self.after(self.app.refresh_interval_ms, self._poll_once)
