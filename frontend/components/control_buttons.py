from __future__ import annotations

from tkinter import ttk

from ...backend.orchestrator.state import SystemState


class ControlButtons(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="运行控制")
        self.init_btn = ttk.Button(self, text="初始化系统")
        self.start_btn = ttk.Button(self, text="开始运行")
        self.pause_btn = ttk.Button(self, text="暂停")
        self.resume_btn = ttk.Button(self, text="继续")
        self.stop_btn = ttk.Button(self, text="停止")
        self._build()

    def _build(self) -> None:
        self.init_btn.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.start_btn.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.pause_btn.grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        self.resume_btn.grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        self.stop_btn.grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        for i in range(5):
            self.columnconfigure(i, weight=1)

    def bind_actions(self, on_init, on_start, on_pause, on_resume, on_stop) -> None:
        self.init_btn.configure(command=on_init)
        self.start_btn.configure(command=on_start)
        self.pause_btn.configure(command=on_pause)
        self.resume_btn.configure(command=on_resume)
        self.stop_btn.configure(command=on_stop)

    def update_by_state(self, state: SystemState) -> None:
        init_enabled = state in {
            SystemState.CONFIGURED,
            SystemState.VIDEO_READY,
            SystemState.STOPPED,
            SystemState.ERROR,
        }
        start_enabled = state in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}
        pause_enabled = state == SystemState.RUNNING
        resume_enabled = state == SystemState.PAUSED
        stop_enabled = state in {
            SystemState.RUNNING,
            SystemState.PAUSED,
            SystemState.INITIALIZED,
            SystemState.ERROR,
            SystemState.INITIALIZING,
        }

        self.init_btn.configure(state=("normal" if init_enabled else "disabled"))
        self.start_btn.configure(state=("normal" if start_enabled else "disabled"))
        self.pause_btn.configure(state=("normal" if pause_enabled else "disabled"))
        self.resume_btn.configure(state=("normal" if resume_enabled else "disabled"))
        self.stop_btn.configure(state=("normal" if stop_enabled else "disabled"))

