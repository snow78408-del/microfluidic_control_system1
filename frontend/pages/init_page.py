from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk


class InitPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.q1_var = tk.StringVar(value="100")
        self.q2_var = tk.StringVar(value="100")
        self.status_var = tk.StringVar(value="未初始化")
        self._build()

    def _build(self) -> None:
        card = ttk.LabelFrame(self, text="初始化参数设置")
        card.pack(fill="x", padx=24, pady=24)

        ttk.Label(card, text="初始 Q1 流速").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.q1_var, width=24).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="初始 Q2 流速").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.q2_var, width=24).grid(row=1, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="初始化状态").grid(row=2, column=0, padx=8, pady=8, sticky="w")
        ttk.Label(card, textvariable=self.status_var).grid(row=2, column=1, padx=8, pady=8, sticky="w")

        btns = ttk.Frame(card)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=16)
        ttk.Button(btns, text="上一步", command=lambda: self.app.show_page("video_source")).pack(side="left")
        ttk.Button(btns, text="初始化系统", command=self._initialize).pack(side="right")
        ttk.Button(btns, text="进入监控页面", command=lambda: self.app.show_page("monitor")).pack(side="right", padx=6)

    def _initialize(self) -> None:
        try:
            q1 = float(self.q1_var.get().strip())
            q2 = float(self.q2_var.get().strip())
            if q1 <= 0:
                raise ValueError("初始 Q1 流速必须大于 0")
            if q2 <= 0:
                raise ValueError("初始 Q2 流速必须大于 0")
        except Exception as e:
            messagebox.showerror("输入错误", str(e))
            return

        self.app.frontend_config["initial_q1"] = q1
        self.app.frontend_config["initial_q2"] = q2
        self.status_var.set("初始化中...")

        def task():
            self.app.configure_prepare_initialize()

        def ok():
            self.status_var.set("初始化完成")
            messagebox.showinfo("初始化成功", "系统已初始化，可进入监控页面启动运行")

        def fail(err: Exception):
            self.status_var.set("初始化失败")
            messagebox.showerror("初始化失败", str(err))

        self.app.run_backend_task(task, on_success=ok, on_error=fail)

