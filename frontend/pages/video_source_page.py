from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class VideoSourcePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.mode_var = tk.StringVar(value="camera")
        self.camera_var = tk.StringVar(value="0")
        self.file_var = tk.StringVar(value="")
        self._build()

    def _build(self) -> None:
        card = ttk.LabelFrame(self, text="视频来源选择")
        card.pack(fill="x", padx=24, pady=24)

        ttk.Radiobutton(card, text="实时摄像头", value="camera", variable=self.mode_var, command=self._toggle).grid(
            row=0, column=0, padx=8, pady=8, sticky="w"
        )
        ttk.Radiobutton(card, text="本地视频文件", value="file", variable=self.mode_var, command=self._toggle).grid(
            row=0, column=1, padx=8, pady=8, sticky="w"
        )

        self.camera_label = ttk.Label(card, text="摄像头编号")
        self.camera_entry = ttk.Entry(card, textvariable=self.camera_var, width=24)
        self.camera_label.grid(row=1, column=0, padx=8, pady=8, sticky="w")
        self.camera_entry.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        self.file_label = ttk.Label(card, text="本地视频路径")
        self.file_entry = ttk.Entry(card, textvariable=self.file_var, width=56)
        self.file_btn = ttk.Button(card, text="浏览", command=self._browse_file)
        self.file_label.grid(row=2, column=0, padx=8, pady=8, sticky="w")
        self.file_entry.grid(row=2, column=1, padx=8, pady=8, sticky="w")
        self.file_btn.grid(row=2, column=2, padx=8, pady=8, sticky="w")

        ttk.Button(card, text="上一步", command=lambda: self.app.show_page("parameter")).grid(
            row=3, column=0, padx=8, pady=16, sticky="w"
        )
        ttk.Button(card, text="下一步", command=self._next_step).grid(row=3, column=2, padx=8, pady=16, sticky="e")
        self._toggle()

    def _toggle(self) -> None:
        is_camera = self.mode_var.get() == "camera"
        self.camera_entry.configure(state=("normal" if is_camera else "disabled"))
        self.file_entry.configure(state=("disabled" if is_camera else "normal"))
        self.file_btn.configure(state=("disabled" if is_camera else "normal"))

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择本地视频文件",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def _next_step(self) -> None:
        mode = self.mode_var.get()
        if mode == "camera":
            try:
                cam_idx = int(self.camera_var.get().strip())
            except Exception:
                messagebox.showerror("输入错误", "实时视频模式必须填写有效摄像头编号")
                return
            self.app.frontend_config["video_source_type"] = "camera"
            self.app.frontend_config["video_source"] = str(cam_idx)
        else:
            path = self.file_var.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showerror("输入错误", "本地视频模式必须选择有效文件路径")
                return
            self.app.frontend_config["video_source_type"] = "file"
            self.app.frontend_config["video_source"] = path

        self.app.show_page("init")

