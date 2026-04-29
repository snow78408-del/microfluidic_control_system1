from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .state import SystemState


@dataclass(slots=True)
class FlowContext:
    """流程上下文骨架（只存放流程阶段所需的最小信息）。"""

    config: Any | None = None
    video_source_type: str = ""
    video_source: str = ""
    initialized: bool = False
    runtime_flags: dict[str, Any] = field(default_factory=dict)


class SystemFlow:
    """
    系统流程骨架。

    说明：
    - 仅定义流程状态与阶段方法，不实现图像识别算法、PID 计算细节、泵通信协议细节。
    - 具体业务由 orchestrator/service.py 中的模块调用实现。
    """

    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        self.current_state: SystemState = SystemState.IDLE
        self.ctx = FlowContext()
        self._log = logger or (lambda _msg: None)

    def _set_state(self, state: SystemState, reason: str = "") -> None:
        self.current_state = state
        if reason:
            self._log(f"[FLOW][{state.value}] {reason}")

    def configure(self, config: Any) -> None:
        """接收前端配置并进入 CONFIGURED。"""
        # TODO: 校验必要字段（目标直径、像素系数、视频来源、初始流速等）
        self.ctx.config = config
        self._set_state(SystemState.CONFIGURED, "基础参数已提交")

    def prepare_video(self, video_source_type: str, video_source: str) -> None:
        """准备视频来源并进入 VIDEO_READY。"""
        # TODO: 调用视觉模块准备输入（摄像头/本地视频）
        self.ctx.video_source_type = video_source_type
        self.ctx.video_source = video_source
        self._set_state(SystemState.VIDEO_READY, "视频来源已准备")

    def initialize(self) -> None:
        """
        初始化流程骨架：
        1) 实时模式：连接泵 -> 探测/回读 -> 下发初始流速 -> 校验
        2) 本地视频模式：跳过泵初始化，仅准备识别流程
        """
        self._set_state(SystemState.INITIALIZING, "初始化开始")

        # TODO: 根据模式分流
        # - realtime: 调用 pump_hardware.connect_and_probe / 初始流速下发与回读校验
        # - file: 仅保留视觉处理准备

        self.ctx.initialized = True
        self._set_state(SystemState.INITIALIZED, "初始化完成")

    def start(self) -> None:
        """进入 RUNNING：启动识别循环与控制循环（实时模式）。"""
        # TODO: 启动视觉采集/识别循环
        # TODO: 实时模式下启动 PID 控制周期
        self._set_state(SystemState.RUNNING, "系统开始运行")

    def pause(self) -> None:
        """进入 PAUSED：暂停控制循环。"""
        # TODO: 暂停控制定时器/线程（识别可按策略继续）
        self._set_state(SystemState.PAUSED, "系统已暂停")

    def resume(self) -> None:
        """从 PAUSED 恢复到 RUNNING。"""
        # TODO: 恢复控制循环
        self._set_state(SystemState.RUNNING, "系统继续运行")

    def stop(self) -> None:
        """
        停止流程骨架：
        1) 停止控制循环
        2) 停泵（实时模式）
        3) 停止识别流程
        4) 进入 STOPPED
        """
        self._set_state(SystemState.STOPPING, "系统停止中")

        # TODO: 先停控制循环
        # TODO: 再停泵（实时模式）
        # TODO: 最后停止视觉识别

        self._set_state(SystemState.STOPPED, "系统已停止")

    def handle_error(self, error: Exception | str) -> None:
        """
        异常流程骨架：
        - 记录异常
        - 如有必要执行停泵保护
        - 进入 ERROR
        """
        # TODO: 调用安全停机逻辑（泵异常/建议停机）
        self.ctx.runtime_flags["last_error"] = str(error)
        self._set_state(SystemState.ERROR, f"异常: {error}")

    def run_control_step(self) -> None:
        """
        单步控制流程骨架（实时模式）：
        1) 读取识别快照
        2) 判断有效性
        3) 调用 PID
        4) 根据 freeze/suggested_stop 分支
        5) 正常时下发并校验泵参数
        """
        # TODO: recognition = vision.get_snapshot()
        # TODO: if !valid -> 保持当前流速并更新状态
        # TODO: cmd = pid.run_feedback_step(...)
        # TODO: if cmd.freeze_feedback -> 保持当前流速
        # TODO: if cmd.suggested_stop -> 停泵并转 ERROR/STOPPED
        # TODO: else -> 下发 q1/q2 并回读校验
        pass