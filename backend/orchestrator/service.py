from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

from ..pid_control import PIDConfig, PumpState, TargetParams, VisionMetrics
from ..pid_control.service import build_controller, reset_controller, run_feedback_step
from ..pump_hardware import ChannelParams, PumpHardwareService
from .config import OrchestratorConfig
from .models import (
    ControlSnapshot,
    PumpRuntimeState,
    RecognitionSnapshot,
    SystemConfig,
    SystemSnapshot,
)
from .state import SystemState
from .vision_adapter import GenericVisionAdapter, VisionAdapterProtocol


class OrchestratorService:
    def __init__(
        self,
        vision_service: Any = None,
        vision_adapter: VisionAdapterProtocol | None = None,
        pump_service: PumpHardwareService | None = None,
        logger: Callable[[str], None] | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        pid_config: PIDConfig | None = None,
    ) -> None:
        self.vision_service = vision_service
        self.vision_adapter = vision_adapter or (
            GenericVisionAdapter(vision_service) if vision_service is not None else None
        )
        self.pump_service = pump_service or PumpHardwareService(logger=logger)
        self._log = logger or (lambda _msg: None)
        self.runtime = orchestrator_config or OrchestratorConfig()
        self.pid_config = pid_config or PIDConfig()

        self._state = SystemState.IDLE
        self._cfg: SystemConfig | None = None
        self._recognition: RecognitionSnapshot | None = None
        self._pump_state = PumpRuntimeState(
            connected=False,
            comm_established=False,
            fully_ready=False,
            q1=0.0,
            q2=0.0,
            running=False,
            last_error="",
        )
        self._control: ControlSnapshot | None = None
        self._message = ""
        self._error = ""

        self._lock = threading.RLock()
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._last_control_ts: float | None = None

    def set_vision_adapter(self, adapter: VisionAdapterProtocol | None) -> None:
        with self._lock:
            self.vision_adapter = adapter

    def _is_realtime_mode(self) -> bool:
        if self._cfg is None:
            return False
        mode = str(self._cfg.video_source_type or "").strip().lower()
        return mode in {"camera", "realtime", "real_time", "live", "rtsp", "usb"}

    def _set_state(self, state: SystemState, message: str = "", error: str = "") -> None:
        with self._lock:
            self._state = state
            if message:
                self._message = message
            if error:
                self._error = error
        if message:
            self._log(f"[ORCH][{state.value}] {message}")
        if error:
            self._log(f"[ORCH][ERROR] {error}")

    def configure(self, system_config: SystemConfig) -> None:
        interval = int(system_config.control_interval_ms)
        interval = max(self.runtime.min_control_interval_ms, interval)
        interval = min(self.runtime.max_control_interval_ms, interval)
        system_config.control_interval_ms = interval
        with self._lock:
            self._cfg = system_config
            self._error = ""
            self._message = "配置完成"
        self._set_state(SystemState.CONFIGURED, message="已接收系统配置")

    def prepare_video(self) -> None:
        with self._lock:
            cfg = self._cfg
            adapter = self.vision_adapter
        if cfg is None:
            raise RuntimeError("未配置系统参数，请先 configure()")
        try:
            if adapter is not None:
                adapter.prepare_video(
                    video_source_type=cfg.video_source_type,
                    video_source=cfg.video_source,
                    pixel_to_micron=cfg.pixel_to_micron,
                )
        except Exception as e:
            # 允许视觉侧延迟接入，先进入 VIDEO_READY 供联调
            self._log(f"[ORCH][WARN] prepare_video 调用失败，进入兼容模式: {e}")
        self._set_state(SystemState.VIDEO_READY, message="视频输入已准备")

    def initialize_system(self) -> None:
        with self._lock:
            cfg = self._cfg
            state = self._state
        if cfg is None:
            raise RuntimeError("未配置系统参数，请先 configure()")
        if state not in {SystemState.VIDEO_READY, SystemState.CONFIGURED, SystemState.STOPPED}:
            raise RuntimeError(f"当前状态不允许初始化: {state.value}")

        self._set_state(SystemState.INITIALIZING, message="系统初始化中")
        try:
            if self._is_realtime_mode():
                probe = self.pump_service.connect_and_probe()
                self._pump_state.connected = bool(probe.serial_connected)
                self._pump_state.comm_established = bool(probe.comm_established)
                self._pump_state.fully_ready = bool(probe.fully_ready)
                if not probe.comm_established:
                    raise RuntimeError(f"泵通信未建立: {probe.failed}")

                init_apply = self._apply_flow_rates(
                    cfg.initial_q1,
                    cfg.initial_q2,
                    reason_prefix="初始化流速",
                )
                if not init_apply.ok:
                    raise RuntimeError(f"初始化流速下发失败: {init_apply.reason or init_apply.error}")
            else:
                self._message = "本地视频模式: 跳过泵初始化与 PID 执行"

            build_controller(self.pid_config)
            reset_controller()
            self._last_control_ts = None
            self._set_state(SystemState.INITIALIZED, message="系统初始化完成")
        except Exception as e:
            self._pump_state.last_error = str(e)
            self._set_state(SystemState.ERROR, error=f"初始化失败: {e}")
            raise

    def start(self) -> None:
        with self._lock:
            if self._state not in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}:
                raise RuntimeError(f"当前状态不允许启动: {self._state.value}")
            if self._loop_thread and self._loop_thread.is_alive():
                raise RuntimeError("控制循环已在运行")
            adapter = self.vision_adapter

        try:
            if adapter is not None:
                adapter.start()
        except Exception as e:
            self._log(f"[ORCH][WARN] 视觉启动失败，继续控制流程: {e}")

        self._stop_event.clear()
        self._pause_event.clear()
        self._loop_thread = threading.Thread(
            target=self._control_loop,
            name="orchestrator-control-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._set_state(SystemState.RUNNING, message="系统运行中")

    def pause(self) -> None:
        with self._lock:
            if self._state != SystemState.RUNNING:
                return
        self._pause_event.set()
        self._set_state(SystemState.PAUSED, message="控制循环已暂停")

    def resume(self) -> None:
        with self._lock:
            if self._state != SystemState.PAUSED:
                return
        self._pause_event.clear()
        self._set_state(SystemState.RUNNING, message="控制循环已恢复")

    def stop(self) -> None:
        self._set_state(SystemState.STOPPING, message="系统停止中")
        self._stop_event.set()

        t = self._loop_thread
        if t and t.is_alive():
            t.join(timeout=float(self.runtime.stop_timeout_s))

        if self._is_realtime_mode():
            try:
                self.pump_service.stop_system_and_verify()
                self._pump_state.running = False
            except Exception as e:
                self._pump_state.last_error = str(e)
                self._log(f"[ORCH][WARN] 停泵校验失败: {e}")

        try:
            adapter = self.vision_adapter
            if adapter is not None:
                adapter.stop()
        except Exception as e:
            self._log(f"[ORCH][WARN] 视觉停止失败: {e}")

        self._set_state(SystemState.STOPPED, message="系统已停止")

    def get_snapshot(self) -> SystemSnapshot:
        with self._lock:
            return copy.deepcopy(
                SystemSnapshot(
                    system_state=self._state,
                    config=self._cfg,
                    recognition=self._recognition,
                    pump_state=self._pump_state,
                    control=self._control,
                    message=self._message,
                    error=self._error,
                )
            )

    def _build_recognition_snapshot(self, raw: Any) -> RecognitionSnapshot:
        if isinstance(raw, RecognitionSnapshot):
            return raw
        if isinstance(raw, dict):
            return RecognitionSnapshot(
                avg_diameter=float(raw.get("avg_diameter", 0.0)),
                droplet_count=int(raw.get("droplet_count", 0)),
                single_cell_rate=float(raw.get("single_cell_rate", 0.0)),
                valid_for_control=bool(raw.get("valid_for_control", False)),
                timestamp=float(raw.get("timestamp", time.time())),
            )
        raise ValueError(f"无法解析识别结果类型: {type(raw)!r}")

    def _read_recognition(self) -> RecognitionSnapshot:
        adapter = self.vision_adapter
        if adapter is None:
            raise RuntimeError("未注入 vision_adapter/vision_service")
        raw = adapter.get_snapshot()
        snap = self._build_recognition_snapshot(raw)
        with self._lock:
            self._recognition = snap
        return snap

    def _update_control_snapshot(self, ctrl: ControlSnapshot) -> None:
        with self._lock:
            self._control = ctrl

    def _pump_running_sync(self) -> None:
        try:
            rs = self.pump_service.read_rse()
            if rs.ok and rs.parsed_reply is not None:
                run_state = rs.parsed_reply
                self._pump_state.running = bool(run_state.system_running)
        except Exception as e:
            self._pump_state.last_error = str(e)

    def _default_channel_params(self, channel: int, q: float) -> ChannelParams:
        dispense_value = 1000
        safe_q = max(float(q), 1e-6)
        infuse_time_value = max(1, min(65535, int(round(dispense_value / safe_q))))
        return ChannelParams(
            channel=channel,
            mode=1,
            syringe_code=0x21,
            dispense_value=dispense_value,
            dispense_unit=4,
            infuse_time_value=infuse_time_value,
            infuse_time_unit=2,
            withdraw_time_value=1,
            withdraw_time_unit=2,
            repeat_count=1,
            interval_value=0,
        )

    def _to_channel_params_with_flow(self, channel: int, q: float) -> ChannelParams:
        rsp = self.pump_service.read_rsp(channel)
        if rsp.ok and rsp.parsed_reply is not None:
            p: ChannelParams = rsp.parsed_reply
            dispense_value = max(1, int(p.dispense_value))
            safe_q = max(float(q), 1e-6)
            infuse_time_value = max(1, min(65535, int(round(dispense_value / safe_q))))
            return ChannelParams(
                channel=channel,
                mode=max(0, int(p.mode)),
                syringe_code=max(0, int(p.syringe_code)),
                dispense_value=dispense_value,
                dispense_unit=int(p.dispense_unit),
                infuse_time_value=infuse_time_value,
                infuse_time_unit=int(p.infuse_time_unit),
                withdraw_time_value=max(1, int(p.withdraw_time_value)),
                withdraw_time_unit=int(p.withdraw_time_unit),
                repeat_count=max(1, int(p.repeat_count)),
                interval_value=max(0, int(p.interval_value)),
            )
        return self._default_channel_params(channel=channel, q=q)

    def _apply_flow_rates(self, q1: float, q2: float, reason_prefix: str) -> Any:
        if q1 <= 0 or q2 <= 0:
            raise RuntimeError(f"{reason_prefix} 非法流速: q1={q1}, q2={q2}")

        if hasattr(self.pump_service, "set_logical_flows_and_verify"):
            return self.pump_service.set_logical_flows_and_verify(q1=q1, q2=q2)

        p1 = self._to_channel_params_with_flow(channel=1, q=q1)
        p2 = self._to_channel_params_with_flow(channel=2, q=q2)
        w1 = self.pump_service.write_wsp_and_verify(1, p1)
        if not w1.ok:
            raise RuntimeError(f"{reason_prefix} CH1 下发失败: {w1.reason or w1.error}")
        w2 = self.pump_service.write_wsp_and_verify(2, p2)
        if not w2.ok:
            raise RuntimeError(f"{reason_prefix} CH2 下发失败: {w2.reason or w2.error}")

        self._pump_state.q1 = float(q1)
        self._pump_state.q2 = float(q2)
        return w2

    def _handle_pid_step(self, recognition: RecognitionSnapshot, dt: float) -> None:
        assert self._cfg is not None
        vision_in = VisionMetrics(
            avg_diameter=float(recognition.avg_diameter),
            droplet_count=int(recognition.droplet_count),
            valid_for_control=bool(recognition.valid_for_control),
        )
        target_in = TargetParams(target_diameter=float(self._cfg.target_diameter))
        pump_in = PumpState(q1=float(self._pump_state.q1), q2=float(self._pump_state.q2))
        cmd = run_feedback_step(
            vision_metrics=vision_in,
            target_params=target_in,
            pump_state=pump_in,
            dt=float(dt),
        )
        now = time.time()
        ctrl = ControlSnapshot(
            diameter_error=float(cmd.diameter_error),
            adjustment=float(cmd.adjustment),
            q1_command=float(cmd.q1),
            q2_command=float(cmd.q2),
            freeze_feedback=bool(cmd.freeze_feedback),
            suggested_stop=bool(cmd.suggested_stop),
            reason=str(cmd.reason),
            timestamp=now,
        )
        self._update_control_snapshot(ctrl)

        if cmd.freeze_feedback:
            return
        if cmd.suggested_stop:
            self.pump_service.stop_system_and_verify()
            self._pump_state.running = False
            raise RuntimeError(f"PID 建议停机: {cmd.reason}")

        self._apply_flow_rates(float(cmd.q1), float(cmd.q2), reason_prefix="PID下发")
        self._pump_state.q1 = float(cmd.q1)
        self._pump_state.q2 = float(cmd.q2)
        self._pump_running_sync()

    def _control_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                cfg = self._cfg
                state = self._state
            if cfg is None:
                time.sleep(0.1)
                continue
            if state == SystemState.PAUSED or self._pause_event.is_set():
                time.sleep(0.05)
                continue

            tick_start = time.monotonic()
            try:
                recognition = self._read_recognition()
                if self._is_realtime_mode():
                    now = time.monotonic()
                    if self._last_control_ts is None:
                        dt = max(0.001, float(cfg.control_interval_ms) / 1000.0)
                    else:
                        dt = max(0.001, now - self._last_control_ts)
                    self._last_control_ts = now
                    self._handle_pid_step(recognition=recognition, dt=dt)
                else:
                    self._update_control_snapshot(
                        ControlSnapshot(
                            diameter_error=0.0,
                            adjustment=0.0,
                            q1_command=self._pump_state.q1,
                            q2_command=self._pump_state.q2,
                            freeze_feedback=True,
                            suggested_stop=False,
                            reason="本地视频模式: 仅识别，不执行 PID 与泵下发",
                            timestamp=time.time(),
                        )
                    )
            except Exception as e:
                self._error = str(e)
                self._pump_state.last_error = str(e)
                self._set_state(SystemState.ERROR, error=f"控制循环异常: {e}")
                break

            elapsed = time.monotonic() - tick_start
            interval_s = float(cfg.control_interval_ms) / 1000.0
            sleep_s = max(0.0, interval_s - elapsed)
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

