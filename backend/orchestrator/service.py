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
from .vision_adapter import GenericVisionAdapter, PipelineVisionService, VisionAdapterProtocol


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
        self._log = logger or (lambda _msg: None)

        if vision_adapter is not None:
            self.vision_service = vision_service
            self.vision_adapter = vision_adapter
        elif vision_service is not None:
            self.vision_service = vision_service
            self.vision_adapter = GenericVisionAdapter(vision_service)
        else:
            self.vision_service = PipelineVisionService(logger=self._log)
            self.vision_adapter = GenericVisionAdapter(self.vision_service)

        self.pump_service = pump_service or PumpHardwareService(logger=logger)
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
            last_update_ok=False,
            last_update_reason="",
        )
        self._control: ControlSnapshot | None = None
        self._message = ""
        self._error = ""

        self._lock = threading.RLock()
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._last_control_ts: float | None = None
        self._pump_control_enabled = False

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

        if not getattr(system_config, "pump_port", ""):
            system_config.pump_port = "COM3"
        if not getattr(system_config, "pump_address", None):
            system_config.pump_address = 1
        if not getattr(system_config, "pump_baudrate", None):
            system_config.pump_baudrate = 1200
        if not getattr(system_config, "pump_parity", ""):
            system_config.pump_parity = "E"

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

        if adapter is not None:
            adapter.prepare_video(
                video_source_type=cfg.video_source_type,
                video_source=cfg.video_source,
                pixel_to_micron=cfg.pixel_to_micron,
            )
        self._set_state(SystemState.VIDEO_READY, message="视频输入已准备")

    def _apply_pump_serial_config(self, cfg: SystemConfig) -> None:
        serial_cfg = self.pump_service.serial_config
        serial_cfg.port = str(cfg.pump_port or "COM3").strip()
        serial_cfg.address = int(cfg.pump_address)
        serial_cfg.baudrate = int(cfg.pump_baudrate)
        serial_cfg.parity = str(cfg.pump_parity or "E").strip().upper()
        if serial_cfg.parity not in {"E", "N"}:
            serial_cfg.parity = "E"

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
                withdraw_time_value=max(0, int(p.withdraw_time_value)),
                withdraw_time_unit=max(0, int(p.withdraw_time_unit)),
                repeat_count=max(0, int(p.repeat_count)),
                interval_value=max(0, int(p.interval_value)),
            )
        return self._default_channel_params(channel, q)

    def _apply_init_flow_rates(self, q1: float, q2: float):
        retries = 3
        last_res = None
        for attempt in range(1, retries + 1):
            en = self.pump_service.enable_channels_and_verify(0x03)
            if not en.ok:
                en2 = self.pump_service.enable_channels(0x03)
                if en2.ok:
                    rss = self.pump_service.read_rss()
                    if rss.ok and rss.parsed_reply is not None:
                        setup = rss.parsed_reply
                        effective = (int(setup.enable_mask) | int(setup.copy_mask)) & 0x0F
                        if (effective & 0x03) == 0x03:
                            en = en2
                if not en.ok:
                    en.reason = en.reason or f"初始化流速: 使能 CH1/CH2 失败(第{attempt}次)"
                    last_res = en
                    time.sleep(0.12)
                    continue

            p1 = self._to_channel_params_with_flow(1, q1)
            w1 = self.pump_service.write_wsp_and_verify(1, p1)
            if not w1.ok:
                w1.reason = w1.reason or f"初始化流速: CH1 下发失败(第{attempt}次)"
                last_res = w1
                time.sleep(0.12)
                continue

            p2 = self._to_channel_params_with_flow(2, q2)
            w2 = self.pump_service.write_wsp_and_verify(2, p2)
            if not w2.ok:
                w2.reason = w2.reason or f"初始化流速: CH2 下发失败(第{attempt}次)"
                last_res = w2
                time.sleep(0.12)
                continue

            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
            return w2

        return last_res

    def _try_resume_infusion(self, source: str) -> tuple[bool, str]:
        self._log(f"[PUMP][RECOVER] {source}: 尝试恢复灌注")
        start_res = self.pump_service.start_infusion_and_verify([1, 2])
        if start_res.ok:
            self._pump_state.running = True
            self._pump_state.last_error = ""
            self._log("[PUMP][RECOVER][OK] 灌注恢复成功")
            return True, "灌注恢复成功"
        reason = start_res.reason or start_res.error or "灌注恢复失败"
        self._pump_state.running = False
        self._pump_state.last_error = str(reason)
        self._log(f"[PUMP][RECOVER][FAIL] {reason}")
        return False, str(reason)

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
            self._pump_control_enabled = False
            if self._is_realtime_mode():
                self._apply_pump_serial_config(cfg)
                probe = self.pump_service.connect_and_probe()
                self._pump_state.connected = bool(probe.serial_connected)
                self._pump_state.comm_established = bool(probe.comm_established)
                self._pump_state.fully_ready = bool(probe.fully_ready)
                if not probe.comm_established:
                    raise RuntimeError(f"泵通信未建立: {probe.failed}")

                init_apply = self._apply_init_flow_rates(cfg.initial_q1, cfg.initial_q2)
                if not init_apply.ok:
                    raise RuntimeError(f"初始化流速下发失败: {init_apply.reason or init_apply.error}")
                self._pump_control_enabled = True
                self._pump_state.last_error = ""
            else:
                self._message = "本地视频模式：跳过泵初始化与 PID 下发"

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

        if adapter is not None:
            adapter.start()

        if self._is_realtime_mode():
            if not self._pump_control_enabled:
                raise RuntimeError("泵参数未完成初始化下发与校验，不能开始 PID 反馈")
            if not self._pump_state.connected or not self._pump_state.comm_established:
                raise RuntimeError("泵硬件未连接或通信未建立，不能开始 PID 反馈")
            start_res = self.pump_service.start_infusion_and_verify([1, 2])
            if not start_res.ok:
                reason = start_res.reason or start_res.error or "泵启动灌注失败"
                self._pump_state.last_error = str(reason)
                self._set_state(SystemState.INITIALIZED, message="启动失败，保持已初始化状态", error=str(reason))
                raise RuntimeError(f"启动失败: {reason}")
            self._pump_state.running = True

        self._stop_event.clear()
        self._pause_event.clear()
        self._loop_thread = threading.Thread(target=self._control_loop, name="orchestrator-control-loop", daemon=True)
        self._loop_thread.start()
        self._log("[PID][START] PID反馈开始")
        self._set_state(SystemState.RUNNING, message="系统运行中")

    def pause(self) -> None:
        with self._lock:
            if self._state != SystemState.RUNNING:
                return
        if self._is_realtime_mode():
            stop_res = self.pump_service.stop_system_and_verify()
            if not stop_res.ok:
                reason = stop_res.reason or stop_res.error or "暂停时停泵失败"
                self._pump_state.last_error = str(reason)
                raise RuntimeError(f"暂停失败: {reason}")
            self._pump_state.running = False
        self._pause_event.set()
        self._set_state(SystemState.PAUSED, message="控制循环已暂停")

    def resume(self) -> None:
        with self._lock:
            if self._state != SystemState.PAUSED:
                return
        if self._is_realtime_mode():
            start_res = self.pump_service.start_infusion_and_verify([1, 2])
            if not start_res.ok:
                reason = start_res.reason or start_res.error or "继续时启泵失败"
                self._pump_state.last_error = str(reason)
                raise RuntimeError(f"继续失败: {reason}")
            self._pump_state.running = True
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
            frame_cnt = int(raw.get("frame_droplet_count", raw.get("active_droplet_count", 0)) or 0)
            total_cnt = int(raw.get("total_droplet_count", raw.get("droplet_count", 0)) or 0)
            new_cnt = int(raw.get("new_crossing_count", 0) or 0)
            has_droplet = bool(raw.get("has_droplet", frame_cnt > 0))
            avg_raw = raw.get("avg_diameter", None)
            avg_diameter = None if avg_raw is None else float(avg_raw)
            reason = str(raw.get("reason", raw.get("control_reason", "")) or "")
            return RecognitionSnapshot(
                frame_droplet_count=frame_cnt,
                total_droplet_count=total_cnt,
                new_crossing_count=new_cnt,
                avg_diameter=avg_diameter,
                single_cell_rate=float(raw.get("single_cell_rate", 0.0)),
                valid_for_control=bool(raw.get("valid_for_control", False)),
                timestamp=float(raw.get("timestamp", time.time())),
                reason=reason,
                droplet_count=total_cnt,
                active_droplet_count=frame_cnt,
                has_droplet=has_droplet,
                control_reason=reason,
                frame_png_base64=raw.get("frame_png_base64"),
                frame_width=int(raw.get("frame_width", 0) or 0),
                frame_height=int(raw.get("frame_height", 0) or 0),
                video_source_type=str(raw.get("video_source_type", "") or ""),
                video_source=str(raw.get("video_source", "") or ""),
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

    def _control_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            try:
                self.run_control_step()
            except Exception as e:
                self._pump_state.last_error = str(e)
                self._set_state(SystemState.ERROR, error=f"控制循环异常: {e}")
                self._stop_event.set()
                break

            interval_s = max(
                0.01,
                (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms) / 1000.0,
            )
            time.sleep(interval_s)

    def run_control_step(self) -> None:
        rec = self._read_recognition()
        now = time.time()
        if self._last_control_ts is None:
            dt = (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms) / 1000.0
        else:
            dt = max(1e-3, now - self._last_control_ts)
        self._last_control_ts = now

        if not self._is_realtime_mode():
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="本地视频模式：仅识别显示，不执行 PID 下发",
                timestamp=now,
            )
            self._log("[PID][FREEZE] 本地视频模式，不执行 PID")
            self._update_control_snapshot(ctrl)
            return

        if not self._pump_control_enabled:
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="泵未完成初始化，PID 未执行",
                timestamp=now,
            )
            self._log("[PID][FREEZE] 泵未完成初始化，PID 未执行")
            self._update_control_snapshot(ctrl)
            return

        run_state_res = self.pump_service.read_run_state()
        if (not run_state_res.ok) or (run_state_res.parsed_reply is None):
            reason = run_state_res.error or run_state_res.reason or "泵运行状态读取失败"
            resumed, resume_reason = self._try_resume_infusion(f"运行态读取失败: {reason}")
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=(
                    f"泵未灌注，PID 暂停: {reason}；已自动恢复灌注"
                    if resumed
                    else f"泵未灌注，PID 未执行: {reason}；恢复失败: {resume_reason}"
                ),
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        running_ok, running_reason = self.pump_service.are_required_channels_running([1, 2], run_state_res.parsed_reply)
        if not running_ok:
            resumed, resume_reason = self._try_resume_infusion(f"运行态校验失败: {running_reason}")
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=(
                    f"泵未灌注，PID 暂停: {running_reason}；已自动恢复灌注"
                    if resumed
                    else f"泵未灌注，PID 未执行: {running_reason}；恢复失败: {resume_reason}"
                ),
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return
        self._pump_state.running = True

        if not rec.valid_for_control:
            reason = rec.reason or rec.control_reason or "识别结果无效"
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=f"{reason}（保持当前灌注，不下发新参数）",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {reason}")
            self._update_control_snapshot(ctrl)
            return

        if self._cfg is None or rec.avg_diameter is None:
            raise RuntimeError("缺少 PID 计算所需参数")

        try:
            q1_now, q2_now = self.pump_service.get_current_q_state()
            self._pump_state.q1 = float(q1_now)
            self._pump_state.q2 = float(q2_now)
        except Exception as e:
            self._log(f"[ORCH][WARN] 读取当前Q状态失败，使用缓存值: {e}")

        vm = VisionMetrics(
            avg_diameter=float(rec.avg_diameter),
            droplet_count=int(rec.frame_droplet_count),
            valid_for_control=bool(rec.valid_for_control),
        )
        tp = TargetParams(target_diameter=float(self._cfg.target_diameter))
        ps = PumpState(q1=float(self._pump_state.q1), q2=float(self._pump_state.q2))

        cmd = run_feedback_step(vm, tp, ps, dt)
        ctrl = ControlSnapshot(
            diameter_error=float(cmd.diameter_error),
            adjustment=float(cmd.adjustment),
            q1_command=float(cmd.q1),
            q2_command=float(cmd.q2),
            freeze_feedback=bool(cmd.freeze_feedback),
            suggested_stop=bool(cmd.suggested_stop),
            reason=str(cmd.reason or ""),
            timestamp=now,
        )

        if cmd.freeze_feedback:
            if not ctrl.reason:
                ctrl.reason = "PID 冻结（保持当前灌注，不下发新参数）"
            elif "保持当前灌注" not in ctrl.reason:
                ctrl.reason = f"{ctrl.reason}（保持当前灌注，不下发新参数）"
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        if cmd.suggested_stop:
            self.pump_service.stop_system_and_verify()
            self._pump_state.running = False
            self._update_control_snapshot(ctrl)
            self._set_state(SystemState.ERROR, error=ctrl.reason or "PID 建议停机")
            return

        self._log(f"[PID][UPDATE] q1={cmd.q1:.6f}, q2={cmd.q2:.6f}, adj={cmd.adjustment:.6f}")
        update_res = self.pump_service.update_flow_while_running(float(cmd.q1), float(cmd.q2))
        if (not update_res.ok) and bool(update_res.still_running):
            # 参数下发失败但灌注仍在时先重试一次，避免控制循环被短暂通信波动打断。
            self._log("[PUMP][UPDATE][RETRY] 首次下发失败但灌注持续，重试一次")
            update_res = self.pump_service.update_flow_while_running(float(cmd.q1), float(cmd.q2))

        if not update_res.ok:
            self._pump_state.last_update_ok = False
            self._pump_state.last_update_reason = update_res.reason or "运行中参数更新失败"
            self._pump_state.last_error = self._pump_state.last_update_reason
            ctrl.reason = self._pump_state.last_update_reason
            if not update_res.still_running:
                resumed, resume_reason = self._try_resume_infusion("参数更新后灌注状态异常")
                if resumed:
                    ctrl.freeze_feedback = True
                    ctrl.reason = f"{ctrl.reason}；已自动恢复灌注，本周期不下发"
                    self._log(f"[PID][FREEZE] {ctrl.reason}")
                else:
                    self._set_state(SystemState.ERROR, error=f"参数更新后灌注状态异常: {ctrl.reason}；恢复失败: {resume_reason}")
        else:
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "参数下发成功"
            self._pump_state.last_error = ""
            self._pump_state.q1 = float(cmd.q1)
            self._pump_state.q2 = float(cmd.q2)

        self._update_control_snapshot(ctrl)
