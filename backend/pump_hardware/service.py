from __future__ import annotations

import time
from typing import Callable

from .client import CommandMismatchError, FrameParseError, NoReplyError, PumpClient
from .config import PumpHardwareConfig, SerialConfig
from .models import (
    ChannelParams,
    PumpConnectionState,
    PumpOperationResult,
    RunState,
    SystemSetup,
)
from . import protocol


class PumpHardwareService:
    """
    可供 orchestrator / PID 直接调用的 TS 注射泵硬件服务层。
    """

    def __init__(
        self,
        serial_config: SerialConfig | None = None,
        runtime_config: PumpHardwareConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.serial_config = serial_config or SerialConfig()
        self.runtime_config = runtime_config or PumpHardwareConfig()
        self._logger = logger or (lambda _msg: None)
        self.client = PumpClient(
            serial_config=self.serial_config,
            runtime_config=self.runtime_config,
            logger=self._logger,
        )
        self.connection_state = PumpConnectionState()
        self.last_system_setup: SystemSetup | None = None
        self.last_run_state: RunState | None = None
        self.last_channel_params: dict[int, ChannelParams] = {}

    def log(self, msg: str) -> None:
        self._logger(msg)

    @staticmethod
    def _ok(parsed=None, raw: bytes | None = None, verified: bool = False, reason: str | None = None):
        return PumpOperationResult(
            ok=True,
            parsed_reply=parsed,
            raw_reply=raw,
            verified=verified,
            reason=reason,
        )

    @staticmethod
    def _fail(error: Exception | str, parsed=None, raw: bytes | None = None, reason: str | None = None):
        return PumpOperationResult(
            ok=False,
            error=str(error),
            parsed_reply=parsed,
            raw_reply=raw,
            verified=False,
            reason=reason or str(error),
        )

    def connect_and_probe(self) -> PumpConnectionState:
        state = PumpConnectionState(serial_connected=False, comm_established=False, fully_ready=False)
        try:
            self.client.connect()
            state.serial_connected = self.client.is_connected()
        except Exception as e:
            state.failed["serial"] = str(e)
            self.connection_state = state
            self.log(f"[CONNECT][FAIL] {e}")
            return state

        probe_results: list[tuple[str, PumpOperationResult]] = []
        probe_results.append(("RSS", self.read_rss()))
        time.sleep(self.runtime_config.probe_step_delay)
        probe_results.append(("RSE", self.read_rse()))
        time.sleep(self.runtime_config.probe_step_delay)
        for ch in (1, 2, 3, 4):
            probe_results.append((f"RSP{ch}", self.read_rsp(ch)))
            time.sleep(self.runtime_config.probe_step_delay)

        for key, res in probe_results:
            if res.ok:
                state.succeeded.append(key)
            else:
                state.failed[key] = res.error or "unknown"

        state.comm_established = any(k in state.succeeded for k in ("RSS", "RSE", "RSP1", "RSP2", "RSP3", "RSP4"))
        state.fully_ready = ("RSS" in state.succeeded and "RSE" in state.succeeded and all(f"RSP{c}" in state.succeeded for c in (1, 2, 3, 4)))
        self.connection_state = state
        if state.fully_ready:
            self.log("[CONNECT][OK] 串口连接、通信建立、设备完全就绪")
        elif state.comm_established:
            self.log(f"[CONNECT][OK] 通信已建立但未完全就绪: failed={state.failed}")
        else:
            self.log(f"[CONNECT][FAIL] 串口已开但通信未建立: failed={state.failed}")
        return state

    def disconnect(self) -> None:
        self.client.disconnect()
        self.connection_state = PumpConnectionState()
        self.log("[CONNECT][OK] 串口已断开")

    def _send(self, pdu: bytes, expect_cmd: str, allow_no_reply: bool = False):
        return self.client.send_pdu(
            pdu=pdu,
            expect_cmd=expect_cmd,
            allow_no_reply=allow_no_reply,
            retries=self.runtime_config.retry_count,
            timeout=self.runtime_config.reply_timeout,
            idle_timeout=self.runtime_config.idle_timeout,
            post_write_delay=self.runtime_config.post_write_delay,
            addr=self.serial_config.address,
        )

    def read_rss(self) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rss(), expect_cmd="RSS")
            setup = protocol.parse_rss_pdu(rep.pdu)
            self.last_system_setup = setup
            self.log(
                f"[RSS][OK] copy=0x{setup.copy_mask:02X}, enable=0x{setup.enable_mask:02X}, "
                f"delay={setup.delay_values}, unit={setup.delay_units}"
            )
            return self._ok(parsed=setup, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSS][FAIL] {e}")
            return self._fail(e)

    def read_rse(self) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rse(), expect_cmd="RSE")
            run_state = protocol.parse_rse_pdu(rep.pdu)
            self.last_run_state = run_state
            self.log(
                f"[RSE][OK] sys=0x{run_state.sys_runstate:02X}, q=0x{run_state.q_runstate:02X}, "
                f"running={run_state.channel_running}"
            )
            return self._ok(parsed=run_state, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSE][FAIL] {e}")
            return self._fail(e)

    def read_rsp(self, channel: int) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rsp(channel), expect_cmd="RSP")
            params = protocol.parse_rsp_pdu(rep.pdu)
            if params.channel != channel:
                raise ValueError(f"RSP 通道号不一致: expect={channel}, got={params.channel}")
            self.last_channel_params[channel] = params
            self.log(
                f"[RSP][OK][CH{channel}] mode={params.mode}, sid={params.syringe_code}, "
                f"dispense={params.dispense_value}/{params.dispense_unit}, infuse={params.infuse_time_value}/{params.infuse_time_unit}"
            )
            return self._ok(parsed=params, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSP][FAIL][CH{channel}] {e}")
            return self._fail(e)

    def write_wss(self, setup: SystemSetup) -> PumpOperationResult:
        pdu = protocol.pdu_wss(
            copy_mask=setup.copy_mask,
            enable_mask=setup.enable_mask,
            delay_values=setup.delay_values,
            delay_units=setup.delay_units,
        )
        try:
            rep = self._send(pdu, expect_cmd="WSS", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            self.log("[WSS][OK] 命令发送完成")
            return self._ok(parsed=setup, raw=raw, verified=False)
        except (NoReplyError, FrameParseError, CommandMismatchError) as e:
            self.log(f"[WSS][FAIL] {e}")
            return self._fail(e)
        except Exception as e:
            self.log(f"[WSS][FAIL] {e}")
            return self._fail(e)

    def write_wss_and_verify(self, setup: SystemSetup) -> PumpOperationResult:
        wr = self.write_wss(setup)
        if not wr.ok:
            return wr
        rd = self.read_rss()
        if not rd.ok:
            return self._fail(f"WSS 写入后 RSS 回读失败: {rd.error}")
        got: SystemSetup = rd.parsed_reply
        mismatch = []
        if got.enable_mask != setup.enable_mask:
            mismatch.append(f"enable_mask expect=0x{setup.enable_mask:02X}, got=0x{got.enable_mask:02X}")
        if got.copy_mask != setup.copy_mask:
            mismatch.append(f"copy_mask expect=0x{setup.copy_mask:02X}, got=0x{got.copy_mask:02X}")
        if got.delay_values != setup.delay_values:
            mismatch.append(f"delay_values expect={setup.delay_values}, got={got.delay_values}")
        if got.delay_units != setup.delay_units:
            mismatch.append(f"delay_units expect={setup.delay_units}, got={got.delay_units}")
        if mismatch:
            reason = "; ".join(mismatch)
            self.log(f"[WSS][FAIL] 回读不一致: {reason}")
            return self._fail("WSS 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)
        self.log("[WSS][OK] 写后读回校验通过")
        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSS 校验通过")

    def write_wsp(self, params: ChannelParams) -> PumpOperationResult:
        pdu = protocol.pdu_wsp(
            channel=params.channel,
            mode=params.mode,
            syringe_code=params.syringe_code,
            dispense_value=params.dispense_value,
            dispense_unit=params.dispense_unit,
            infuse_time_value=params.infuse_time_value,
            infuse_time_unit=params.infuse_time_unit,
            withdraw_time_value=params.withdraw_time_value,
            withdraw_time_unit=params.withdraw_time_unit,
            repeat_count=params.repeat_count,
            interval_value=params.interval_value,
        )
        try:
            rep = self._send(pdu, expect_cmd="WSP", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            self.log(f"[WSP][OK][CH{params.channel}] 命令发送完成")
            return self._ok(parsed=params, raw=raw, verified=False)
        except Exception as e:
            self.log(f"[WSP][FAIL][CH{params.channel}] {e}")
            return self._fail(e)

    def write_wsp_and_verify(self, channel: int, params: ChannelParams) -> PumpOperationResult:
        wr = self.write_wsp(params)
        if not wr.ok:
            return wr
        rd = self.read_rsp(channel)
        if not rd.ok:
            return self._fail(f"WSP 写入后 RSP 回读失败: {rd.error}")
        got: ChannelParams = rd.parsed_reply
        mismatch = []
        fields = [
            "channel",
            "mode",
            "syringe_code",
            "dispense_value",
            "dispense_unit",
            "infuse_time_value",
            "infuse_time_unit",
        ]
        for name in fields:
            if int(getattr(got, name)) != int(getattr(params, name)):
                mismatch.append(f"{name} expect={getattr(params, name)}, got={getattr(got, name)}")
        if mismatch:
            reason = "; ".join(mismatch)
            self.log(f"[VERIFY][FAIL][CH{channel}] {reason}")
            return self._fail("WSP 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)
        self.log(f"[VERIFY][OK][CH{channel}] WSP 写后读回一致")
        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSP 校验通过")

    def write_wse(self, sys_runstate: int, q_runstate: int = 0x00) -> PumpOperationResult:
        pdu = protocol.pdu_wse(sys_runstate=sys_runstate, q_runstate=q_runstate)
        try:
            rep = self._send(pdu, expect_cmd="WSE", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            self.log(f"[WSE][OK] sys=0x{sys_runstate:02X}, q=0x{q_runstate:02X}")
            return self._ok(raw=raw, parsed={"sys_runstate": sys_runstate, "q_runstate": q_runstate})
        except Exception as e:
            self.log(f"[WSE][FAIL] {e}")
            return self._fail(e)

    def write_wse_and_verify(self, sys_runstate: int, q_runstate: int = 0x00) -> PumpOperationResult:
        wr = self.write_wse(sys_runstate=sys_runstate, q_runstate=q_runstate)
        if not wr.ok:
            return wr
        rd = self.read_rse()
        if not rd.ok:
            return self._fail(f"WSE 写入后 RSE 回读失败: {rd.error}")
        got: RunState = rd.parsed_reply
        if int(got.sys_runstate) != (int(sys_runstate) & 0xFF):
            reason = f"sys_runstate expect=0x{sys_runstate:02X}, got=0x{got.sys_runstate:02X}"
            self.log(f"[WSE][FAIL] {reason}")
            return self._fail("WSE 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)
        self.log("[WSE][OK] 写后读回校验通过")
        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSE 校验通过")

    def enable_channels(self, mask: int) -> PumpOperationResult:
        rss = self.read_rss()
        if not rss.ok:
            return self._fail(f"enable_channels 前读取 RSS 失败: {rss.error}")
        setup: SystemSetup = rss.parsed_reply
        enable = int(mask) & 0x0F
        copy_mask = setup.copy_mask & 0x0F
        if enable == 0:
            copy_mask = 0
        elif (copy_mask & enable) == 0:
            copy_mask = enable & -enable
        req = SystemSetup(
            enable_mask=enable,
            copy_mask=copy_mask & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        return self.write_wss(req)

    def enable_channels_and_verify(self, mask: int) -> PumpOperationResult:
        rss = self.read_rss()
        if not rss.ok:
            return self._fail(f"enable_channels_and_verify 前读取 RSS 失败: {rss.error}")
        setup: SystemSetup = rss.parsed_reply
        enable = int(mask) & 0x0F
        copy_mask = setup.copy_mask & 0x0F
        if enable == 0:
            copy_mask = 0
        elif (copy_mask & enable) == 0:
            copy_mask = enable & -enable
        req = SystemSetup(
            enable_mask=enable,
            copy_mask=copy_mask & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        return self.write_wss_and_verify(req)

    def stop_system(self) -> PumpOperationResult:
        return self.write_wse(sys_runstate=0x00, q_runstate=0x00)

    def stop_system_and_verify(self) -> PumpOperationResult:
        return self.write_wse_and_verify(sys_runstate=0x00, q_runstate=0x00)

    def start_system(self) -> PumpOperationResult:
        rs = self.read_rss()
        if not rs.ok:
            return self._fail(f"系统启动前 RSS 读取失败: {rs.error}")
        setup: SystemSetup = rs.parsed_reply
        run_mask = (setup.enable_mask & 0x0F) << 1
        target_sys = (0x01 | run_mask) if run_mask else 0x00
        return self.write_wse(sys_runstate=target_sys, q_runstate=0x00)

    def start_system_and_verify(self) -> PumpOperationResult:
        rs = self.read_rss()
        if not rs.ok:
            return self._fail(f"系统启动前 RSS 读取失败: {rs.error}")
        setup: SystemSetup = rs.parsed_reply
        run_mask = (setup.enable_mask & 0x0F) << 1
        target_sys = (0x01 | run_mask) if run_mask else 0x00
        return self.write_wse_and_verify(sys_runstate=target_sys, q_runstate=0x00)

    def start_single_channel_safely(self, channel: int) -> PumpOperationResult:
        if not (1 <= int(channel) <= 4):
            return self._fail(f"无效通道: {channel}")
        rse_now = self.read_rse()
        if not rse_now.ok:
            return self._fail(f"单通道启动前 RSE 读取失败: {rse_now.error}")
        rss_now = self.read_rss()
        if not rss_now.ok:
            return self._fail(f"单通道启动前 RSS 读取失败: {rss_now.error}")

        run_state: RunState = rse_now.parsed_reply
        setup: SystemSetup = rss_now.parsed_reply
        current_run_mask = run_state.sys_runstate & 0x1E
        target_bit = (1 << int(channel)) & 0x1E
        expected_after_mask = (current_run_mask | target_bit) & 0x1E
        desired_enable_mask = (expected_after_mask >> 1) & 0x0F
        current_enable_mask = setup.enable_mask & 0x0F
        self.log(
            f"[START][CH{channel}] current_enable=0x{current_enable_mask:02X}, "
            f"desired_enable=0x{desired_enable_mask:02X}, current_run_mask=0x{current_run_mask:02X}"
        )

        if desired_enable_mask != current_enable_mask:
            adjust_setup = SystemSetup(
                enable_mask=desired_enable_mask,
                copy_mask=(desired_enable_mask & -desired_enable_mask) if desired_enable_mask else 0,
                delay_values=list(setup.delay_values),
                delay_units=list(setup.delay_units),
            )
            en = self.write_wss_and_verify(adjust_setup)
            if not en.ok:
                return self._fail(f"[START][FAIL][CH{channel}] 启动前使能收敛失败: {en.reason or en.error}")
            self.log(
                f"[START][CH{channel}] enable_mask 收敛成功: 0x{current_enable_mask:02X} -> 0x{desired_enable_mask:02X}"
            )

        target_sys = 0x01 | expected_after_mask
        wr = self.write_wse_and_verify(sys_runstate=target_sys, q_runstate=run_state.q_runstate)
        if not wr.ok:
            self.log(f"[START][FAIL][CH{channel}] WSE/RSE 校验失败: {wr.reason or wr.error}")
            return wr

        final_rse = self.read_rse()
        if not final_rse.ok:
            return self._fail(f"[START][FAIL][CH{channel}] 启动后 RSE 二次确认失败: {final_rse.error}")
        final_state: RunState = final_rse.parsed_reply
        final_run_mask = final_state.sys_runstate & 0x1E
        if final_run_mask != expected_after_mask:
            extra = final_run_mask & (~expected_after_mask & 0x1E)
            extra_channels = [i + 1 for i in range(4) if extra & (1 << (i + 1))]
            reason = (
                f"检测到额外通道被启动, expected_mask=0x{expected_after_mask:02X}, "
                f"actual_mask=0x{final_run_mask:02X}, extra_channels={extra_channels}"
            )
            self.log(f"[START][FAIL][CH{channel}] {reason}")
            return self._fail("[START] 检测到额外通道启动", parsed=final_state, raw=final_rse.raw_reply, reason=reason)

        self.log(f"[START][OK][CH{channel}] 单通道启动成功")
        return self._ok(parsed=final_state, raw=final_rse.raw_reply, verified=True, reason="单通道启动校验通过")

    def stop_single_channel_safely(self, channel: int) -> PumpOperationResult:
        if not (1 <= int(channel) <= 4):
            return self._fail(f"无效通道: {channel}")
        rse_now = self.read_rse()
        if not rse_now.ok:
            return self._fail(f"单通道停止前 RSE 读取失败: {rse_now.error}")
        run_state: RunState = rse_now.parsed_reply

        current_run_mask = run_state.sys_runstate & 0x1E
        target_bit = (1 << int(channel)) & 0x1E
        expected_after_mask = current_run_mask & (~target_bit & 0x1E)
        target_sys = (0x01 | expected_after_mask) if expected_after_mask else 0x00

        wr = self.write_wse_and_verify(sys_runstate=target_sys, q_runstate=run_state.q_runstate)
        if not wr.ok:
            self.log(f"[STOP][FAIL][CH{channel}] WSE/RSE 校验失败: {wr.reason or wr.error}")
            return wr

        final_rse = self.read_rse()
        if not final_rse.ok:
            return self._fail(f"[STOP][FAIL][CH{channel}] 停止后 RSE 二次确认失败: {final_rse.error}")
        final_state: RunState = final_rse.parsed_reply
        final_mask = final_state.sys_runstate & 0x1E

        expected_other = expected_after_mask
        if (final_mask & target_bit) != 0:
            reason = f"目标通道仍在运行: ch={channel}, final_mask=0x{final_mask:02X}"
            self.log(f"[STOP][FAIL][CH{channel}] {reason}")
            return self._fail("[STOP] 目标通道未停止", parsed=final_state, raw=final_rse.raw_reply, reason=reason)
        if (final_mask & (~target_bit & 0x1E)) != expected_other:
            reason = (
                f"其他通道运行位异常变化: expected_other=0x{expected_other:02X}, "
                f"actual=0x{(final_mask & (~target_bit & 0x1E)):02X}"
            )
            self.log(f"[STOP][FAIL][CH{channel}] {reason}")
            return self._fail("[STOP] 其他通道状态异常", parsed=final_state, raw=final_rse.raw_reply, reason=reason)

        self.log(f"[STOP][OK][CH{channel}] 单通道停止成功")
        return self._ok(parsed=final_state, raw=final_rse.raw_reply, verified=True, reason="单通道停止校验通过")

