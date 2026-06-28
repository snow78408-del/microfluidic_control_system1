from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from pathlib import Path

from ..config import CameraSystemConfig
from .manager import CameraManager


def _print(title: str, value) -> None:
    print(f"\n=== {title} ===")
    if isinstance(value, (dict, list, tuple)):
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    else:
        print(value)


def _cti_candidates(config: CameraSystemConfig) -> list[str]:
    paths: list[str] = []
    env = os.environ.get("GENICAM_GENTL64_PATH", "")
    if env:
        paths.extend(env.split(os.pathsep))
    paths.extend(list(config.gentl_producer_paths or ()))
    paths.extend(
        [
            r"C:\Program Files\Basler\pylon\Runtime\x64",
            r"C:\Program Files\Basler\pylon\Runtime\x64\ProducerU3V.cti",
            r"C:\Program Files\Basler\pylon\Runtime\x64\ProducerGEV.cti",
            r"C:\Program Files\Teledyne\Spinnaker\cti64",
            r"C:\Program Files\Allied Vision\Vimba X\cti",
            r"C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win64",
            r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
        ]
    )
    found: list[str] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        try:
            if path.is_file() and path.suffix.lower() == ".cti":
                found.append(str(path))
            elif path.is_dir():
                found.extend(str(item) for item in path.rglob("*.cti"))
        except Exception:
            print(traceback.format_exc())
    return list(dict.fromkeys(found))


def main() -> None:
    config = CameraSystemConfig()
    logs: list[str] = []

    def logger(message: str) -> None:
        logs.append(message)
        print(message)

    _print("Python version", sys.version)
    _print("Python executable", sys.executable)
    _print("Python architecture", platform.architecture()[0])
    _print("Operating system", platform.platform())
    _print("Current working directory", os.getcwd())
    _print("sys.path", sys.path)
    _print("GENICAM_GENTL64_PATH", os.environ.get("GENICAM_GENTL64_PATH", ""))
    _print("MVS_SDK_PATH", os.environ.get("MVS_SDK_PATH", ""))
    _print("CTI files", _cti_candidates(config))
    print("\n提示：扫描和测试前请关闭厂商官方相机软件及其预览窗口，避免设备被独占。")

    manager = CameraManager(config=config, logger=logger)
    try:
        result = manager.discover_all_result()
    except Exception:
        _print("Discovery traceback", traceback.format_exc())
        raise

    _print("Backend statuses", [status.to_dict() for status in result.backend_statuses])
    _print("raw_device_count", result.raw_device_count)
    _print("deduplicated_count", result.final_device_count)
    _print("raw_devices", [device.to_dict() for device in result.raw_devices])
    _print("deduplicated_devices", [device.to_dict() for device in result.devices])
    _print("errors", result.errors)
    _print("logs", logs)

    statuses = {status.backend_name: status for status in result.backend_statuses}
    hik = statuses.get("hikrobot")
    gentl = statuses.get("gentl")
    _print(
        "工业相机后端摘要",
        {
            "MVS安装路径": hik.sdk_path if hik else "",
            "MVS Python接口路径": hik.python_module_path if hik else "",
            "MVS DLL路径": hik.dll_path if hik else "",
            "Harvester导入状态": bool(gentl and gentl.python_module_loaded),
            "CTI加载列表": gentl.cti_paths if gentl else [],
            "MVS枚举设备数": hik.raw_device_count if hik else 0,
            "GenTL枚举设备数": gentl.raw_device_count if gentl else 0,
            "最终发现设备列表": [device.to_dict() for device in result.devices],
        },
    )

    if result.raw_device_count == 0:
        print("\n未枚举到设备。请根据上方后端状态区分：SDK未加载、DLL失败、CTI缺失、枚举返回0，或设备被官方软件占用。")


if __name__ == "__main__":
    main()