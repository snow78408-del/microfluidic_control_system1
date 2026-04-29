from __future__ import annotations

import os
import pathlib
import subprocess
import sys

RELAUNCH_FLAG = "MICROFLUIDIC_FRONTEND_RELAUNCHED"
REQUIRED_MODULES = ("serial", "numpy", "cv2")


def _paths() -> tuple[pathlib.Path, pathlib.Path]:
    current = pathlib.Path(__file__).resolve()
    package_root = current.parents[1]  # microfluidic_control_system
    workspace_root = current.parents[2]
    return package_root, workspace_root


def _candidate_pythons() -> list[pathlib.Path]:
    package_root, workspace_root = _paths()
    return [
        package_root / "backend" / "venv" / "Scripts" / "python.exe",
        workspace_root / ".venv" / "Scripts" / "python.exe",
        package_root / ".venv" / "Scripts" / "python.exe",
    ]


def _python_has_required_modules(python_exe: pathlib.Path) -> bool:
    check_cmd = [
        str(python_exe),
        "-c",
        (
            "import importlib.util,sys; "
            f"mods={REQUIRED_MODULES!r}; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "sys.exit(0 if not missing else 1)"
        ),
    ]
    try:
        result = subprocess.run(
            check_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
        return result.returncode == 0
    except Exception:
        return False


def _pick_existing_venv_python() -> pathlib.Path | None:
    for candidate in _candidate_pythons():
        if candidate.exists():
            return candidate
    return None


def _pick_ready_venv_python() -> pathlib.Path | None:
    for candidate in _candidate_pythons():
        if candidate.exists() and _python_has_required_modules(candidate):
            return candidate
    return None


def _ensure_project_root_on_path() -> pathlib.Path:
    _, workspace_root = _paths()
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return workspace_root


def _maybe_relaunch_with_venv_python() -> bool:
    if os.environ.get(RELAUNCH_FLAG) == "1":
        return False

    # 优先使用 backend/venv，其次使用其它可用 venv。
    venv_python = _pick_ready_venv_python() or _pick_existing_venv_python()
    if venv_python is None:
        return False
    try:
        if pathlib.Path(sys.executable).resolve() == venv_python.resolve():
            return False
    except Exception:
        pass

    env = dict(os.environ)
    env[RELAUNCH_FLAG] = "1"
    this_file = pathlib.Path(__file__).resolve()
    workspace_root = _ensure_project_root_on_path()
    subprocess.Popen([str(venv_python), str(this_file)], cwd=str(workspace_root), env=env)
    return True


def _missing_runtime_modules() -> list[str]:
    missing: list[str] = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return missing


def _print_env_help(missing: list[str]) -> None:
    _, workspace_root = _paths()
    venv_python = _pick_ready_venv_python() or _pick_existing_venv_python()
    print("前端启动失败：当前环境未满足依赖。", file=sys.stderr)
    print(f"缺少模块: {', '.join(missing)}", file=sys.stderr)
    if venv_python is not None:
        print(f"请使用已存在且依赖完整的虚拟环境启动: {venv_python}", file=sys.stderr)
    else:
        print("未检测到可用虚拟环境（backend/venv 或 .venv）。", file=sys.stderr)
    print("不会自动下载依赖，请先准备好虚拟环境后再启动。", file=sys.stderr)
    print(f"当前工作目录: {workspace_root}", file=sys.stderr)


def main() -> None:
    if _maybe_relaunch_with_venv_python():
        return
    _ensure_project_root_on_path()
    missing = _missing_runtime_modules()
    if missing:
        _print_env_help(missing)
        raise SystemExit(1)

    from microfluidic_control_system.frontend.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
