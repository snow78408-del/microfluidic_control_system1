# microfluidic_control_system

微流控液滴控制系统。项目按职责拆成 `frontend/` 与 `backend/`，顶层 `run.py` 是统一启动入口，不再绑定 `backend/venv`、PowerShell 或批处理脚本。

## 目录结构

```text
microfluidic_control_system/
├── run.py
├── pyproject.toml
├── requirements.txt
├── frontend/
└── backend/
    ├── vision/
    ├── pid_control/
    ├── pump_hardware/
    └── orchestrator/
```

## 启动方式

默认启动前端界面：

```bash
python run.py
```

也可以显式启动：

```bash
python run.py frontend
```

独立运行视觉流水线：

```bash
python run.py vision --video input.mp4
python run.py vision --camera 0
```

`vision` 后面的参数会继续传给 `backend/vision/run_vision.py`。

## 环境配置

任选一种 Python 环境管理方式即可，项目不依赖固定目录中的虚拟环境。

普通 Python / venv：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

Windows 中不激活环境也可以直接运行虚拟环境里的 Python：

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

conda：

```bash
conda create -n microfluidic-control python=3.11
conda activate microfluidic-control
conda install -c conda-forge numpy opencv pyserial
python run.py
```

uv：

```bash
uv sync
uv run python run.py
```

安装为命令行工具后也可运行：

```bash
python -m pip install -e .
microfluidic-control
```

## 子目录职责概览

- `frontend/`：前端用户交互页面与状态展示层。
- `backend/vision/`：图像识别与统计输出。
- `backend/pid_control/`：PID 反馈控制逻辑（当前仅保留平均直径反馈链路）。
- `backend/pump_hardware/`：泵硬件连接、下发与停机控制。
- `backend/orchestrator/`：后端主流程耦合与状态调度入口。
