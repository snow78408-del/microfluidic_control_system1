# microfluidic_control_system

本项目当前用于微流控液滴系统的结构化分层，现阶段仅进行目录职责划分，不包含具体业务实现代码。

## 目录结构

```text
microfluidic_control_system/
├── frontend/
└── backend/
    ├── vision/
    ├── pid_control/
    ├── pump_hardware/
    ├── orchestrator/
    └── venv/
```

## 子目录职责概览

- `frontend/`：前端用户交互页面与状态展示层。
- `backend/vision/`：图像识别与统计输出。
- `backend/pid_control/`：PID 反馈控制逻辑（当前仅保留平均直径反馈链路）。
- `backend/pump_hardware/`：泵硬件连接、下发与停机控制。
- `backend/orchestrator/`：后端主流程耦合与状态调度入口。
- `backend/venv/`：后端虚拟运行环境目录（可先保留结构说明）。
