# backend/venv 统一虚拟环境说明

本目录是 **microfluidic_control_system 整个项目** 的统一 Python 虚拟环境路径。

- 固定路径：`D:\学习\microfluidic_control_system\backend\venv`
- 统一依赖文件：`requirements.txt`
- 一键配置脚本：`setup_venv.ps1` / `setup_venv.bat`

## 1. 依赖范围
当前依赖覆盖：
- `backend/vision`：`numpy`、`opencv-python`
- `backend/pump_hardware`：`pyserial`
- `backend/pid_control`、`backend/orchestrator`：使用标准库（无额外第三方）
- `frontend`：基于 Tkinter（Python 标准库）

## 2. 一键配置（推荐）
### PowerShell
```powershell
D:\学习\microfluidic_control_system\backend\venv\setup_venv.ps1
```

### CMD
```bat
D:\学习\microfluidic_control_system\backend\venv\setup_venv.bat
```

## 3. 手动配置
```powershell
python -m venv D:\学习\microfluidic_control_system\backend\venv
D:\学习\microfluidic_control_system\backend\venv\Scripts\python.exe -m pip install --upgrade pip
D:\学习\microfluidic_control_system\backend\venv\Scripts\python.exe -m pip install -r D:\学习\microfluidic_control_system\backend\venv\requirements.txt
```

## 4. 激活方式
### PowerShell
```powershell
D:\学习\microfluidic_control_system\backend\venv\Scripts\Activate.ps1
```

### CMD
```bat
D:\学习\microfluidic_control_system\backend\venv\Scripts\activate.bat
```

## 5. 运行示例
```powershell
# 前端
D:\学习\microfluidic_control_system\backend\venv\Scripts\python.exe D:\学习\microfluidic_control_system\frontend\run.py
```
