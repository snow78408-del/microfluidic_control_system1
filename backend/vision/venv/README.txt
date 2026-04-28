vision/venv 目录说明
=====================

该目录用于放置图像识别模块（vision）的虚拟运行环境。
当前阶段即使不创建完整二进制环境，也保留该目录和说明文件。

1) 创建虚拟环境（Windows PowerShell）
   python -m venv venv

2) 激活虚拟环境
   .\venv\Scripts\Activate.ps1

3) 安装依赖
   python -m pip install --upgrade pip
   pip install -r ..\requirements.txt

4) 运行 vision 模块入口
   cd ..
   python run_vision.py --video <your_video_path>

5) 示例（启用 Kalman + 可视化）
   python run_vision.py --video input.mp4 --tracker kalman --display
