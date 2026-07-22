# qianji-animal-motion

从动物视频提取关键点，并整理为可用于仿生运动分析的时间序列轨迹。

当前输出固定包含六个语义关键点：

- `spine_front`（躯干前部）
- `spine_rear`（躯干后部）
- `front_left_foot`（左前足）
- `front_right_foot`（右前足）
- `rear_left_foot`（左后足）
- `rear_right_foot`（右后足）

## 安装

建议使用 Conda/Mamba：

```bash
conda env create -f environment.yml
conda activate animal_pose
```

或使用现有 Python 环境：

```bash
pip install -e '.[gui,dev]'
```

## 验证

```bash
pytest
python -c "from qianji_animal_motion import __version__; print(__version__)"
```

原始视频放入 `data/raw/`，处理后的中间数据放入 `data/processed/`，运行结果放入 `outputs/`；这些实际数据均不提交 Git。

## 视频预处理

所有送入 DeepLabCut 的视频先统一为恒定 30 FPS、720p、H.264、
`yuv420p` 且不含音频的 MP4。程序不会裁剪画面，也不会覆盖原视频。

处理单个视频：

```bash
qianji-preprocess-video data/raw/cat_walk.mov \
  --output data/processed/cat_walk_30fps_720p.mp4
```

批量处理目录中的视频：

```bash
qianji-preprocess-video data/raw --output data/processed
```

默认输出旁边会生成同名的 `.metadata.json`，记录转换参数以及转换前后的
编解码、尺寸、帧率和音频信息。输出文件已经存在时程序会拒绝覆盖；明确需要
替换时使用 `--overwrite`。

后续集成时，只将本项目中自有的代码复制到 `QianJi/video_pose_extraction`，不克隆或搬运上游 DeepLabCut 源码。
