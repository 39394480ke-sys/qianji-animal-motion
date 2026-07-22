# qianji-animal-motion

从动物视频提取关键点，并整理为可用于仿生运动分析的时间序列轨迹。

当前标注方案包含六个关键点：`nose`（鼻尖）、`left_ear`（左耳）、`right_ear`（右耳）、`body_center`（躯干中心）、`tail_base`（尾根）和 `tail_tip`（尾尖）。

## 安装

建议使用 Conda/Mamba：

```bash
conda env create -f environment.yml
conda activate qianji-animal-motion
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

后续集成时，只将本项目中自有的代码复制到 `QianJi/video_pose_extraction`，不克隆或搬运上游 DeepLabCut 源码。
