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

## 六点语义映射

DeepLabCut SuperAnimal-Quadruped 的 39 点原始结果保留不变。中间层读取
H5 预测，将 `back_base`、`back_end` 分别映射为 `spine_front`、
`spine_rear`，并将四个 paw 映射为四个足端：

```bash
qianji-map-keypoints \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --predictions outputs/zero_shot/cat_walk/predictions.h5 \
  --output outputs/semantic_six/cat_walk
```

程序用 `back_end -> back_middle -> back_base` 背部折线检查躯干结构，并且
只在可靠帧证明相对位置稳定后，允许用同一帧的 `neck_end`、`tail_base`
分别作为两个躯干点的备用来源。如果前躯两个来源都失效，但可靠帧已证明
背部比例稳定，还可由同一帧的 `back_end` 和 `back_middle` 外推
`spine_front`。程序也会检查足端与大腿、膝关节的结构关系，
并使用整段时序连续性纠正左右足身份交换。低置信度、越界、异常跳变、结构
异常或无法可靠判断身份的点会输出 `x_px: null`、`y_px: null` 和对应
`flags`；不会跨帧插值或平滑，也不会修改原始 39 点 H5。每个六点结果记录
`source`、`confidence`、`valid`、`flags`、`identity_corrected` 和
`fallback_used`。输出包含 `keypoint_trajectory_2d.json`、
`mapping_report.json` 和 `six_keypoints_preview.mp4`。默认拒绝覆盖已有结果，
明确需要重跑时使用 `--overwrite`。

后续集成时，只将本项目中自有的代码复制到 `QianJi/video_pose_extraction`，不克隆或搬运上游 DeepLabCut 源码。
