# qianji-animal-motion

为 QianJi 准备动物静态 3D Mesh 和视频关键点轨迹等生物输入。
当前工具包含 FBX 转 GLB 以及动物视频标准化。

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

## FBX 转 GLB

静态 3D Mesh 统一转换为 `QianJi` 标准输入格式 GLB。转换使用轻量级
Assimp，不需要安装 Blender。macOS 首次使用前安装：

```bash
brew install assimp
```

转换单个 FBX：

```bash
qianji-convert-mesh data/raw/cat.fbx \
  --output data/processed/cat.glb
```

批量转换目录中的 FBX：

```bash
qianji-convert-mesh data/raw --output data/processed
```

程序固定导出二进制 glTF 2.0，并验证 mesh、顶点、三角面和有限包围盒。
只有转换和验证全部成功才会生成最终 `.glb`，同时写入同名
`.metadata.json`。默认不覆盖已有文件；明确需要替换时使用 `--overwrite`。

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
## 六点语义映射

DeepLabCut SuperAnimal-Quadruped 的 39 点原始结果保留不变。中间层读取
H5 预测，将 `back_base`、`back_end` 分别映射为 `spine_front`、
`spine_rear`，并将四个 paw 映射为四个足端。

正式映射前，先生成最多 12 个左右腿清晰、置信度高的候选锚点：

```bash
qianji-suggest-anchor \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --predictions outputs/zero_shot/cat_walk/predictions.h5 \
  --output outputs/anchor_review/cat_walk
```

程序会生成 `identity_anchor_candidates.jpg` 和
`identity_anchor_candidates.json`。打开拼图，以动物自身方向选择一帧，
并分别确认 DLC 前腿和后腿标签是正确（`keep`）还是需要交换（`swap`）。
随后执行正式映射：

```bash
qianji-map-keypoints \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --predictions outputs/zero_shot/cat_walk/predictions.h5 \
  --anchor-manifest outputs/anchor_review/cat_walk/identity_anchor_candidates.json \
  --anchor-frame 152 \
  --front-anchor keep \
  --rear-anchor keep \
  --output outputs/semantic_six/cat_walk
```

人工锚点是正式结果的必需输入。候选 manifest 保存视频和 H5 的 SHA-256，
映射时会验证哈希，防止混用输入。前腿和后腿分别从锚点向视频开头和结尾
双向追踪。完全遮挡不会清除历史；重叠但仍有观测的帧保留当前身份并继续
推进观测时间，避免把正常的腿部交叉误判为标签交换。无法可靠区分时，对应
足端标记为 `identity_ambiguous` 并置空，不进行跨帧插值。

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
