# CVAT 六点人工校正教程

本教程用于检查和修正 `qianji-map-keypoints` 生成的六点二维轨迹。CVAT 只
承担网页界面；原始 39 点预测、自动六点映射和人工修正结果分别保存。

## 1. 安全和回退原则

整个流程有三层数据，绝不相互覆盖：

```text
outputs/zero_shot/              DeepLabCut 原始 39 点
outputs/semantic_six/           自动生成的六点基线
outputs/manual_correction/      人工修正的一个或多个版本
```

导出、导回命令发现目标文件已经存在时会直接停止。不要删除或改写
`outputs/semantic_six/cat_walk/keypoint_trajectory_2d.json`。每次导回使用新目录：

```text
review_v1
review_v2
review_v3
```

某次效果不好时，停止使用该目录并退回上一版即可。删除人工校正目录也不会影响
原始预测或自动六点结果。

## 2. 安装项目

```bash
mamba activate animal_pose
pip install -e '.[dev]'
```

确认命令可用：

```bash
qianji-export-cvat --help
qianji-import-cvat --help
```

## 3. 生成 CVAT 导入包

在项目根目录运行：

```bash
qianji-export-cvat \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --trajectory outputs/semantic_six/cat_walk/keypoint_trajectory_2d.json \
  --report outputs/semantic_six/cat_walk/mapping_report.json \
  --output outputs/manual_correction/cat_walk/cvat_export
```

输出：

```text
cvat_export/
├── annotations.xml
├── cvat_manifest.json
├── review_queue.json
└── skeleton_definition.json
```

- `annotations.xml`：需要导入 CVAT 的逐帧六点 Skeleton。
- `cvat_manifest.json`：保存视频、基线轨迹和报告的哈希，防止导错文件。
- `review_queue.json`：建议优先检查的异常帧。
- `skeleton_definition.json`：六点名称和连线定义。

程序会核对视频尺寸、帧数和帧率，并校验轨迹与映射报告中的帧索引和来源
哈希。导出期间任一输入发生变化时，不会发布导出包。

## 4. 创建 CVAT Online 任务

1. 打开 [CVAT Online](https://app.cvat.ai/) 并登录。
2. 选择 `Tasks`，点击 `+` 创建任务。
3. 任务名可填写 `qianji-cat-walk-six-point-review`。
4. 上传 `data/processed/cat_walk_30fps_720p.mp4`。
5. 新建一个 Skeleton 标签，名称必须是 `quadruped_6`。
6. 按以下顺序创建六个点，名称必须完全一致：

```text
spine_front
spine_rear
front_left_foot
front_right_foot
rear_left_foot
rear_right_foot
```

7. 添加以下连线；连线只用于观察，不代表完整解剖骨架：

```text
spine_rear  -> spine_front
spine_front -> front_left_foot
spine_front -> front_right_foot
spine_rear  -> rear_left_foot
spine_rear  -> rear_right_foot
```

8. 创建并打开任务。
9. 在任务的 `Actions` 中选择上传标注。
10. 格式选择 `CVAT for video 1.1`，上传 `cvat_export/annotations.xml`。

如果提示标签不匹配，先检查 Skeleton 名称和六个点名，不要为了通过导入而修改
XML。

## 5. 检查和拖动关键点

先打开 `review_queue.json`。每条记录包含帧号、时间、优先级和原因。

检查顺序：

1. `critical`：置空、身份模糊或异常跳变。
2. `high`：左右身份修正边界，以及备用来源区间的起点、中点和终点。
3. `context`：异常帧前后的上下文。
4. 完成重点帧后，以 `0.5x` 速度播放整段视频。

CVAT 中的操作原则：

- 点位明确错误：把该点拖到正确位置。
- 左右足身份错误：分别把左右点拖回对应足端，不要修改点名。
- 原来置空但现在能明确判断：取消该点的 `Outside`，再放到正确位置。
- 无法可靠判断：将该点设为 `Outside`。
- 不使用跨帧插值补足端轨迹。
- 不删除整个 `quadruped_6` Skeleton track。

导入文件已经把所有帧设成独立关键帧，因此拖动一帧不会自动改动前后帧。

## 6. 导出人工结果

完成一轮修改后：

1. 回到 CVAT 任务页面。
2. 选择导出标注。
3. 格式必须选择 `CVAT for video 1.1`。
4. 不需要包含视频或图片。
5. 解压下载文件，找到 `annotations.xml`。

不要选择普通 `COCO`。本项目使用 CVAT 原生视频 XML，以保留逐帧 Skeleton、
Outside 和 Occluded 状态。

## 7. 导回并生成预览

第一轮使用 `review_v1`：

```bash
qianji-import-cvat \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --baseline outputs/semantic_six/cat_walk/keypoint_trajectory_2d.json \
  --manifest outputs/manual_correction/cat_walk/cvat_export/cvat_manifest.json \
  --annotations ~/Downloads/annotations.xml \
  --output outputs/manual_correction/cat_walk/review_v1
```

输出：

```text
review_v1/
├── keypoint_trajectory_2d_corrected.json
├── corrections.json
├── correction_report.json
└── six_keypoints_corrected_preview.mp4
```

- `keypoint_trajectory_2d_corrected.json` 是供后续流程使用的最终候选轨迹。
- `corrections.json` 只记录人工移动、恢复或置空的点。
- `correction_report.json` 汇总修改数量和剩余无效帧。
- `six_keypoints_corrected_preview.mp4` 用于完整复看。

如果视频、基线或 manifest 不是同一轮文件，哈希检查会拒绝导入。

人工移动点的 `confidence` 仍保留原始模型分数；是否人工修改应读取
`position_source: manual` 和 `review_status`，不能把模型置信度当成人工
置信度。

## 8. 反复复查

打开预览：

```bash
open outputs/manual_correction/cat_walk/review_v1/six_keypoints_corrected_preview.mp4
```

重点观察：

- 两个躯干点是否沿背部稳定移动。
- 足端是否跟随对应脚掌。
- 腿交叉时左右身份是否保持。
- 是否存在单帧跳点。
- 所有空值是否确实无法可靠判断。

仍有问题时直接回到原 CVAT 任务继续修改，再次导出 XML，并导入新目录：

```bash
qianji-import-cvat \
  --video data/processed/cat_walk_30fps_720p.mp4 \
  --baseline outputs/semantic_six/cat_walk/keypoint_trajectory_2d.json \
  --manifest outputs/manual_correction/cat_walk/cvat_export/cvat_manifest.json \
  --annotations ~/Downloads/annotations_v2.xml \
  --output outputs/manual_correction/cat_walk/review_v2
```

注意：每一轮都相对同一个自动六点基线计算完整 correction log，不要用
`review_v1` 充当下一轮 baseline。

## 9. 如何回退

### 退回自动六点结果

后续程序继续使用：

```text
outputs/semantic_six/cat_walk/keypoint_trajectory_2d.json
```

### 退回某轮人工结果

把后续输入改回：

```text
outputs/manual_correction/cat_walk/review_v1/keypoint_trajectory_2d_corrected.json
```

### 放弃整个 CVAT 实验

停止使用 `outputs/manual_correction/` 即可。原始 H5 和自动六点目录均不会被
修改，不需要重新运行 DeepLabCut。

## 10. 完成标准

- 校正预览覆盖基线报告中的全部帧且能完整播放。
- 六个点没有明显单帧跳变。
- 四个足端没有可见的长期左右交换。
- 所有剩余空值在报告中有明确记录。
- `correction_report.json` 显示 `interpolation_used: false`。
- 原始 39 点 H5 和自动六点 JSON 哈希未变化。
