# 数据契约

本项目输出的是可追溯的动物二维关键点证据，不会把单目像素坐标伪装成三维
世界坐标。

## 自动六点轨迹

- Schema：`qianji.keypoint_trajectory_2d`
- 当前版本：`1.2.0`
- 坐标系：图像左上角为原点，x 向右，y 向下，单位为像素
- 无效点：`valid: false`，且 `x_px`、`y_px` 为 `null`
- 自动过程不使用跨帧插值或平滑

每个点包含模型来源、模型置信度、有效性、质量标记、身份修正和备用来源信息。
`mapping_report.json` 记录无效帧、备用来源帧、身份修正区间和身份模糊帧。
轨迹和报告都记录源视频与 H5 的 SHA-256。

## 人工校正轨迹

- Schema：`qianji.keypoint_trajectory_2d`
- 有人工修改时版本：`1.3.0`
- 没有人工修改时：字节语义保持基线版本

人工移动、恢复或置空会写入 `position_source: manual`、`review_status` 和
`correction`。点上的 `confidence` 仍是原始模型置信度，不代表人工标注的
置信程度。下游判断来源时必须读取 `position_source`，不能只看
`confidence`。人工轨迹还记录基线、CVAT manifest 和 annotations XML 的
SHA-256，便于独立追溯。

## 左右身份

`front_left_foot` 等左右名称始终以动物自身为准。锚点 manifest 会记录
`camera_side`：

- `animal_left_visible`
- `animal_right_visible`
- `unknown`

当值为 `unknown` 时，左右身份仍是待人工确认的假设，不能把它当作已证明的
解剖学标签。应在 CVAT 中结合整段运动和拍摄信息完成复核。

## QianJi 边界

### 身体相对 2.5D 输出

`qianji-lift-keypoints` 输出三份不可覆盖、整体发布的文件：

- `keypoint_motion.json`：Schema 为 `qianji-keypoint-trajectory-v1`，每点
  为 `[x, y, z, confidence]`。
- `mesh_binding.json`：六个角色到 QianJi site 的映射、中性位置、三维坐标
  基，以及 trajectory、robot、rig 的绝对路径和 SHA-256。
- `lift_report.json`：参考帧、尺度、位移统计、逐点替代记录和科学边界。

每帧以两个躯干点定义身体二维纵向，以画面向上定义局部竖直，并用该帧躯干
长度归一化。相对于参考帧的纵向、竖直变化映射到 QianJi 中性 rig 的 forward
和 up 方向。Mesh rig 原有的 lateral 坐标保留，不从视频推断深度。因此报告
固定包含：

```text
reconstruction_kind: body_relative_2_5d_retarget
metric_depth_observed: false
camera_calibrated: false
global_translation_preserved: false
```

单点无效时输出该角色的中性坐标和置信度 `0.0`；躯干无效时整帧如此处理。
不会跨帧插值、平滑或沿 lateral 方向制造运动。QianJi 当前控制器不会用
confidence 自动屏蔽目标，所以中性替代也是控制安全策略，而非缺失值填零。

### 不可替代的真实三维

需要相机或世界坐标中的真实三维运动时，仍必须使用相机标定、多视角三角化
或经过验证的单目三维估计，并进行尺度与骨架校准。2.5D 输出不能用于声称
观测到了深度、全局位移、生物关节角或完整 Mesh 形变。
