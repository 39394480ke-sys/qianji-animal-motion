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

## 完整 39 点观测

`qianji-export-39-keypoints` 发布四个文件：

- `input_manifest.json`：视频、H5、GLB、人工校正六点轨迹的绝对路径和
  SHA-256；视频属性；H5 scorer/individual/bodyparts；Hunyuan3D 来源声明；
  不重建 Mesh、不观察深度、不做动力学、不变形顶点的限制。
- `keypoint_trajectory_2d_39.json`：Schema
  `qianji.keypoint_trajectory_2d_39`，严格包含源 H5 的 39 个角色和全部帧。
- `keypoint_39_quality_report.json`：逐角色低置信度、越界、时序跳变、腿部
  身份状态和无效帧；`interpolation_applied` 与 `smoothing_applied` 均为
  `false`。
- `keypoint_39_preview.mp4`：保持源视频尺寸、帧率和帧数的检查视频。

每个二维点同时保留 `raw_x_px/raw_y_px`。有效点有有限的 `x_px/y_px`；
无效点的输出坐标为 `null`，但角色不会消失。

## 39 点中性地标与 2.5D 运动

`neutral_landmarks_39.json` 的 Schema 是
`qianji.neutral_landmarks_39`。它包含参考帧、VGT 三维 origin、正交的
forward/up/lateral 基、39 个中性 XYZ、左右侧语义和参考置信度。猫不具备的
四个 antler 角色仍保留，但 `anatomy_applicable: false`。

`keypoint_motion_3d_39.json` 的 Schema 是
`qianji-keypoint-trajectory-39-v1`。每帧严格包含 39 个
`[x, y, z, confidence]`。其定义为：

```text
p_k(t) = neutral_k
       + motion_scale * torso_length_3d
       * (delta_long_k(t) * forward_3d
          + delta_vertical_k(t) * up_3d)
```

`delta_long` 和 `delta_vertical` 来自每帧校正躯干坐标系相对参考帧的变化。
相机平移、画面内旋转和均匀缩放被消除；lateral 由中性 rig 固定，不从单目
视频估计。无效点回到自身中性 XYZ、置信度为零，并在
`lift_39_report.json` 中逐项记录。

## 观测与 VGT 控制

`vgt_control_map.json` 的 Schema 是 `qianji.vgt_control_map`。39 点是观测
输出，只有以下六个结构角色交给 QianJi 几何控制器：

| 控制角色 | 主观测 | 同帧备用 |
| --- | --- | --- |
| `spine_front` | `back_base` | `neck_base` |
| `spine_rear` | `back_end` | `tail_base` |
| `front_left_foot` | `front_left_paw` | 无 |
| `front_right_foot` | `front_right_paw` | 无 |
| `rear_left_foot` | `back_left_paw` | 无 |
| `rear_right_foot` | `back_right_paw` | 无 |

控制目标等于“结构 site 中性 XYZ + 对应观测相对自身中性地标的位移”。因此
参考帧严格回到 rig，且不会错误地把解剖地标绝对位置当作结构节点位置。主
观测无效时只能使用表中的同帧备用；否则回到 site 中性位置且置信度为零。

## VGT 模型序列

`vgt_motion.npz` 必须包含：

- `site_names`：形状 `(12,)`，顺序与 `robot.json` 一致且无重复；
- `times`：形状 `(272,)`，有限且严格递增；
- `positions`：形状 `(272, 12, 3)`，全部有限。

`robot.json` 必须有 12 个 site 和 30 个 rod，每根杆的两个端点必须引用现有
site。`vgt_motion_manifest.json` 记录上述形状、帧率、时间范围、30 根杆、
robot/rig/目标文件的路径与哈希，以及不同的 desired 和 projected 控制轨迹。

`positions` 是抽象 VGT 节点序列，不是 2,154 个原始 Mesh 顶点的形变序列。
当前流程也不做步态动力学仿真。
