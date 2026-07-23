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

QianJi 后续若需要世界坐标 `[x, y, z, confidence]`，还必须经过相机标定、
多视角三角化或经过验证的单目三维估计，再进行骨架绑定和尺度对齐。本仓库
目前不提供这一步，也不会用固定深度或零值补出伪三维数据。
