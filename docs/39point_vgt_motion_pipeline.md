# 39 点小猫 VGT 运动流程

## 任务定义

本流程完成以下可复现实验：

```text
视频 + 已有初始 Mesh
-> 39 点二维关键点和质量
-> Mesh 辅助的身体相对 2.5D 运动
-> 39 点三维关键点序列
-> QianJi 12 节点、30 根杆 VGT 模型序列
-> 30 FPS 结构动画
```

初始 GLB 来自视频截帧后使用 Hunyuan3D 生成的闭合模型，再经 Assimp 转为
GLB。流水线不在运行时重新生成三角 Mesh，原始 2,154 个顶点也不参与运动
传播。

QianJi 的 Mesh 转 VGT 预处理曾产生一个 12-site/30-rod 模型。该模型现在以
冻结模板保存，并绑定源 Mesh SHA-256、QianJi commit、生成参数和模板哈希。
一键运动实验会验证这些记录，而不重新运行会改变节点数和节点身份的四面体
采样。冻结的是“初始 VGT 模型”，不是运动结果；每次运行仍会重新计算 39 点、
2.5D、控制、可达性、候选选择和 272 帧 VGT 序列。

## 为什么 2.5D 不是完整 3D

每个输出点确实有三个坐标，但信息来源不是三个独立观测维度：

- forward 和 up 的相对变化来自视频二维关键点；
- lateral 来自 Mesh/VGT 中性结构并保持固定；
- 相机平移、画面内旋转和均匀缩放被身体坐标归一化消除。

所以它可用于“把视频动作重定向到一个三维结构”，不能用于声称从视频测得
真实深度、世界位移、完整关节角或 Mesh 表面形变。输出统一标记为
`body_relative_2_5d_retarget`。

## 处理逻辑

### 1. 39 点二维观测

SuperAnimal H5 的 39 个 bodypart 原名和顺序逐帧保留。置信度低于 0.30、
越界或相对躯干尺度异常跳变的点变为显式无效，不被插值。前后腿各自按
thigh/knee/paw 整条链纠正身份，避免只交换足端。

H5 必须使用严格的 `RangeIndex(0, 272)`，视频、H5 和人工校正轨迹必须具有
相同帧数、时间基和来源哈希。人工校正文件中的前后腿 `keep/swap` 锚点会
显式传入身份解析器。身份模糊时，同侧区域的左右两条完整
thigh/knee/paw 链都设为无效，同时保留 raw 坐标供审计，不能直接驱动 VGT。

真实案例是 272 帧、30 FPS、1246x720，参考帧为 152，人工锚点为前腿
`keep`、后腿 `keep`。后腿身份在第 116 帧发生一次链级交换，模糊帧为
74、112、193、229。

### 2. 中性 39 地标

每个角色使用参考帧附近 `+/-15` 帧内最近的有效观测建立中性参考；该窗口内
仍不可用的角色采用显式中性替代，不能用无效坐标定义整段运动零点。二维点
随后投影到校正躯干的局部纵向/竖直坐标。QianJi rig 的两个 spine site 定义
三维 forward，中性结构的全局 Z 正交分量定义 up，两者叉积定义 lateral。
左右角色获得相反的固定 lateral 偏移，中心角色为零。

四个 antler 角色因为猫不适用而保留为命名占位，置信度为零。这样既不伪造
解剖结构，也不破坏固定 39 角色契约。

### 3. 逐帧 2.5D 重定向

对每个有效角色，计算它在逐帧躯干坐标系中的归一化位置与参考位置之差，只
把 longitudinal/vertical 两个分量传给三维中性地标。无效角色使用记录了
原因和来源帧的中性替代，不做跨帧插值或平滑。真实选择结果使用
`motion_scale=0.10`，不从单目视频伪造 lateral 深度运动。

### 4. 六个结构控制

39 点完整输出和六个控制目标是两个层次。六个控制只负责让现有 QianJi
几何可达性工具工作：

- 两个躯干控制读取背部地标位移，并有明确同帧备用；
- 四个足控制读取对应 paw 位移；
- 每个位移加到分配的 VGT site 中性坐标，而不是直接把地标 XYZ 当 site。

除了 QianJi 的 `quadruped_bbox` rig，还计算一个全局最小距离、site 不重复的
motion-informed rig。

### 5. 可达性和收缩范围

每根杆的 10% permitted contraction 表示：

```text
effective_min_length = 0.90 * effective_current_length
```

`effective_current_length` 和 `effective_max_length` 不变。这不是把动作缩小到
10%，而是允许整根有效杆长最多收短 10%。双侧滑杆控制范围写成：

```text
slide_min_each_side = -0.5 * 0.10 * effective_current_length
slide_max_each_side =  0.5 * (effective_max_length - effective_current_length)
```

因此两侧合计才是 10%，不会在 QianJi 控制、JSON 和 MuJoCo XML 中重复计算。
最终 XML 由支持负滑杆下限的 QianJi converter 从被选中的 canonical robot
重新生成，并逐杆核对 joint range 和 actuator ctrlrange。

实验包含 7 个候选，比较 0.05/0.10/0.15 三种运动尺度、0/10% 收缩、
bbox/motion-informed rig，以及不带收缩的固定拓扑 morphology 优化。候选
只有同时满足以下条件才可被选择：

- 272 帧、12 site、30 rod，所有数组有限且时间严格对应 30 FPS；
- unreachable 为 0，最大六控制点投影误差不超过 0.05 m；
- QianJi 报告及从 NPZ 独立重算的最大杆长违反都不超过 0.0005 m；
- 最大估计 clipping 比例和独立重算的违反杆比例都不超过 0.05；
- robot 的实际收缩范围、候选声明和 `extension_only` 完全一致。

### 6. VGT 序列和动画

QianJi 的 `feasible_site_targets.npz` 被严格校验并重新打包为
`vgt_motion.npz`。`272 x 12 x 3` 表示：

- 272 个视频时刻；
- 每个时刻 12 个 VGT 结构节点；
- 每个节点一个 `(x,y,z)` 位置。

30 根杆是 robot 中固定的 30 对节点连接。动画每帧用当帧 12 个位置画出相同
的 30 条连接，输出 top XY、side XZ、front YZ、isometric 四个视图。

## 真实结果

严格自动验收案例位于：

```text
outputs/experiments/39point_vgt_cat_v5
```

最终选择 `scale_010_motion_informed_base_c010`：

| 指标 | 结果 |
| --- | ---: |
| 运动尺度 | 0.10 |
| permitted contraction | 10% |
| rig | motion-informed |
| feasible | 266 / 272 |
| marginal | 6 / 272 |
| unreachable | 0 / 272 |
| 最大控制点误差 | 0.01609 m |
| 最大杆长违反（报告和独立重算） | 0.000282492 m |
| 单帧最大违反杆比例 | 1 / 30 = 0.03333 |
| VGT positions | `(272, 12, 3)` |
| 动画 | 1246x720、30 FPS、272 帧 |

这比 bbox、无收缩的 `scale=0.10` 明显改善：后者为 1 feasible、259
marginal、12 unreachable，最大误差 0.05277 m。motion-informed + 10% 收缩
把不可达帧降到零，并让 266 帧进入严格 feasible；还剩 6 帧 marginal，说明
它通过本项目外层几何门禁，但并非每帧都达到 QianJi 更严的 `feasible`
分类，更不等于动力学步态有效。

Desired 控制目标与 QianJi projected 控制目标是不同文件和不同哈希。最终
`vgt_motion.npz` SHA-256 为
`44d54913c289260f1690adebf5936ee8255b66985497a17474521179db03dc6f`。

`final_acceptance_report.json` 的全部自动检查为 `passed: true`。抽取帧检查
确认二维覆盖图和四视图结构图非空；完整运动的人工视觉验收仍应在合并前
完成，因此 PR 保持 Draft。`v4` 保留为前一份通过记录；`v5` 进一步把所有
候选元数据改为可移动的相对路径，并已在发布后的最终目录重新运行候选比较。

## 一键运行

```bash
uv pip install --python .venv/bin/python -e .

ANIMAL_DATA_ROOT="/absolute/qianji-animal-motion" \
QIANJI_ROOT="/absolute/QianJi" \
OUTPUT_ROOT="$PWD/outputs/experiments/39point_vgt_cat_v6" \
bash experiments/39point_vgt_cat/run_experiment.sh
```

输出目录必须不存在。脚本先在同级隐藏 staging 目录中构建整个案例，失败时
清理，全部通过后才原子发布到 `OUTPUT_ROOT`。脚本对 QianJi 只读，所有收缩
robot 和 morphology robot 都写入实验目录。最终记录两个仓库 commit、dirty
状态、脚本哈希、Python 包版本和真实调用参数，并运行 `verify_case.py`。

## 完成标准

- 39 个二维角色和 39 个 2.5D 角色逐帧齐全；
- 角色无缺失，所有 JSON/NPZ 有限，不跨帧插值或平滑；
- robot 为 12 site、30 rod，端点引用有效；
- VGT NPZ 为 `(272,12,3)`，时间严格递增且对应 30 FPS；
- desired/projected 分别与六控制语义、所选 rig 和 NPZ site 位置一致；
- unreachable 为零，控制误差、杆长误差和 clipping 比例均通过门禁；
- 2D/VGT 视频均为 30 FPS、272 帧且非空；
- 输入、冻结初始 VGT、selected robot/XML、rig、控制轨迹和输出均可溯源；
- 明确不主张度量深度、相机标定、全局位移、动力学或 Mesh 顶点形变。
