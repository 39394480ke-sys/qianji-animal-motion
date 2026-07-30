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
GLB。流水线不在运行时重新生成三角 Mesh。QianJi 只把该 Mesh 转换为抽象
12-site/30-rod VGT，原始 2,154 个顶点不参与运动传播。

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

真实案例是 272 帧、30 FPS、1246x720。参考帧是 152。后腿身份在第 116 帧
发生一次链级交换，模糊帧为 74、112、193、229；这些证据都在质量报告中。

### 2. 中性 39 地标

参考帧的二维点先投影到校正躯干的局部纵向/竖直坐标。QianJi rig 的两个
spine site 定义三维 forward，中性结构的全局 Z 正交分量定义 up，两者叉积
定义 lateral。左右角色获得相反的固定 lateral 偏移，中心角色为零。

四个 antler 角色因为猫不适用而保留为命名占位，置信度为零。这样既不伪造
解剖结构，也不破坏固定 39 角色契约。

### 3. 逐帧 2.5D 重定向

对每个有效角色，计算它在逐帧躯干坐标系中的归一化位置与参考位置之差，只
把 longitudinal/vertical 两个分量传给三维中性地标。无效角色回到中性位置。
真实选择结果使用 `motion_scale=0.10`，最大 lateral 数值误差约
`1.89e-18 m`，即没有人为制造侧向运动。

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
10%，而是允许每根伸缩杆在当前长度基础上最多收短 10%。原先 min 等于
current 时只能伸长，很多视频动作需要局部收缩，因此会大量落入 marginal
甚至 unreachable。

实验比较 0.05/0.10/0.15 三种运动尺度、0/10% 收缩、bbox/motion-informed
rig，以及固定拓扑 morphology 优化。合格条件是 272 帧、unreachable 为零、
最大六控制点投影误差不超过 0.05 m、VGT 数组有限且拓扑为 12/30。

### 6. VGT 序列和动画

QianJi 的 `feasible_site_targets.npz` 被严格校验并重新打包为
`vgt_motion.npz`。`272 x 12 x 3` 表示：

- 272 个视频时刻；
- 每个时刻 12 个 VGT 结构节点；
- 每个节点一个 `(x,y,z)` 位置。

30 根杆是 robot 中固定的 30 对节点连接。动画每帧用当帧 12 个位置画出相同
的 30 条连接，输出 top XY、side XZ、front YZ、isometric 四个视图。

## 真实结果

验收案例位于：

```text
outputs/experiments/39point_vgt_cat_v3
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
| 最大杆长违反 | 0.0002825 m |
| VGT positions | `(272, 12, 3)` |
| 动画 | 1246x720、30 FPS、272 帧 |

这比 bbox、无收缩的 `scale=0.10` 明显改善：后者为 2 feasible、259
marginal、11 unreachable，最大误差 0.05271 m。motion-informed + 10% 收缩
把不可达帧降到零，并让 266 帧进入严格 feasible；还剩 6 帧 marginal，说明
几何可达性已满足本项目门槛，但并不等于动力学步态有效。

Desired 控制目标与 QianJi projected 控制目标是不同文件和不同哈希。最终
`vgt_motion.npz` SHA-256 为
`22313aabfaa343980e3a71350219bbee1c00578d3d78848ad6ef7a0d492a5177`。

## 一键运行

```bash
uv pip install --python .venv/bin/python -e .

ANIMAL_DATA_ROOT="/absolute/qianji-animal-motion" \
QIANJI_ROOT="/absolute/QianJi" \
OUTPUT_ROOT="$PWD/outputs/experiments/39point_vgt_cat_v4" \
bash experiments/39point_vgt_cat/run_experiment.sh
```

输出目录必须不存在。脚本对 QianJi 只读，所有收缩 robot 和 morphology
robot 都写入实验目录。最后运行 `verify_case.py`，任何验收项失败都会以非零
状态退出。

## 完成标准

- 39 个二维角色和 39 个 2.5D 角色逐帧齐全；
- 角色无缺失，所有 JSON/NPZ 有限，不跨帧插值或平滑；
- robot 为 12 site、30 rod，端点引用有效；
- VGT NPZ 为 `(272,12,3)`，时间严格递增且对应 30 FPS；
- desired/projected 路径与哈希不同；
- unreachable 为零，最大控制点误差不超过 5 cm；
- 2D/VGT 视频均为 30 FPS、272 帧且非空；
- 输入、输出、robot、rig 和控制轨迹有 SHA-256；
- 明确不主张度量深度、相机标定、全局位移、动力学或 Mesh 顶点形变。
