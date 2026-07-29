# 小猫二维到 QianJi 四足三维实验

## 目标

本实验验证当前小猫案例的完整链路：

```text
校正后的 272 帧六点二维轨迹
  + 当前小猫 GLB
  -> Mesh 派生的 12 节点、30 杆 QianJi VGT
  -> quadruped_bbox 六点 rig
  -> 身体相对 2.5D 三维关键点运动
  -> QianJi 三视图、视频和几何可达性
```

这里只解决当前六点四足合同。非四足、完整骨架、Mesh 蒙皮或生物学准确的
关节运动不在本实验范围内。

## 方法

参考帧使用人工身份锚点 `152`。每帧在二维躯干局部坐标中归一化，因而消除
画面平移、平面内旋转和统一缩放。只把相对于参考帧的纵向和竖直变化映射到
Mesh rig；中性 lateral 坐标保持不变。

这称为 `body_relative_2_5d_retarget`，不是单目三维重建。相机未标定，深度
未观测，全局位移被移除。

## 当前小猫结果

输入 GLB 包含 2,154 个顶点和 4,304 个三角面，并被 QianJi 判定为 watertight。
`abstract` preset 生成 12 个 sites、30 个 rods，刚性矩阵 rank 为 30，缺失
端点为 0。272 帧都被 QianJi 预览器成功加载并生成约 9.03 秒的视频。

| motion scale | feasible | marginal | unreachable | max error (m) | mean error (m) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0 | 272 | 0 | 0.0440 | 0.0090 |
| 0.25 | 0 | 166 | 106 | 0.1089 | 0.0216 |
| 0.50 | 0 | 93 | 179 | 0.2132 | 0.0457 |

当前示例选择 `0.10`，因为它是三个候选中唯一没有 unreachable 帧的尺度。
所有帧仍是 marginal，而不是 feasible；主要限制来自自动 rig 的初始几何与
extension-only 杆长约束。因此结论是“链路跑通且目标可投影”，不是“原始
目标被机械结构无误差复现”。

轨迹中剩余 9 个无效关键点均按合同替换为对应的中性 site 和置信度 `0.0`，
没有跨帧插值。

## 复现

仓库中的 `experiments/2d_to_3d_cat/run_experiment.sh` 接受：

```bash
ANIMAL_DATA_ROOT=/path/to/qianji-animal-motion \
QIANJI_ROOT=/path/to/QianJi \
OUTPUT_ROOT=/path/to/new/output \
bash experiments/2d_to_3d_cat/run_experiment.sh
```

脚本要求 `biomimic` Mamba 环境和本项目 Python 3.12 环境，拒绝使用已有的
`OUTPUT_ROOT`，依次生成 morphology、rig、三个尺度、预览、可达性和汇总。

生成产物默认位于被 Git 忽略的 `outputs/experiments/2d_to_3d_cat/`。其中
`experiment_summary.json` 是机器可读证据，`experiment_summary.md` 是简表。

## 限制

- 相机侧向仍是 `unknown`，左右身份依赖现有人工确认。
- 没有相机内外参、真实尺度或深度监督。
- 六点不足以恢复膝、髋、肩、脊柱关节或完整表面形变。
- reachability 是几何优化检查，不是 MuJoCo 动力学步态稳定性证明。
- 若要增大运动幅度，应先优化 morphology/rig 或允许明确的收缩模型，不能
  仅通过增大 `motion_scale` 强推目标。
