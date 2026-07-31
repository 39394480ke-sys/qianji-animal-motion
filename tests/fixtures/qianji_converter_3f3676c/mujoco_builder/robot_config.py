# -*- coding: utf-8 -*-
"""
机器人配置文件 - 单一真理源 (Single Source of Truth)

本文件集中管理所有机器人仿真参数，包括：
- 几何参数（尺寸、形状）
- 物理参数（质量、惯量、摩擦）
- 求解器参数（时间步长、迭代次数）
- 关节/控制参数（增益、阻尼）

使用方法：
    from robot_config import DEFAULT_CONFIG
    
    # 访问参数
    ball_dist = DEFAULT_CONFIG.geometry.ball_dist
    kp = DEFAULT_CONFIG.joint_control.kp
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# 1. 锚点配置 (Anchor)
# =============================================================================

@dataclass
class AnchorConfig:
    """
    锚点（节点球）配置
    
    锚点是杆组之间的连接点，包含一个带 free 关节的球体。
    """
    # --- 几何参数 ---
    radius: float = 0.025
    """锚点球体半径 (m)，用于碰撞检测和可视化"""
    
    # --- 物理参数 ---
    mass: float = 0.05
    """锚点质量 (kg)，影响惯性和动力学响应"""
    
    armature: float = 0.1
    """
    虚拟惯量/电枢 (kg·m²)
    增加关节的有效惯量，提高数值稳定性
    """
    
    damping: float = 0.1
    """
    空气阻尼 (N·s/m)
    模拟空气阻力，防止自由运动时的数值发散
    """


# =============================================================================
# 2. 球铰配置 (Ball Joint)
# =============================================================================

@dataclass
class BallJointConfig:
    """
    球铰（球头关节）配置
    
    球铰连接杆端与锚点，允许 3 自由度旋转。
    """
    armature: float = 0.1
    """
    电枢/虚拟惯量 (kg·m²)
    增加关节的有效转动惯量，改善高速运动时的稳定性
    """
    
    frictionloss: float = 0.17
    """
    摩擦损耗 (N·m)
    模拟关节内部的恒定摩擦力矩，与角速度无关
    """
    
    damping: float = 0.05
    """
    阻尼系数 (N·m·s/rad)
    与角速度成正比的阻尼力矩，抑制高频振动
    """
    
    rot_limit_deg: float = 41
    """
    旋转限制角度 (deg)
    球铰允许的最大偏转角度，<=0 表示禁用限制
    """


# =============================================================================
# 3. 杆组配置 (Rod Group)
# =============================================================================

@dataclass
class RodGroupConfig:
    """
    杆组配置
    
    每个杆组包含：套筒(sleeve) + 左杆(left) + 右杆(right)
    """
    # --- 几何参数 ---
    rod_radius: float = 0.0125
    """杆的半径 (m)，用于碰撞和可视化"""
    
    rod_trim: float = 0.01
    """
    杆端修剪量 (m)
    将杆几何的两端各缩短此值，避免与端点球/套筒直接顶碰
    """
    
    sleeve_half: List[float] = field(default_factory=lambda: [0.0175, 0.0175, 0.135])
    """
    套筒半尺寸 [rx, ry, hz] (m)
    - rx: X 方向半径
    - ry: Y 方向半径  
    - hz: Z 方向半长（沿杆轴向）
    """
    
    # 注意：sleeve_rgba 已移至 VisualConfig
    
    # --- 物理参数 ---
    group_mass: float = 0.33
    """杆组总质量 (kg)，会按策略分配给左杆、套筒、右杆"""
    
    split_strategy: str = "manual"
    """
    质量分配策略:
    - "manual": 使用 mass_split 手动指定比例
    - "length": 按杆长度比例分配
    - "volume": 按体积比例分配
    """
    
    mass_split: List[float] = field(default_factory=lambda: [0.2, 0.6, 0.2])
    """
    质量分配比例 [左杆, 套筒, 右杆]
    仅当 split_strategy="manual" 时生效，比例会自动归一化
    """
    
    # --- 杆起始偏移 ---
    rod_start_offset_left: float = 0.0
    """左杆起始偏移 (m)，从套筒中心向左偏移的初始距离"""
    
    rod_start_offset_right: float = 0.0
    """右杆起始偏移 (m)，从套筒中心向右偏移的初始距离"""


# =============================================================================
# 4. 关节/控制默认值 (Joint/Control)
# =============================================================================

@dataclass
class JointControlConfig:
    """
    关节和控制器默认配置
    
    主要用于滑动关节(slide joint)的 PD 控制器参数。
    """
    # --- PD 控制器参数 ---
    kp: float = 50000.0
    """
    位置增益 (N/m)
    PD 控制器的比例系数，决定位置误差产生的力
    """
    
    kv: float = 200.0
    """
    速度增益 (N·s/m)
    PD 控制器的微分系数，决定速度误差产生的阻尼力
    """
    
    # --- 滑动关节参数 ---
    slide_damping: float = 0.05
    """
    滑动关节阻尼 (N·s/m)
    与滑动速度成正比的阻尼力
    """
    
    slide_frictionloss: float = 0.2
    """
    滑动关节摩擦损耗 (N)
    恒定的滑动摩擦力，与速度无关
    """
    
    slide_armature: float = 0.2
    """
    滑动关节电枢 (kg)
    增加关节的有效惯量，提高数值稳定性
    """


# =============================================================================
# 5. 轮胎配置 (Tire)
# =============================================================================

@dataclass
class TireConfig:
    """
    轮胎配置
    
    轮胎安装在杆端的 Anchor Port 上，用于移动和接触地面。
    """
    # --- 几何参数（会写入 JSON，影响结构） ---
    distance: float = 0.36
    """
    [已废弃] 旧版安装距离参数 (m)
    
    此参数已被 axle_offset 替代，保留仅用于向后兼容。
    在早期版本中，此参数用于表示轮胎中心到杆端点的距离，
    但由于轮胎实际挂载在节点上，该参数定义存在歧义。
    
    新代码请使用 axle_offset 参数。
    """

    mount_length: float = 0.055
    """
    固定杆长度 (m)

    从节点中心到固定杆末端沿 `mount_dir` 的刚性安装长度。
    该长度用于显式建模轮式模组的固定支杆，而不再用纯粹的向下偏移近似。
    """

    axle_offset: float = 0.035
    """
    轮轴外偏量 (m)

    从固定杆末端到轮心沿轮轴方向的额外外偏距离。
    最终轮心位置为：

        节点中心 + mount_dir * mount_length + wheel_axis * axle_offset
    """

    mount_outward_weight: float = 1.0
    """自动生成 mount_dir 时，向外分量的权重。"""

    mount_downward_weight: float = 0.75
    """自动生成 mount_dir 时，向下分量的权重。"""

    mount_forward_weight: float = 0.5
    """自动生成 mount_dir 时，向前分量的权重。"""

    mount_radius: float = 0.008
    """固定杆/支架的碰撞半径 (m)。"""

    mount_mass: float = 0.0
    """
    固定杆/支架总质量 (kg)。

    默认不额外计入质量，避免与 wheel mass 中已包含的轮式模组质量重复计算。
    若后续确认轮胎质量仅表示轮圈/轮胎本体，可再单独补上该值。
    """

    mount_rgba: List[float] = field(default_factory=lambda: [0.82, 0.52, 0.72, 1.0])
    """固定杆/支架的可视化颜色 [R, G, B, A]。"""

    diameter: float = 0.185
    """轮胎直径 (m)"""
    
    thickness: float = 0.07
    """轮胎厚度/宽度 (m)"""


    # --- 物理参数（从 robot_config 读取，不写入 JSON） ---
    mass: float = 0.588
    """轮胎质量 (kg)"""
    
    friction: List[float] = field(default_factory=lambda: [1.0, 0.002, 0.01])
    """
    轮胎与地面的摩擦系数 [滑动, 自旋, 滚动]
    - [0] 滑动摩擦 (无量纲): 主要影响抓地力
    - [1] 自旋摩擦 (m): 原地转向阻力
    - [2] 滚动摩擦 (m): 滚动阻力系数
    """
    
    rgba: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 1.0])
    """轮胎颜色 [R, G, B, A]"""
    
    # --- 关节参数 ---
    joint_damping: float = 0
    """轮胎转动关节阻尼 (N·m·s/rad)"""
    
    joint_frictionloss: float = 0
    """轮胎转动关节摩擦损耗 (N·m)"""
    
    armature: float = 0.005
    """轮胎转动关节电枢 (kg·m²)"""
    
    # --- 速度执行器参数（Velocity Actuator）---
    # 轮子采用速度控制，分为高速档和低速档：
    # - 高速档：速度 38.525 rpm (4.034 rad/s)，最大扭矩 5.0 N·m
    # - 低速档：速度 12.93 rpm (1.354 rad/s)，最大扭矩 10 N·m（默认）
    
    velocity_kv: float = 20.0
    """
    速度增益 (N·m·s/rad)
    velocity 执行器的速度增益参数
    """
    
    velocity_forcerange: List[float] = field(default_factory=lambda: [-5.0, 5.0])
    """
    力矩范围 [最小值, 最大值] (N·m)
    默认使用低速档的最大扭矩：±10 N·m
    高速档可设置为 [-5.0, 5.0]
    """
    
    velocity_ctrlrange: List[float] = field(default_factory=lambda: [-4.034, 4.034])
    """
    速度控制范围 [最小值, 最大值] (rad/s)
    默认使用低速档的速度范围：±1.354 rad/s (±12.93 rpm)
    高速档可设置为 [-4.034, 4.034] (±38.525 rpm)
    """
    
    velocity_gear: float = 1.0
    """
    传动比（无量纲）
    velocity 执行器的齿轮传动比
    """

# =============================================================================
# 6. 物理/求解器配置 (Physics/Solver)
# =============================================================================

@dataclass
class PhysicsSolverConfig:
    """
    MuJoCo 物理引擎和求解器配置
    """
    timestep: float = 0.001
    """
    仿真时间步长 (s)
    较小的步长提高精度但降低速度，建议 0.0005-0.002
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    重力加速度 (m/s²)
    默认 Z 轴向下 -9.81，无重力环境设为 (0, 0, 0)
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    iterations: int = 80
    """
    约束求解器主迭代次数
    更多迭代提高约束精度，但增加计算时间
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    ls_iterations: int = 50
    """
    线搜索迭代次数
    Newton 求解器的线搜索子迭代
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    tolerance: float = 1e-10
    """
    求解器收敛容差
    残差低于此值时认为收敛
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    njmax: int = 4000
    """
    最大约束数量
    需根据模型复杂度调整，过小会导致仿真失败
    注：该值实际并未启用，而是统一在scene中设置
    """
    
    nconmax: int = 4000
    """
    最大接触点数量
    需根据碰撞复杂度调整
    注：该值实际并未启用，而是统一在scene中设置
    """


# =============================================================================
# 7. 等式约束配置 (Equality Constraints)
# =============================================================================

@dataclass
class EqualityConstraintConfig:
    """
    等式约束（weld/connect）的软硬度配置
    
    控制约束的柔性和响应特性。
    """
    solref: Tuple[float, float] = (0.015, 1.0)
    """
    约束求解参考参数 [时间常数, 阻尼比]
    - [0] 时间常数 (s): 约束恢复的时间尺度，越小越硬
    - [1] 阻尼比 (无量纲): 1.0 为临界阻尼，<1 欠阻尼，>1 过阻尼
    """
    
    solimp: Tuple[float, float, float] = (0.99, 0.999, 0.001)
    """
    约束求解阻抗参数 [dmin, dmax, width]
    - [0] dmin: 最小阻尼系数
    - [1] dmax: 最大阻尼系数
    - [2] width: 过渡区宽度 (m)
    """


# =============================================================================
# 8. 几何/结构参数 (Geometry/Structure)
# =============================================================================

@dataclass
class GeometryConfig:
    """
    几何和结构参数
    
    这些参数影响机器人的整体尺寸和形状，
    主要由 structure_generator.py 使用。
    """
    ball_dist: float = 0.34261
    """
    球副间距 (m)
    两个球铰之间的距离，即杆的有效旋转长度
    """
    
    stretch_ratio: float =  1.46
    """
    伸缩比 (无量纲)
    最大长度 / 最小长度，决定滑动关节的行程范围
    """
    
    node_offset: float = 0.02409
    """
    节点偏移 (m)
    节点中心到球副中心的距离，也称为 port_offset
    """
    
    site_radius: float = 0.001
    """site 可视化半径 (m)，用于调试显示"""
    
    @property
    def init_length(self) -> float:
        """
        计算初始边长/节点间距 (m)
        init_length = ball_dist + 2 * node_offset
        """
        return self.ball_dist + 2 * self.node_offset
    
    @property
    def slide_range(self) -> float:
        """
        计算滑动关节单侧行程 (m)
        slide_range = ball_dist * (stretch_ratio - 1) / 2
        """
        return self.ball_dist * (self.stretch_ratio - 1) / 2


# =============================================================================
# 9. 可视化参数 (Visual)
# =============================================================================

@dataclass
class VisualConfig:
    """
    可视化和辅助几何参数
    
    集中管理所有与可视化相关的颜色和外观参数。
    """
    root_stub_mass: float = 1e-5
    """
    根桩极小质量 (kg)
    附加在杆组根 body 上的极小质量球，确保惯量合法
    """
    
    # --- 锚点可视化 ---
    anchor_geom_rgba: List[float] = field(default_factory=lambda: [0.3, 0.6, 1.0, 0.5])
    """锚点碰撞球体颜色 [R, G, B, A]"""
    
    anchor_port_rgba: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.5, 0.5])
    """锚点 Port 颜色 [R, G, B, A]"""
    
    # --- 杆组可视化 ---
    sleeve_rgba: List[float] = field(default_factory=lambda: [0.20, 0.65, 0.90, 0.7])
    """
    套筒颜色 [R, G, B, A]
    - R, G, B: 颜色分量 (0.0-1.0)
    - A: 透明度 (0.0=全透明, 1.0=不透明)
    """
    
    enable_sleeve_mesh: bool = False
    """是否启用套筒的视觉几何（mesh），如果启用则使用 sleeve_mesh_rgba 作为 mesh 的颜色"""
    meshdir: str = "../assets/meshes"   #或者"../meshes"
    """MuJoCo 编译器引用的 meshdir 路径 (用于 <compiler meshdir='...'>)。相对于 scene 文件所在目录"""
    sleeve_mesh_file: str = "sleeve_visual.stl"
    """Mesh 文件名，相对于 meshdir 指定的目录"""
    sleeve_mesh_asset_name: str = "visual_sleeve_mesh_asset"
    """MuJoCo Asset 中引用的 Mesh 名称 (用于 <asset><mesh name='...'/>)"""
    sleeve_mesh_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    """Mesh 缩放比例 [X, Y, Z]"""
    sleeve_mesh_rgba: List[float] = field(default_factory=lambda: [0.671705, 0.692426, 0.774270, 1.0])
    """Mesh 颜色 [R, G, B, A]"""
    
    rod_rgba: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    """杆体颜色 [R, G, B, A]"""
    
    tip_rgba: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.5])
    """杆端 tip 颜色 [R, G, B, A]"""


# =============================================================================
# 10. 传感器配置 (Sensors)
# =============================================================================

@dataclass
class SensorConfig:
    """
    传感器开关配置
    """
    add_imu: bool = True
    """是否在锚点添加 IMU（加速度计 + 陀螺仪）"""
    
    add_jointaxial: bool = True
    """是否添加关节轴向力传感器 (jointactuatorfrc)"""
    
    add_tip_force: bool = True
    """是否添加端点力/力矩传感器 (force/torque)"""


# =============================================================================
# 11. 完整配置类 (RobotConfig)
# =============================================================================

@dataclass
class RobotConfig:
    """
    机器人完整配置 - 单一真理源 (Single Source of Truth)
    
    聚合所有子配置，提供统一的配置访问接口。
    
    使用示例:
        config = RobotConfig()
        
        # 访问子配置
        config.anchor.mass
        config.geometry.ball_dist
        config.physics.timestep
        
        # 修改配置
        config.physics.gravity = (0, 0, 0)  # 无重力
    """
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    """锚点配置"""
    
    ball_joint: BallJointConfig = field(default_factory=BallJointConfig)
    """球铰配置"""
    
    rod_group: RodGroupConfig = field(default_factory=RodGroupConfig)
    """杆组配置"""
    
    joint_control: JointControlConfig = field(default_factory=JointControlConfig)
    """关节/控制配置"""
    
    tire: TireConfig = field(default_factory=TireConfig)
    """轮胎配置"""
    
    physics: PhysicsSolverConfig = field(default_factory=PhysicsSolverConfig)
    """物理/求解器配置"""
    
    equality: EqualityConstraintConfig = field(default_factory=EqualityConstraintConfig)
    """等式约束配置"""
    
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    """几何/结构配置"""
    
    visual: VisualConfig = field(default_factory=VisualConfig)
    """可视化配置"""
    
    sensor: SensorConfig = field(default_factory=SensorConfig)
    """传感器配置"""


# =============================================================================
# 12. 全局默认配置实例
# =============================================================================

DEFAULT_CONFIG = RobotConfig()
"""
全局默认配置实例

其他模块应该导入并使用此实例：
    from robot_config import DEFAULT_CONFIG
"""


# =============================================================================
# 13. 辅助函数
# =============================================================================

def load_config_from_dict(config_dict: dict) -> RobotConfig:
    """
    从字典加载配置（用于从 JSON/YAML 加载）
    
    Args:
        config_dict: 配置字典，结构应与 RobotConfig 匹配
        
    Returns:
        RobotConfig 实例
    """
    config = RobotConfig()
    
    # 遍历顶层键
    for key, value in config_dict.items():
        if hasattr(config, key) and isinstance(value, dict):
            sub_config = getattr(config, key)
            for sub_key, sub_value in value.items():
                if hasattr(sub_config, sub_key):
                    setattr(sub_config, sub_key, sub_value)
    
    return config


def config_to_dict(config: RobotConfig) -> dict:
    """
    将配置转换为字典（用于序列化为 JSON/YAML）
    
    Args:
        config: RobotConfig 实例
        
    Returns:
        配置字典
    """
    from dataclasses import asdict
    return asdict(config)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 打印默认配置
    print("=== 机器人默认配置 ===\n")
    
    print(f"[几何参数]")
    print(f"  球副间距: {DEFAULT_CONFIG.geometry.ball_dist} m")
    print(f"  伸缩比: {DEFAULT_CONFIG.geometry.stretch_ratio}")
    print(f"  节点偏移: {DEFAULT_CONFIG.geometry.node_offset} m")
    print(f"  初始边长: {DEFAULT_CONFIG.geometry.init_length} m")
    print(f"  滑动行程: ±{DEFAULT_CONFIG.geometry.slide_range} m")
    
    print(f"\n[锚点参数]")
    print(f"  质量: {DEFAULT_CONFIG.anchor.mass} kg")
    print(f"  半径: {DEFAULT_CONFIG.anchor.radius} m")
    
    print(f"\n[杆组参数]")
    print(f"  杆半径: {DEFAULT_CONFIG.rod_group.rod_radius} m")
    print(f"  总质量: {DEFAULT_CONFIG.rod_group.group_mass} kg")
    
    print(f"\n[控制参数]")
    print(f"  Kp: {DEFAULT_CONFIG.joint_control.kp} N/m")
    print(f"  Kv: {DEFAULT_CONFIG.joint_control.kv} N·s/m")
    
    print(f"\n[物理参数]")
    print(f"  时间步长: {DEFAULT_CONFIG.physics.timestep} s")
    print(f"  重力: {DEFAULT_CONFIG.physics.gravity} m/s²")
