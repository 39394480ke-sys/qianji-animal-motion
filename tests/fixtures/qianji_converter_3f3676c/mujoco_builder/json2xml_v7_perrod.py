# -*- coding: utf-8 -*-
"""
JSON -> MuJoCo XML 生成器（双P副：套筒 + 左/右杆）

设计要点（阅读必看）：
----------------------------------------------------------------------
1) 模型结构（每个杆组）：
   - 在 site1 的世界坐标处创建 root body，并赋予 free 关节（6 自由度）。
   - 在 site1 与 site2 的中点处放置套筒（sleeve），其局部 +Z 轴对齐向量 v=(site2-site1)。
   - 在套筒下面挂两个子 body：left 与 right，分别各带一个 slide 关节。
     * right：slide 轴沿局部 +Z；其端点 site 与全局锚点 site2 相连（equality connect）。
     * left ：slide 轴沿局部 -Z（或 +Z 但几何朝负向）；其端点 site 与全局锚点 site1 相连。
   - 组内排碰：left/right/sleeve 互相 exclude，避免自碰导致数值震荡。

2) 全局锚点（sites）：
   - 每个 JSON.sites[...] 都会在 worldbody 下生成一个独立的“锚点体”（有 free 关节）。
   - 锚点体包含一个不可碰撞的可视球 geom + 一个 site（用于与杆端点 connect）。
   - 可选在锚点 site 上挂 IMU 传感器（accelerometer/gyro）。

3) 传感器（可选）：
   - jointactuatorfrc：测量“执行器通过传动施加在关节轴上的力”（标量）。
   - force/torque：测量“端点 site 上的三维力/力矩向量”（在 site 坐标系下）。

4) 质量与惯量：
   - 支持 group_mass（整体质量）+ 分配策略（length/volume/manual），并允许单体 mass/density 覆盖。
   - root body 附一个极小质量的透明小球（不参与碰撞），确保“moving bodies 质量/惯量 > mjMINVAL”。

5) 数值稳定：
   - 端点修剪（rod_trim）：将杆几何的 fromto 两端缩短一点，避免与端点/套筒几何直接顶碰。
   - equality connect 的 solref/solimp 默认较“柔”，既稳又不太软，可按需通过 CLI 调整。
   - compiler inertiafromgeom="true"：由几何估算惯量（常用设置）。

坐标系与命名约定：
----------------------------------------------------------------------
- 全局坐标：MuJoCo 世界坐标。
- 杆组局部：将套筒 body 的 +Z 轴对齐到 v=(site2-site1)，左右杆沿该轴滑动/伸缩。
- 命名：
  anchor_<sid>        : 某锚点 body（worldbody 的子节点）
  anchor_<sid>_site   : 锚点上的 site（用于 equality connect）
  <name>__root        : 杆组根 body（位于 site1）
  <name>__sleeve      : 套筒 body（位于中点，朝向 v）
  <name>__left/right  : 左/右杆 body
  <name>__slide_<...> : 左/右 slide 关节
  act_<joint_name>    : 对应关节的 position 执行器
"""

import argparse
import json
import math
import sys
from xml.etree.ElementTree import Element, SubElement, ElementTree

# 导入统一配置 - 单一真理源
from robot_config import DEFAULT_CONFIG


# ----------------------- 工具函数 -----------------------
def normalize(v):
    """
    将 3D 向量归一化。若长度极小，回退到 (0,0,1)。
    参数:
        v: tuple[float, float, float]
    返回:
        tuple[float, float, float]  归一化后的向量
    """
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return (v[0]/n, v[1]/n, v[2]/n) if n > 1e-12 else (0.0, 0.0, 1.0)


def vec_sub(a, b):
    """向量减法 a - b（逐元素）。"""
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def vec_scale(a, s):
    """向量标量乘 a * s。"""
    return (a[0]*s, a[1]*s, a[2]*s)

def quat_mul(q1, q2):
    """四元数乘法 q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

def quat_inv(q):
    """四元数求逆 (假设已归一化)"""
    return (q[0], -q[1], -q[2], -q[3])

def quat_rotate_vec(v, q):
    """用四元数 q 旋转向量 v"""
    # p = (0, v)
    # p' = q * p * q_inv
    # return p'.vec
    
    qw, qx, qy, qz = q
    vx, vy, vz = v
    
    # q * p
    # w = -x*vx - y*vy - z*vz
    # x = w*vx + y*vz - z*vy
    # y = w*vy + z*vx - x*vz
    # z = w*vz + x*vy - y*vx
    
    pw = -qx*vx - qy*vy - qz*vz
    px = qw*vx + qy*vz - qz*vy
    py = qw*vy + qz*vx - qx*vz
    pz = qw*vz + qx*vy - qy*vx
    
    # p' = (q * p) * q_inv
    # q_inv = (qw, -qx, -qy, -qz)
    
    # w' = pw*qw - px*-qx - py*-qy - pz*-qz (should be 0)
    # x' = pw*-qx + px*qw + py*-qz - pz*-qy
    # y' = pw*-qy + px*qz + py*qw - pz*-qx
    # z' = pw*-qz + px*-qy - py*qx + pz*qw
    
    rx = -pw*qx + px*qw - py*qz + pz*qy
    ry = -pw*qy + px*qz + py*qw - pz*qx
    rz = -pw*qz - px*qy + py*qx + pz*qw
    
    return (rx, ry, rz)

def cross_product(a, b):
    """向量叉乘 a x b"""
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def deg_to_rad(deg):
    """角度转弧度"""
    return deg * math.pi / 180.0

def quat_from_two_vecs_z_to_u(u):
    """
    计算四元数 q，使“全局 Z 轴 (0,0,1)”旋转到“目标单位向量 u”的方向。
    用于将套筒局部 +Z 对齐到 (site2 - site1)。
    返回格式为 (w, x, y, z)。
    """
    uz = (0, 0, 1)
    u = normalize(u)
    dot = uz[0]*u[0] + uz[1]*u[1] + uz[2]*u[2]

    # 180° 与 0° 特殊情况避免数值不稳定
    if dot < -0.999999:
        return (0, 1, 0, 0)   # 180deg about X
    if dot > 0.999999:
        return (1, 0, 0, 0)   # identity

    # 一般情况：用叉乘近似旋转轴
    cx, cy, cz = -u[1], u[0], 0.0      # (0,0,1) x u
    s = math.sqrt((1.0 + dot) * 2.0)
    qw = s * 0.5
    qx, qy, qz = cx / s, cy / s, cz / s

    # 再正则化一次，保险
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return (qw/n, qx/n, qy/n, qz/n)


def trim_segment(a, b, eps):
    """
    将杆几何 fromto 段的两端各缩短 eps，防止端点与球/套筒直接顶到（更稳）。
    参数:
        a, b : tuple[float, float, float]  段端点（局部坐标）
        eps  : float  修剪量（会被 clamp 到 0.45*段长以内）
    返回:
        (a2, b2) 修剪后的新端点
    """
    v = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    L = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if L < 1e-9:
        return a, b
    u = (v[0]/L, v[1]/L, v[2]/L)
    eps = min(eps, 0.45 * L)
    a2 = (a[0] + u[0]*eps, a[1] + u[1]*eps, a[2] + u[2]*eps)
    b2 = (b[0] - u[0]*eps, b[1] - u[1]*eps, b[2] - u[2]*eps)
    return a2, b2


def ensure_child(root, tag):
    """
    若 root 下不存在 tag 对应的子节点，则创建一个。
    常用于确保 <sensor> 容器存在。
    """
    child = root.find(tag)
    if child is None:
        child = SubElement(root, tag)
    return child

def port_body_name(rod_name, side):
    return f"port_{rod_name}_{side}"


def canonicalize_axis_direction(axis):
    """
    统一关节轴的符号，避免左右轮因为轴向取反而导致同号控制量含义不一致。
    这只影响“正方向”的定义，不改变物理转轴所在直线。
    """
    ax = normalize(axis)
    if (
        ax[0] < -1e-6
        or (abs(ax[0]) < 1e-6 and ax[1] < -1e-6)
        or (abs(ax[0]) < 1e-6 and abs(ax[1]) < 1e-6 and ax[2] < -1e-6)
    ):
        return (-ax[0], -ax[1], -ax[2])
    return ax


def resolve_tire_mount(tire_cfg, opts):
    """
    解析轮胎的安装几何。

    新版：
      - mount_dir: 固定杆方向
      - wheel_axis: 轮轴 / hinge 方向
      - mount_length: 节点到固定杆末端的长度
      - axle_offset: 固定杆末端到轮心沿轮轴方向的外偏

      轮心位置：
        mount_dir * mount_length + wheel_axis * axle_offset

    兼容旧版：
      - wheel_center = wheel_pos_axis * distance + drop_axis_local * drop_offset
    """
    mount_length = float(tire_cfg.get("mount_length", opts.tire_mount_length))
    axle_offset = float(tire_cfg.get("axle_offset", opts.tire_axle_offset))
    mount_dir_raw = tire_cfg.get("mount_dir")
    wheel_axis_raw = tire_cfg.get(
        "wheel_axis",
        tire_cfg.get("wheel_joint_axis", tire_cfg.get("wheel_pos_axis")),
    )

    has_new_mount_model = (
        mount_dir_raw is not None
        or "mount_length" in tire_cfg
        or "axle_offset" in tire_cfg
        or "wheel_axis" in tire_cfg
        or "wheel_joint_axis" in tire_cfg
    )

    if has_new_mount_model:
        if mount_dir_raw is None:
            mount_dir_raw = wheel_axis_raw if wheel_axis_raw is not None else (1.0, 0.0, 0.0)
        mount_dir = normalize(tuple(mount_dir_raw))

        if wheel_axis_raw is None:
            wheel_axis_raw = mount_dir
        wheel_axis = normalize(tuple(wheel_axis_raw))

        mount_end_offset = (
            mount_dir[0] * mount_length,
            mount_dir[1] * mount_length,
            mount_dir[2] * mount_length,
        )
        wheel_center_offset = (
            mount_end_offset[0] + wheel_axis[0] * axle_offset,
            mount_end_offset[1] + wheel_axis[1] * axle_offset,
            mount_end_offset[2] + wheel_axis[2] * axle_offset,
        )
        return mount_dir, wheel_axis, mount_end_offset, wheel_center_offset

    pos_axis = normalize(tuple(tire_cfg.get("wheel_pos_axis", tire_cfg.get("wheel_axis", (1.0, 0.0, 0.0)))))
    drop_axis = normalize(tuple(tire_cfg.get("drop_axis_local", (0.0, 0.0, -1.0))))
    drop_offset = float(tire_cfg.get("drop_offset", opts.tire_drop_offset))
    distance = float(tire_cfg.get("distance", opts.tire_axle_offset))

    wheel_axis = normalize(tuple(tire_cfg.get("wheel_axis", tire_cfg.get("wheel_joint_axis", pos_axis))))
    wheel_center_offset = (
        pos_axis[0] * distance + drop_axis[0] * drop_offset,
        pos_axis[1] * distance + drop_axis[1] * drop_offset,
        pos_axis[2] * distance + drop_axis[2] * drop_offset,
    )

    mount_norm = math.sqrt(
        wheel_center_offset[0] * wheel_center_offset[0]
        + wheel_center_offset[1] * wheel_center_offset[1]
        + wheel_center_offset[2] * wheel_center_offset[2]
    )
    if mount_norm > 1e-9:
        mount_dir = normalize(wheel_center_offset)
    else:
        mount_dir = pos_axis

    return mount_dir, wheel_axis, (0.0, 0.0, 0.0), wheel_center_offset

def add_tire(parent_body, tire_cfg, rod_name, end_side, opts):
    """
    生成轮胎组件（挂载在 Anchor Body 上）：
      - 固定杆/支架几何（沿 mount_dir）
      - 轮胎 body
      - 转动关节（hinge，轴线与真实轮轴方向一致）
      - 圆盘几何（cylinder，轴线与轮轴一致）
      - velocity 驱动

    参数:
        parent_body : 父body（Anchor Body）
        tire_cfg    : 轮胎配置字典
        rod_name    : 杆组名称
        end_side    : 端点类型（"left" 或 "right"）
        opts        : 全局配置选项
    返回:
        tire_joint_name : 轮胎关节名称
    """
    # 轮胎基础配置（从 robot_config 读取默认值）
    tire_name = tire_cfg.get("name", f"{rod_name}__tire_{end_side}")

    diameter = float(tire_cfg.get("diameter", opts.tire_diameter))         # 轮胎直径（米）
    thickness = float(tire_cfg.get("thickness", opts.tire_thickness))      # 轮胎厚度（米）
    mass = float(tire_cfg.get("mass", opts.tire_mass))                     # 轮胎质量（kg）
    rgba = tire_cfg.get("rgba", opts.tire_rgba)                            # 轮胎颜色
    mount_dir, wheel_axis, mount_end_offset, wheel_center_offset = resolve_tire_mount(tire_cfg, opts)

    # 关节配置
    tire_damp = float(tire_cfg.get("tire_damping", opts.tire_damping))
    tire_fric = float(tire_cfg.get("tire_friction_loss", opts.tire_joint_frictionloss))
    tire_arm  = float(tire_cfg.get("joint_armature", opts.tire_armature))
    
    # 摩擦配置（轮胎与地面的摩擦系数）
    friction = tire_cfg.get("friction", opts.tire_friction)                # 滑动摩擦、滚动摩擦、自旋摩擦

    # 固定杆/支架：不需要单独关节，它的质量和碰撞直接并入 anchor 自由体
    mount_body = SubElement(parent_body, "body", name=f"{tire_name}__mount")
    mount_end_norm = math.sqrt(
        mount_end_offset[0] * mount_end_offset[0]
        + mount_end_offset[1] * mount_end_offset[1]
        + mount_end_offset[2] * mount_end_offset[2]
    )
    if mount_end_norm > 1e-9:
        SubElement(
            mount_body,
            "geom",
            type="capsule",
            fromto=f"0 0 0 {mount_end_offset[0]} {mount_end_offset[1]} {mount_end_offset[2]}",
            size=str(opts.tire_mount_radius),
            mass=str(opts.tire_mount_mass),
            rgba=f"{opts.tire_mount_rgba[0]} {opts.tire_mount_rgba[1]} {opts.tire_mount_rgba[2]} {opts.tire_mount_rgba[3]}",
            contype="0",
            conaffinity="0",
        )

    tire_body = SubElement(
        mount_body,
        "body",
        name=tire_name,
        pos=f"{wheel_center_offset[0]} {wheel_center_offset[1]} {wheel_center_offset[2]}",
    )

    # 2. 创建转动关节（hinge）
    tire_joint_name = f"{tire_name}__joint"
    joint_axis = canonicalize_axis_direction(
        tire_cfg.get("wheel_joint_axis", wheel_axis)
    )
    
    SubElement(tire_body, "joint",
            name=tire_joint_name,
            type="hinge",
            axis=f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}",
            damping=str(tire_damp),
            frictionloss=str(tire_fric),
            armature=str(tire_arm))

    # 3. 创建轮胎几何（圆盘：cylinder类型）
    # cylinder 默认轴线是 Z 轴，因此需要把它旋到 wheel_axis。
    
    tire_radius = diameter / 2.0                # 轮胎半径（米）
    tire_halfheight = thickness / 2.0           # 轮胎半厚度（米）
    
    gq = quat_from_two_vecs_z_to_u(wheel_axis)
    tire_quat = f"{gq[0]} {gq[1]} {gq[2]} {gq[3]}"

    SubElement(tire_body, "geom", type="cylinder",
              size=f"{tire_radius} {tire_halfheight}", quat=tire_quat,
              rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
              mass=str(mass),
              friction=f"{friction[0]} {friction[1]} {friction[2]}",
              condim="3",
              margin="0.002",
              solref="0.02 1.0",
              solimp="0.001 0.99 0.01",
              contype="1", conaffinity="1")

    return tire_joint_name

# ----------------------- 组件生成 -----------------------
def add_anchor(worldbody, sid, pos, radius, site_radius, mass, imu_on, sensor_root, ports_config=None, opts=None):
    """
    在 worldbody 下生成一个“锚点体”：
      - 一个带 free 关节的 body（便于通过等式把它“带走”）
      - 一个仅可视的小球 geom（不参与碰撞）
      - 一个 site（用于 equality connect 杆端） -> 改为在 port 上生成 site
      - 可选：IMU 传感器（加速度计与陀螺）
      - [New] 根据 ports_config 生成若干 port 子 body，用于与杆端 weld

    参数:
        worldbody   : XML 节点 <worldbody>
        sid         : 锚点 ID（如 's1'）
        pos         : 3 元序列，锚点初始化位置（世界坐标，米）
        radius      : 小球半径（米）
        site_radius : site的可视半径（米）
        mass        : 小球质量（kg），提供惯量但不参与碰撞
        imu_on      : bool，是否在该 site 上添加 IMU
        sensor_root : XML 节点 <sensor>
        sensor_root : XML 节点 <sensor>
        ports_config : list of dict, 每个元素包含 {"dir": (x,y,z), "rod_name": str, "side": str, "offset": float, "tire_cfg": dict}
                       用于生成对应的 port body
        opts        : 全局配置选项
    返回:
        anchor_data : dict, 包含 {"body_name": ..., "center_site": ..., "ports": {rod_name__side: port_site_name}, "tire_joints": [{"name":..., "ctrlrange":...}]}
    """
    x, y, z = pos
    abody_name = f"anchor_{sid}"
    abody = SubElement(worldbody, "body", name=abody_name, pos=f"{x} {y} {z}")
    SubElement(abody, "joint", name=f"{abody_name}__free", type="free", 
               armature=str(opts.anchor_armature), damping=str(opts.anchor_damping))

    # 锚点碰撞球体
    anchor_geom_rgba = opts.anchor_geom_rgba
    SubElement(abody, "geom",
               type="sphere", size=f"{radius}",
               contype="2", conaffinity="1",
               mass=str(float(mass)),
               rgba=f"{anchor_geom_rgba[0]} {anchor_geom_rgba[1]} {anchor_geom_rgba[2]} {anchor_geom_rgba[3]}",
               margin="0.002",
               solref="0.01 1.0",
               solimp="0.001 0.99 0.01")

    # 锚点site，用于挂载IMU
    asite_center = f"{abody_name}_center_site"
    SubElement(abody, "site", name=asite_center, type="sphere",
               pos="0 0 0", size=str(site_radius),  
               rgba="0.3 0.6 1.0 0.7")

    if imu_on:
        SubElement(sensor_root, "accelerometer", name=f"imu_{sid}__acc", site=asite_center)
        SubElement(sensor_root, "gyro",          name=f"imu_{sid}__gyro", site=asite_center)

    # 生成 Ports
    # ports_map: key=f"{rod_name}__{side}", value=port_site_name
    ports_map = {}
    tire_joints = []
    
    if ports_config:
        for pcfg in ports_config:
            # pcfg: {"dir": (nx, ny, nz), "rod_name": "...", "side": "left/right", "offset": float}
            # 注意：offset 可以是全局统一的，也可以是每个 port 不同的（如果未来支持）。
            # 这里我们假设使用传入的 offset (在 pcfg 里或者外部统一控制，这里假设 pcfg 里已经包含了计算好的 pos 或者 dir/offset)
            
            # 为了方便, 这里假设 pcfg 包含 {"pos": (x,y,z), "quat": (w,x,y,z), "id": unique_id}
            # 或者更简单的: 我们在这里计算 pos/quat。
            # 让我们在 build_xml 里算好 dir, 这里只负责生成。
            
            d = pcfg["dir"]
            off = pcfg["offset"]
            rname = pcfg["rod_name"]
            side = pcfg["side"]
            
            # port position relative to anchor center
            px, py, pz = d[0]*off, d[1]*off, d[2]*off
            
            # port orientation: Z axis aligns with dir
            # 复用 quat_from_two_vecs_z_to_u
            # 如果 pcfg 中包含 "quat"，则直接使用；否则回退到 dir 计算（兼容旧逻辑）
            if "quat" in pcfg:
                pq_str = pcfg["quat"]
            else:
                d = pcfg["dir"]
                pq = quat_from_two_vecs_z_to_u(d)
                pq_str = f"{pq[0]} {pq[1]} {pq[2]} {pq[3]}"
            
            port_name = f"{abody_name}_port_{rname}_{side}"
            port_body = SubElement(abody, "body", name=port_body_name(rname, side),
                                   pos=f"{px} {py} {pz}", quat=pq_str)
            
            # Port：与杆组连接的点
            port_rgba = opts.anchor_port_rgba
            SubElement(port_body, "geom", 
                       type="sphere", size=str(site_radius), 
                       rgba=f"{port_rgba[0]} {port_rgba[1]} {port_rgba[2]} {port_rgba[3]}", 
                       contype="0", conaffinity="0")
            
            # Port Site (用于 weld)
            port_site_name = f"{port_name}_site"
            SubElement(port_body, "site", name=port_site_name, 
                       pos="0 0 0", size=str(site_radius), 
                       rgba="1 0 0 0.8")
            
            key = f"{rname}__{side}"
            ports_map[key] = port_site_name
            
            # 添加轮胎，直接挂载在 Anchor Body 下
            if "tire_cfg" in pcfg and pcfg["tire_cfg"]:
                tire_cfg = pcfg["tire_cfg"]
                # 传入 abody (Anchor Body) 而不是 port_body
                tire_joint = add_tire(abody, tire_cfg, rname, side, opts)
                
                # 收集关节信息用于生成 velocity 执行器
                kv = tire_cfg.get("velocity_kv", opts.velocity_kv)
                forcerange = tire_cfg.get("velocity_forcerange", opts.velocity_forcerange)
                ctrlrange = tire_cfg.get("velocity_ctrlrange", opts.velocity_ctrlrange)
                gear = tire_cfg.get("velocity_gear", opts.velocity_gear)
                
                tire_joints.append({
                    "name": tire_joint,
                    "kv": str(kv),
                    "forcerange": f"{forcerange[0]} {forcerange[1]}",
                    "ctrlrange": f"{ctrlrange[0]} {ctrlrange[1]}",
                    "gear": str(gear)
                })

    return {"body_name": abody_name, "center_site": asite_center, "ports": ports_map, "tire_joints": tire_joints}


def add_group(mj, rg, sites_map, opts, eq_sites, collision_groups):
    """
    生成一个杆组（双 P 副）：
      root@site1 -> sleeve@midpoint(对齐 v) -> left/right 两个 slide 伸缩杆（180°相对）
      写入：几何/质量、关节、端点 site、（可选）传感器、并登记到等式聚合器 eq_sites。
      端点增加 tip_body (box) + ball joint，site 挂在 tip_body 下。

    参数:
        mj         : XML 根 <mujoco>
        rg         : 单个杆组的 JSON 字典
        sites_map  : JSON.sites（用于查 site 坐标）
        opts       : 命令行选项对象（含默认参数）
        eq_sites   : dict, 用于收集“局部端点 site -> 全局锚点 sid”的映射
                     格式: {anchor_sid: [{"rod": name, "side": "left/right", "site": site_name}, ...]}
        collision_groups : dict, 用于收集“锚点 -> 连接该锚点的所有 tip_body”
                           格式: {anchor_sid: [tip_body_name, ...]}
    备注：
        - slide 轴向：right 用 +Z，left 用 -Z（或 +Z 但几何指向相反），这样两端自然相背。
        - 端点 site：right 对应 s2，left 对应 s1。随后 equality.connect 到各自 anchor_*_site。
    """
    worldbody = mj.find("worldbody")
    sensor_root = ensure_child(mj, "sensor")

    # ---- 读取站点位置与朝向（以 site1->site2 为杆组局部 +Z） ----
    name  = rg["name"]
    s1_id = rg["site1"]
    s2_id = rg["site2"]
    p1    = tuple(rg.get("pos1", sites_map[s1_id]["pos"]))  # 允许个别组覆盖锚点坐标（可用于测试）
    p2    = tuple(rg.get("pos2", sites_map[s2_id]["pos"]))

    v = vec_sub(p2, p1)                 # 朝向向量
    L_original = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if L_original < 1e-9:
        raise ValueError(f"{name}: site1 and site2 must not coincide.")

    # 计算 Effective Length (扣除两端的 port_offset)
    # 假设 port_offset 是沿着杆轴向向内缩
    # 注意：如果 offset 很大导致 L_eff < 0，需要报错或处理
    port_off = opts.port_offset
    L = L_original - 2 * port_off
    if L < 1e-4:
        # 如果扣除后长度太小，可能 offset 设置过大
        print(f"[WARNING] {name}: effective length {L:.4f} is too small (offset={port_off}, original={L_original}). Clamping to 1e-4.")
        L = 1e-4

    quat = quat_from_two_vecs_z_to_u(v) # 将局部 +Z 旋到 v
    mid  = vec_scale(v, 0.5)            # 中点 (相对于 p1)
    # 注意：mid 还是基于原始 p1, p2 的中点，因为我们希望杆组整体位于两个锚点中心

    # ---- 尺寸与外观（默认值从 robot_config 读取）----
    sleeve_half = tuple(rg.get("sleeve_half", opts.sleeve_half))
    hx, hy, hz  = sleeve_half
    sleeve_rgba = rg.get("sleeve_rgba", opts.sleeve_rgba)
    sleeve_rad  = float(rg.get("sleeve_radius", max(hx, hy)))  # cylinder 半径
    site_radius = float(rg.get("site_radius", opts.site_radius))
    rod_radius  = float(rg.get("rod_radius",  opts.rod_radius))
    rod_trim    = float(rg.get("rod_trim",    opts.rod_trim))  # 端点修剪量

    # 两端杆的名义长度与起始偏置（“从中心起算”的伸出长度/起点）
    # 默认长度现在基于 Effective L
    rod_len_L = float(rg.get("rod_len_left",  L * 0.5))
    rod_len_R = float(rg.get("rod_len_right", L * 0.5))
    off_L     = float(rg.get("rod_start_offset_left",  opts.rod_start_offset_left))
    off_R     = float(rg.get("rod_start_offset_right", opts.rod_start_offset_right))
    constraint = rg.get("constraint", {})
    initial_slide_L = float(
        rg.get("initial_slide_left", constraint.get("initial_slide_left", constraint.get("initial_slide_each_side", 0.0)))
    )
    initial_slide_R = float(
        rg.get("initial_slide_right", constraint.get("initial_slide_right", constraint.get("initial_slide_each_side", 0.0)))
    )
    slide_control_mode = str(constraint.get("slide_control_mode", ""))
    uses_absolute_initial_slide = slide_control_mode in {
        "absolute_from_hardware_min",
        "absolute_shared_initial_slide",
    }

    # ---- 关节设置（slide_range 从 robot_config 读取）----
    slide_range = float(constraint.get("slide_range_each_side_required", DEFAULT_CONFIG.geometry.slide_range))
    slide_min = float(constraint.get("slide_min_each_side_required", 0.0))
    slide_max = float(constraint.get("slide_max_each_side_required", slide_range))
    slideL = (slide_min, slide_max)
    slideR = (slide_min, slide_max)
    slide_damp = str(rg.get("slide_damping",      opts.slide_damping))
    slide_fric = str(rg.get("slide_frictionloss", opts.slide_frictionloss))
    slide_arm  = str(rg.get("slide_armature",     opts.slide_armature))
    ball_arm   = str(rg.get("ball_armature",      opts.ball_armature))
    
    ball_damp = str(opts.ball_damping)
    ball_fric = str(opts.ball_frictionloss)
    ball_limit_deg = float(opts.ball_rot_limit_deg)
    ball_limit_rad = deg_to_rad(ball_limit_deg)
    ball_limit_enabled = ball_limit_deg > 0.0

    # ---- 根体：放在 site1，上挂 free 关节，附极小质量保证惯量合法 ----
    root = SubElement(worldbody, "body", name=f"{name}__root", pos=f"{p1[0]} {p1[1]} {p1[2]}")
    SubElement(root, "joint", name=f"{name}__free", type="free")
    if opts.root_stub_mass > 0:
        SubElement(
            root, "geom", type="sphere", size="0.0005",
            rgba="0 0 0 0", 
            mass=f"{opts.root_stub_mass:.9g}",
            contype="2", conaffinity="1"
        )

    # 中间套筒：摆在中点，方向对齐 v
    s1_body = SubElement(root, "body", name=f"{name}__s1_body", pos="0 0 0")
    sleeve = SubElement(
        s1_body, "body", name=f"{name}__sleeve",
        pos=f"{mid[0]} {mid[1]} {mid[2]}",
        quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
    )

    # 套筒几何（设为 capsule；不参与碰撞，仅可视/质量）
    sleeve_attrs = {
        "type": "capsule",
        "size": f"{sleeve_rad} {hz*0.95}",     # 因为capsule会长一点点，所以需要缩放 
        "rgba": "{} {} {} {}".format(*sleeve_rgba),
        "contype": "2", "conaffinity": "1"
    }

    if opts.enable_sleeve_mesh:
        # 如果启用 Mesh，将碰撞体设为透明
        sleeve_attrs["rgba"] = "0 0 0 0"

        # 添加视觉 Mesh Geom
        SubElement(sleeve, "geom", 
                   type="mesh", mesh=opts.sleeve_mesh_asset_name, rgba="{} {} {} {}".format(*opts.sleeve_mesh_rgba),
                   size="{} {} {}".format(*opts.sleeve_mesh_scale), contype="0", conaffinity="0", group="1", mass="0")
    # ---- 质量分配：整体质量 -> （左杆/套筒/右杆），且单体 mass/density 可覆盖 ----
    group_mass     = rg.get("group_mass", None)
    split_strategy = rg.get("split_strategy", "length")  # "length"|"volume"|"manual"
    mL_exp = rg.get("rod_mass_left",  None)
    mR_exp = rg.get("rod_mass_right", None)
    mS_exp = rg.get("sleeve_mass",    None)

    # 用于估算“分配权重”的名义几何
    aR = (0.0, 0.0, max(0.0, off_R)); bR = (0.0, 0.0, max(0.0, off_R + rod_len_R))
    aL = (0.0, 0.0, -max(0.0, off_L)); bL = (0.0, 0.0, -max(0.0, off_L + rod_len_L))

    def dist(a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    L_R = dist(aR, bR); L_L = dist(aL, bL)
    V_R = math.pi*rod_radius*rod_radius*L_R + (4.0/3.0)*math.pi*rod_radius**3
    V_L = math.pi*rod_radius*rod_radius*L_L + (4.0/3.0)*math.pi*rod_radius**3
    V_S = 8.0 * hx * hy * hz

    mL = mS = mR = None
    if group_mass is not None:
        # 先扣除显式指定的质量，其余按策略分配
        M = float(group_mass)
        known = 0.0
        if mL_exp is not None: known += float(mL_exp)
        if mR_exp is not None: known += float(mR_exp)
        if mS_exp is not None: known += float(mS_exp)
        remain = max(M - known, 0.0)

        if split_strategy == "manual":
            # 用户给权重（总和自动归一）
            wL, wS, wR = rg.get("mass_split", [0.5, 0.0, 0.5])
            s = max(wL,0) + max(wS,0) + max(wR,0) or 1.0
            wL, wS, wR = max(wL,0)/s, max(wS,0)/s, max(wR,0)/s
        elif split_strategy == "volume":
            s = V_L + V_S + V_R or 1.0
            wL, wS, wR = V_L/s, V_S/s, V_R/s
        else:  # "length"：常用默认。可配合 sleeve_min_frac 规定套筒至少分到多少。
            s = max(L_L,1e-9) + max(L_R,1e-9)
            wS = float(rg.get("sleeve_min_frac", 0.0))
            rem = max(1.0 - wS, 1e-9)
            wL, wR = (max(L_L,1e-9)/s) * rem, (max(L_R,1e-9)/s) * rem

        mL = (remain * wL) + (float(mL_exp) if mL_exp is not None else 0.0)
        mS = (remain * wS) + (float(mS_exp) if mS_exp is not None else 0.0)
        mR = (remain * wR) + (float(mR_exp) if mR_exp is not None else 0.0)
    else:
        # 没有给整体质量，就直接使用单体 mass 或者 density（如果有）
        mL, mS, mR = mL_exp, mS_exp, mR_exp

    if mS is not None:
        sleeve_attrs["mass"] = f"{float(mS):.9g}"
    SubElement(sleeve, "geom", **sleeve_attrs)
    rod_rgba = opts.rod_rgba
    exposed_radius = max(rod_radius * 0.62, rod_radius - 0.006)
    if initial_slide_R > 1e-6:
        SubElement(
            sleeve,
            "geom",
            type="capsule",
            fromto=f"0 0 0 0 0 {initial_slide_R}",
            size=f"{exposed_radius}",
            rgba=f"{rod_rgba[0]} {rod_rgba[1]} {rod_rgba[2]} {rod_rgba[3]}",
            contype="0",
            conaffinity="0",
        )
    if initial_slide_L > 1e-6:
        SubElement(
            sleeve,
            "geom",
            type="capsule",
            fromto=f"0 0 0 0 0 {-initial_slide_L}",
            size=f"{exposed_radius}",
            rgba=f"{rod_rgba[0]} {rod_rgba[1]} {rod_rgba[2]} {rod_rgba[3]}",
            contype="0",
            conaffinity="0",
        )

    # ---- 右杆：slide(+Z) + TipBody + BallJoint + Site ----
    # absolute_from_hardware_min 模式下，body 的静态 pos offset 仍然用于定义
    # 模型参考几何；joint ref/qpos 用于让“初始伸长量”成为控制状态本身。
    # MuJoCo 的 slide 几何位移为 body_pos + axis * (qpos - ref)，所以这里
    # 必须同时保留 body_pos=initial_slide，并设置 ref=initial_slide。
    right_body_pos = initial_slide_R
    right = SubElement(sleeve, "body", name=f"{name}__right", pos=f"0 0 {right_body_pos}")
    right_joint_attrs = {
        "name": f"{name}__slide_right",
        "type": "slide",
        "axis": "0 0 1",
        "range": f"{slideR[0]} {slideR[1]}",
        "limited": "true",
        "damping": slide_damp,
        "frictionloss": slide_fric,
        "armature": slide_arm,
    }
    if uses_absolute_initial_slide:
        right_joint_attrs["ref"] = f"{initial_slide_R}"
    SubElement(right, "joint", **right_joint_attrs)

    # Rod Geometry
    aRt, bRt = trim_segment(aR, bR, rod_trim)
    rodR_attrs = {
        "type": "capsule",
        # "type": "cylinder",   # 使用cylinder会产生内部的力
        "fromto": f"{aRt[0]} {aRt[1]} {aRt[2]} {bRt[0]} {bRt[1]} {bRt[2]}",
        "size": str(rod_radius),
        "rgba": f"{rod_rgba[0]} {rod_rgba[1]} {rod_rgba[2]} {rod_rgba[3]}",
        "contype": "1", "conaffinity": "1"
    }
    if mR is not None:
        rodR_attrs["mass"] = f"{float(mR):.9g}"
    SubElement(right, "geom", **rodR_attrs)

    # Tip Body (Sphere)
    tipR_name = f"{name}__right_tip"
    # TipR 默认 Z 轴朝向 +Z (即 v 方向，p1->p2)。
    # 右侧 Port (在 p2) 的 Z 轴朝向 -v (p2->p1)。
    # PortR 的姿态是 Rod * Flip180。
    # TipR 的姿态相对于 Rod 是 Flip180 (quat="0 1 0 0")。
    # 这样 TipR 和 PortR 在全局空间中完全对齐。
    tipR = SubElement(right, "body", name=tipR_name, pos=f"{bR[0]} {bR[1]} {bR[2]}", quat="0 1 0 0")
    
    # Check if this rod is an axle (has tires on both ends)
    tires_cfg_check = rg.get("tires", [])
    has_tire_left = any(t.get("end") == "left" for t in tires_cfg_check)
    has_tire_right = any(t.get("end") == "right" for t in tires_cfg_check)
    is_axle = has_tire_left and has_tire_right

    if not is_axle:
        ball_right_attrs = {
            "name": f"{name}__ball_right",
            "type": "ball",
            "damping": ball_damp,
            "frictionloss": ball_fric,
            "armature": ball_arm,
        }
        if ball_limit_enabled:
            ball_right_attrs["limited"] = "true"
            ball_right_attrs["range"] = f"0 {ball_limit_rad}"
        SubElement(tipR, "joint", **ball_right_attrs)
        
    # Tip碰撞体与可视化
    tip_rgba = opts.tip_rgba
    SubElement(tipR, "geom", type="box", size=f"{rod_radius} {rod_radius} {rod_radius}", 
               rgba=f"{tip_rgba[0]} {tip_rgba[1]} {tip_rgba[2]} {tip_rgba[3]}", contype="0", conaffinity="0")

    # Site (attached to Tip Body)
    s2_site = f"{name}__s2_site"
    SubElement(tipR, "site", name=s2_site, pos="0 0 0",
               size=str(site_radius), type="sphere", rgba="0 0 0 0")


    # ---- 左杆：slide(-Z) + TipBody + BallJoint + Site ----
    
    left_body_pos = -initial_slide_L
    left = SubElement(sleeve, "body", name=f"{name}__left", pos=f"0 0 {left_body_pos}")
    left_joint_attrs = {
        "name": f"{name}__slide_left",
        "type": "slide",
        "axis": "0 0 -1",
        "range": f"{slideL[0]} {slideL[1]}",
        "limited": "true",
        "damping": slide_damp,
        "frictionloss": slide_fric,
        "armature": slide_arm,
    }
    if uses_absolute_initial_slide:
        left_joint_attrs["ref"] = f"{initial_slide_L}"
    SubElement(left, "joint", **left_joint_attrs)
    
    # Rod Geometry
    aLt, bLt = trim_segment(aL, bL, rod_trim)
    rodL_attrs = {
        "type": "capsule",
        # "type": "cylinder",   # 使用cylinder会产生内部的力   
        "fromto": f"{aLt[0]} {aLt[1]} {aLt[2]} {bLt[0]} {bLt[1]} {bLt[2]}",
        "size": str(rod_radius),
        "rgba": f"{rod_rgba[0]} {rod_rgba[1]} {rod_rgba[2]} {rod_rgba[3]}",
        "contype": "1", "conaffinity": "1"
    }
    if mL is not None:
        rodL_attrs["mass"] = f"{float(mL):.9g}"
    SubElement(left, "geom", **rodL_attrs)

    # Tip Body (Sphere)
    tipL_name = f"{name}__left_tip"
    # TipL 默认 Z 轴朝向 +Z (即 v 方向，p1->p2)。
    # 左侧 Port (在 p1) 的 Z 轴朝向 v (p1->p2)。
    # PortL 的姿态是 Rod (Identity)。
    # TipL 的姿态相对于 Rod 是 Identity (默认)。
    # 这样 TipL 和 PortL 在全局空间中完全对齐。
    tipL = SubElement(left, "body", name=tipL_name, pos=f"{bL[0]} {bL[1]} {bL[2]}")
    
    if not is_axle:
        ball_left_attrs = {
            "name": f"{name}__ball_left",
            "type": "ball",
            "damping": ball_damp,
            "frictionloss": ball_fric,
            "armature": ball_arm,
        }
        if ball_limit_enabled:
            ball_left_attrs["limited"] = "true"
            ball_left_attrs["range"] = f"0 {ball_limit_rad}"
        SubElement(tipL, "joint", **ball_left_attrs)
        
    # Tip碰撞体与可视化
    SubElement(tipL, "geom", type="box", size=f"{rod_radius} {rod_radius} {rod_radius}", 
               rgba=f"{tip_rgba[0]} {tip_rgba[1]} {tip_rgba[2]} {tip_rgba[3]}", contype="0", conaffinity="0")

    # Site (attached to Tip Body)
    s1_site = f"{name}__s1_site"
    SubElement(tipL, "site", name=s1_site, pos="0 0 0",
               size=str(site_radius), type="sphere", rgba="0 0 0 0")


    # ---- 等式聚合登记：记录“局部端点 site 属于哪个全局锚点 sid” ----
    # 格式: {anchor_sid: [{"rod": name, "side": "left/right", "site": site_name}, ...]}
    eq_sites.setdefault(s1_id, []).append({"rod": name, "side": "left", "site": s1_site})
    eq_sites.setdefault(s2_id, []).append({"rod": name, "side": "right", "site": s2_site})
    
    # ---- 碰撞聚合登记：记录“锚点 -> 连接该锚点的所有 Rod Arm Body” ----
    # 之前是 tipL_name, tipR_name。现在改为 rod arm body (left/right)
    # 因为 tip 已经在 geom 层面 disable collision 了，所以主要是 rod arm 之间的碰撞。
    collision_groups.setdefault(s1_id, []).append(f"{name}__left")
    collision_groups.setdefault(s2_id, []).append(f"{name}__right")

    # ---- 传感器（可选）----
    if opts.add_jointaxial:
        SubElement(sensor_root, "jointactuatorfrc",
                   name=f"{name}__F_ax_left",  joint=f"{name}__slide_left")
        SubElement(sensor_root, "jointactuatorfrc",
                   name=f"{name}__F_ax_right", joint=f"{name}__slide_right")
    if opts.add_tip_force:
        SubElement(sensor_root, "force",  name=f"{name}__tipF_left_vec",  site=s1_site)
        SubElement(sensor_root, "force",  name=f"{name}__tipF_right_vec", site=s2_site)
        SubElement(sensor_root, "torque", name=f"{name}__tipT_left_vec",  site=s1_site)
        SubElement(sensor_root, "torque", name=f"{name}__tipT_right_vec", site=s2_site)


def build_xml(cfg, opts):
    """
    从 JSON 配置构建完整 MuJoCo XML 的 Element 树。
    主要步骤：
      1) 在 worldbody 下生成所有锚点（各自 free）。
      2) 按 rod_groups 生成所有杆组。
      3) 建立 equality.connect：将每个“局部端点 site”连到对应“全局锚点 site”。
      4) 为 actuated 杆组创建 position 执行器（左右各一）。
      5) 组内排碰（left/right/sleeve）。

    参数:
        cfg  : JSON 字典，包含 "sites" 与 "rod_groups"
        opts : 命令行配置对象
    返回:
        mj   : XML 根节点 <mujoco>
    """
    mj = Element("mujoco", model=opts.model_name)

    # # 求解/物理参数
    # SubElement(mj, "option",
    #            solver="Newton",
    #            timestep=str(opts.timestep),
    #            gravity="{} {} {}".format(*opts.gravity),
    #            iterations=str(opts.iterations),
    #            ls_iterations=str(opts.ls_iterations),
    #            tolerance=str(opts.tolerance))
    SubElement(mj, "size", njmax=str(opts.njmax), nconmax=str(opts.nconmax))

    # 惯量由几何估算（省去手写惯量张量）
    compiler_attrs = {"inertiafromgeom": "true", "angle": "radian"}
    if opts.enable_sleeve_mesh:
        compiler_attrs["meshdir"] = opts.meshdir
    SubElement(mj, "compiler", **compiler_attrs)
    
    if opts.enable_sleeve_mesh:
        asset_root = SubElement(mj, "asset")
        SubElement(asset_root, "mesh", name=opts.sleeve_mesh_asset_name, file=opts.sleeve_mesh_file, scale="{} {} {}".format(*opts.sleeve_mesh_scale))
    
    worldbody = SubElement(mj, "worldbody")
    sensor_root = ensure_child(mj, "sensor")

    # 1) 预计算：收集每个 anchor 需要连接的杆信息 (dir, rod_name, side)
    #    用于在 add_anchor 时生成对应的 ports
    anchor_ports_config = {} # {sid: [{"dir": (x,y,z), "rod_name": str, "side": str, "offset": float, "tire_cfg": dict}, ...]}
    
    for rg in cfg["rod_groups"]:
        name = rg["name"]
        s1_id = rg["site1"]
        s2_id = rg["site2"]
        p1 = tuple(rg.get("pos1", cfg["sites"][s1_id]["pos"]))
        p2 = tuple(rg.get("pos2", cfg["sites"][s2_id]["pos"]))
        
        v = vec_sub(p2, p1)
        L = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if L > 1e-9:
            d = (v[0]/L, v[1]/L, v[2]/L)
            d_neg = (-d[0], -d[1], -d[2])
            
            # 计算 Rod 的全局姿态 (Z轴指向 v)
            q_rod = quat_from_two_vecs_z_to_u(d)
            q_rod_str = f"{q_rod[0]} {q_rod[1]} {q_rod[2]} {q_rod[3]}"
            
            # 计算 Right Port 的全局姿态 (Rod * Flip180)
            # Flip180 around X: (0, 1, 0, 0)
            q_flip = (0, 1, 0, 0)
            q_port_right = quat_mul(q_rod, q_flip)
            q_port_right_str = f"{q_port_right[0]} {q_port_right[1]} {q_port_right[2]} {q_port_right[3]}"
            
            # s1 连接的是 left side (方向 v, 即 d) -> PortL 姿态 = Rod 姿态
            # s2 连接的是 right side (方向 -v, 即 d_neg) -> PortR 姿态 = Rod 姿态 * Flip180
            
            # 查找是否有轮胎配置
            tires_cfg = rg.get("tires", [])
            # 必须复制，因为我们要注入计算出的轴信息，不能修改原始配置
            tire_cfg_left = next((t.copy() for t in tires_cfg if t.get("end") == "left"), None)
            tire_cfg_right = next((t.copy() for t in tires_cfg if t.get("end") == "right"), None)

            # 旧版兼容：
            # 旧 JSON 没有显式 mount_dir / wheel_axis 时，仍回退为“沿杆方向 + 全局下偏”的近似放置方式。
            # 新版 structure_generator 会直接写入 mount_dir / wheel_axis，此处不覆盖它们。
            if tire_cfg_left:
                if "mount_dir" not in tire_cfg_left:
                    tire_cfg_left.setdefault("wheel_pos_axis", (-d[0], -d[1], -d[2]))
                tire_cfg_left.setdefault("wheel_axis", tire_cfg_left.get("wheel_pos_axis", (-d[0], -d[1], -d[2])))
                tire_cfg_left.setdefault("drop_axis_local", (0.0, 0.0, -1.0))
                
            if tire_cfg_right:
                if "mount_dir" not in tire_cfg_right:
                    tire_cfg_right.setdefault("wheel_pos_axis", d)
                tire_cfg_right.setdefault("wheel_axis", tire_cfg_right.get("wheel_pos_axis", d))
                tire_cfg_right.setdefault("drop_axis_local", (0.0, 0.0, -1.0))

            anchor_ports_config.setdefault(s1_id, []).append({
                "dir": d, "rod_name": name, "side": "left", "offset": opts.port_offset,
                "quat": q_rod_str,
                "tire_cfg": tire_cfg_left
            })
            anchor_ports_config.setdefault(s2_id, []).append({
                "dir": d_neg, "rod_name": name, "side": "right", "offset": opts.port_offset,
                "quat": q_port_right_str,
                "tire_cfg": tire_cfg_right
            })

    # 2) 锚点
    anchor_data_map = {} # {sid: {"body_name":..., "ports": {...}}}
    for sid, s in cfg["sites"].items():
        adata = add_anchor(
            worldbody, sid=sid, pos=s["pos"],
            radius=s.get("radius", opts.anchor_radius),
            site_radius=s.get("site_radius", opts.site_radius),
            mass=s.get("mass",   opts.anchor_mass),
            imu_on=opts.add_imu, sensor_root=sensor_root,
            ports_config=anchor_ports_config.get(sid),
            opts=opts
        )
        anchor_data_map[sid] = adata

    # 3) 杆组
    eq_sites = {}  # {global_sid: [{"rod":..., "side":..., "site":...}, ...]}
    collision_groups = {} # {global_sid: [tip_body_name, ...]}
    
    for rg in cfg["rod_groups"]:
        add_group(mj, rg, cfg["sites"], opts, eq_sites, collision_groups)

    # 4) 等式连接（将局部端点 site 与全局锚点 Port site 对齐 -> weld）
    eq = SubElement(mj, "equality")
    for sid, connections in eq_sites.items():
        if sid not in anchor_data_map:
            continue
        adata = anchor_data_map[sid]
        
        for conn in connections:
            # conn: {"rod": name, "side": "left", "site": s1_site}
            # 找到对应的 port site
            port_key = f"{conn['rod']}__{conn['side']}"
            if port_key in adata["ports"]:
                port_site = adata["ports"][port_key]
                
                # 使用 weld 连接
                SubElement(eq, "weld", site1=conn['site'], site2=port_site,
                           solref="{} {}".format(opts.solref[0], opts.solref[1]),
                           solimp="{} {} {}".format(opts.solimp[0], opts.solimp[1], opts.solimp[2]),
                        #    torquescale="0.1"
                           )
            else:
                print(f"[WARN] Port not found for {port_key} at anchor {sid}")

    # 5) 执行器（仅对 actuated 组；左右各一个 position 控制器）
    act = SubElement(mj, "actuator")
    initial_ctrl_values = []
    for rg in cfg["rod_groups"]:
        constraint = rg.get("constraint", {})
        slide_range = float(constraint.get("slide_range_each_side_required", DEFAULT_CONFIG.geometry.slide_range))
        slide_min = float(constraint.get("slide_min_each_side_required", 0.0))
        slide_max = float(constraint.get("slide_max_each_side_required", slide_range))
        slide_mode = str(constraint.get("slide_control_mode", ""))
        if rg.get("actuated", False):
            for side in ("left", "right"):
                jname = f"{rg['name']}__slide_{side}"
                initial_slide = 0.0
                if slide_mode in {"absolute_from_hardware_min", "absolute_shared_initial_slide"}:
                    initial_slide = float(
                        rg.get(
                            f"initial_slide_{side}",
                            constraint.get(f"initial_slide_{side}", constraint.get("initial_slide_each_side", 0.0)),
                        )
                    )
                SubElement(
                    act,
                    "position",
                    name=f"act_{jname}",
                    joint=jname,
                    kp=str(opts.kp),
                    kv=str(opts.kv),
                    ctrlrange=f"{slide_min} {slide_max}"   # 一侧杆的 slide 控制范围；绝对模式下所有同型号杆相同
                )
                initial_ctrl_values.append(initial_slide)
    
    # Actuators for Tires (使用 velocity 执行器，默认低速档参数)
    for sid, adata in anchor_data_map.items():
        for tire_info in adata.get("tire_joints", []):
             SubElement(
                act,
                "velocity",
                name=f"act_{tire_info['name']}",
                joint=tire_info['name'],
                kv=tire_info['kv'],
                forcerange=tire_info['forcerange'],
                ctrlrange=tire_info['ctrlrange'],
                gear=tire_info['gear']
            )
             initial_ctrl_values.append(0.0)

    if initial_ctrl_values:
        keyframe = SubElement(mj, "keyframe")
        SubElement(
            keyframe,
            "key",
            name="initial_ctrl",
            ctrl=" ".join(f"{value:.12g}" for value in initial_ctrl_values),
        )

    # 6) 组内排碰（避免 left/right/sleeve 自碰）
    contact = SubElement(mj, "contact")
    for rg in cfg["rod_groups"]:
        n = rg["name"]
        SubElement(contact, "exclude", body1=f"{n}__left",  body2=f"{n}__right")
        SubElement(contact, "exclude", body1=f"{n}__left",  body2=f"{n}__sleeve")
        SubElement(contact, "exclude", body1=f"{n}__right", body2=f"{n}__sleeve")
        
        # 还要排除 tip body 与其父级 rod 的碰撞 (虽然通常父子之间默认不碰，但显式排除更安全)
        SubElement(contact, "exclude", body1=f"{n}__left", body2=f"{n}__left_tip")
        SubElement(contact, "exclude", body1=f"{n}__right", body2=f"{n}__right_tip")

        # 排除轮胎与所属杆的碰撞 (虽然现在轮胎在 Anchor Port 上，但为了保险起见，或者如果需要排除轮胎与杆的碰撞)
        # 轮胎名字: {rod_name}__tire_{side}
        # 杆名字: {rod_name}__{side}
        # 注意：轮胎现在是 Anchor 的子孙，杆是 Anchor 的连接对象（通过 Weld）。
        # 它们之间可能会发生碰撞。
        tires_cfg = rg.get("tires", [])
        for tire_cfg in tires_cfg:
            side = tire_cfg.get("end")
            tire_name = tire_cfg.get("name", f"{n}__tire_{side}")
            # rod_body = f"{n}__{side}"
            # SubElement(contact, "exclude", body1=rod_body, body2=tire_name)
            
            # 同时也排除轮胎与 Anchor Body 的碰撞 (虽然是子孙，但 Mujoco 默认只排除父子)
            # 找到对应的 Anchor ID
            anchor_id = rg["site1"] if side == "left" else rg["site2"]
            if anchor_id in anchor_data_map:
                anchor_body = anchor_data_map[anchor_id]["body_name"]
                SubElement(contact, "exclude", body1=anchor_body, body2=tire_name)

    # 7) 锚点处排碰
    #    - 杆臂 (rod arm) 与其连接的 anchor body 互不碰撞（避免节点处几何重叠导致排斥力）
    #    - 不同杆组的杆臂之间 **允许碰撞**（contype=1 天然互碰，无需额外处理）
    for sid, rod_arms in collision_groups.items():
        if sid not in anchor_data_map:
            continue
        anchor_body = anchor_data_map[sid]["body_name"]
        
        # Rod arm vs Anchor: 排除碰撞
        for arm in rod_arms:
            SubElement(contact, "exclude", body1=arm, body2=anchor_body)

    return mj


# ----------------------- CLI / 主函数 -----------------------

def apply_robot_config(opts):
    """
    从 robot_config.DEFAULT_CONFIG 填充 opts 对象的所有物理/几何参数。
    
    这个函数实现了"单一真理源"模式：
    - 命令行只保留 I/O 和布尔开关
    - 所有物理参数从 robot_config.py 统一读取
    
    Args:
        opts: 一个空的对象，用于存储配置参数
    """
    cfg = DEFAULT_CONFIG
    
    # ===== 物理/求解器参数 =====
    opts.timestep = cfg.physics.timestep
    opts.gravity = cfg.physics.gravity
    opts.iterations = cfg.physics.iterations
    opts.ls_iterations = cfg.physics.ls_iterations
    opts.tolerance = cfg.physics.tolerance
    opts.njmax = cfg.physics.njmax
    opts.nconmax = cfg.physics.nconmax
    
    # ===== 关节/控制参数 =====
    opts.kp = cfg.joint_control.kp
    opts.kv = cfg.joint_control.kv
    opts.slide_damping = cfg.joint_control.slide_damping
    opts.slide_frictionloss = cfg.joint_control.slide_frictionloss
    opts.slide_armature = cfg.joint_control.slide_armature
    
    # ===== 锚点参数 =====
    opts.anchor_radius = cfg.anchor.radius
    opts.anchor_mass = cfg.anchor.mass
    opts.anchor_armature = cfg.anchor.armature
    opts.anchor_damping = cfg.anchor.damping
    
    # ===== 球铰参数 =====
    opts.ball_armature = cfg.ball_joint.armature
    opts.ball_frictionloss = cfg.ball_joint.frictionloss
    opts.ball_damping = cfg.ball_joint.damping
    opts.ball_rot_limit_deg = cfg.ball_joint.rot_limit_deg
    
    # ===== 杆组参数 =====
    opts.rod_radius = cfg.rod_group.rod_radius
    opts.rod_trim = cfg.rod_group.rod_trim
    opts.sleeve_half = cfg.rod_group.sleeve_half
    opts.group_mass = cfg.rod_group.group_mass
    opts.split_strategy = cfg.rod_group.split_strategy
    opts.mass_split = cfg.rod_group.mass_split
    opts.rod_start_offset_left = cfg.rod_group.rod_start_offset_left
    opts.rod_start_offset_right = cfg.rod_group.rod_start_offset_right
    
    # ===== 几何参数 =====
    opts.site_radius = cfg.geometry.site_radius
    opts.port_offset = cfg.geometry.node_offset
    
    # ===== 可视化参数 =====
    opts.root_stub_mass = cfg.visual.root_stub_mass
    opts.anchor_geom_rgba = cfg.visual.anchor_geom_rgba
    opts.anchor_port_rgba = cfg.visual.anchor_port_rgba
    opts.sleeve_rgba = cfg.visual.sleeve_rgba
    opts.rod_rgba = cfg.visual.rod_rgba
    opts.tip_rgba = cfg.visual.tip_rgba
    opts.enable_sleeve_mesh = cfg.visual.enable_sleeve_mesh
    opts.meshdir = cfg.visual.meshdir
    opts.sleeve_mesh_file = cfg.visual.sleeve_mesh_file
    opts.sleeve_mesh_asset_name = cfg.visual.sleeve_mesh_asset_name
    opts.sleeve_mesh_scale = cfg.visual.sleeve_mesh_scale
    opts.sleeve_mesh_rgba = cfg.visual.sleeve_mesh_rgba
    
    # ===== 等式约束参数 =====
    opts.solref = list(cfg.equality.solref)
    opts.solimp = list(cfg.equality.solimp)
    
    # ===== 轮胎参数 =====
    opts.tire_mass = cfg.tire.mass
    opts.tire_diameter = cfg.tire.diameter
    opts.tire_thickness = cfg.tire.thickness
    opts.tire_distance = cfg.tire.distance
    opts.tire_mount_length = cfg.tire.mount_length
    opts.tire_axle_offset = cfg.tire.axle_offset
    opts.tire_mount_radius = cfg.tire.mount_radius
    opts.tire_mount_mass = cfg.tire.mount_mass
    opts.tire_mount_rgba = cfg.tire.mount_rgba
    opts.tire_friction = cfg.tire.friction
    opts.tire_rgba = cfg.tire.rgba
    opts.tire_damping = cfg.tire.joint_damping
    opts.tire_joint_frictionloss = cfg.tire.joint_frictionloss
    opts.tire_armature = cfg.tire.armature

    # ===== 轮胎速度执行器参数 =====
    opts.velocity_kv = cfg.tire.velocity_kv
    opts.velocity_forcerange = cfg.tire.velocity_forcerange
    opts.velocity_ctrlrange = cfg.tire.velocity_ctrlrange
    opts.velocity_gear = cfg.tire.velocity_gear
    
    # ===== 传感器配置 =====
    opts.add_imu = cfg.sensor.add_imu
    opts.add_jointaxial = cfg.sensor.add_jointaxial
    opts.add_tip_force = cfg.sensor.add_tip_force
    
    return opts


def parse_args(argv=None):
    """
    命令行参数：
    
    仅保留：
      - I/O 参数 (-i, -o)
      - 命名参数 (--model-name)
      - 布尔开关 (--no-imu, --no-jointaxial, --no-tip-force)
      
    所有物理/几何参数从 robot_config.py 统一读取，无需命令行指定。
    """
    p = argparse.ArgumentParser(
        description="Multi-rod-group JSON -> MuJoCo XML generator (参数从 robot_config.py 读取)"
    )
    
    # ===== I/O 参数 =====
    p.add_argument("-i", "--input",  default="moce_robot/json/test2.json", 
                   help="JSON 结构文件路径")
    p.add_argument("-o", "--output", default="moce_robot/json/test2.xml",  
                   help="输出 XML 文件路径")
    
    # ===== 命名参数 =====
    p.add_argument("--model-name", default="multi_rod_groups_equiv",
                   help="MuJoCo 模型名称")
    
    # ===== 传感器开关 =====
    p.add_argument("--no-imu", action="store_true", 
                   help="不添加锚点 IMU（accelerometer/gyro）")
    p.add_argument("--no-jointaxial", action="store_true", 
                   help="不添加 jointactuatorfrc 关节轴向力传感器")
    p.add_argument("--no-tip-force", action="store_true", 
                   help="不添加端点力/力矩传感器")
    
    return p.parse_args(argv)


def main(argv=None):
    """
    入口：
      1) 解析命令行参数（仅 I/O 和开关）
      2) 从 robot_config.py 加载物理/几何参数
      3) 读取 JSON（仅结构信息）
      4) 构建 XML ElementTree
      5) 写出带 XML 声明的文件
    """
    args = parse_args(argv)

    # 创建配置对象并从 robot_config 填充参数
    class O: pass
    o = O()
    
    # 复制命令行参数
    o.input = args.input
    o.output = args.output
    o.model_name = args.model_name.replace("-", "_")
    
    # 从 robot_config.py 统一加载所有物理/几何参数
    apply_robot_config(o)
    
    # 命令行开关覆盖配置文件中的传感器设置
    if args.no_imu:
        o.add_imu = False
    if args.no_jointaxial:
        o.add_jointaxial = False
    if args.no_tip_force:
        o.add_tip_force = False

    # 读 JSON（现在只包含结构信息）
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] input JSON not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 如果 JSON 中有 port_offset，使用它覆盖默认值
    if "port_offset" in cfg:
        o.port_offset = cfg["port_offset"]

    # 构建 XML
    mj = build_xml(cfg, o)

    # 写文件
    ElementTree(mj).write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote XML -> {args.output}")
    print(f"[INFO] 物理参数从 robot_config.py 加载")


if __name__ == "__main__":
    main()


# ----------------------- 使用示例（复制到命令行） -----------------------
# python json2xml_v2.py -i ../json/test_Tetrahedron.json -o ../xml/test_Tetrahedron.xml
#
# # 改时间步与重力、关闭 IMU 与端点力矩传感器
# python json2xml_v2.py -i ../json/test_Tetrahedron.json -o ../xml/test_Tetrahedron.xml \
#   --timestep 0.001 --gravity 0 0 -9.81 \
#   --no-imu --no-tip-force
#
# # 调整 equality 约束柔性与执行器增益
# python json2xml.py -i ../json/test_Tetrahedron.json -o ../json/test_Tetrahedron.xml \
#   --solref 0.01 2.0 --solimp 0.95 0.995 0.001 \
#   --kp 150 --kv 20
