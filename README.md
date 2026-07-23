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

后续集成时，只将本项目中自有的代码复制到 `QianJi/video_pose_extraction`，不克隆或搬运上游 DeepLabCut 源码。
