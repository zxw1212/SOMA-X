# BONES SOMA BVH 转 SMPL-X 使用说明

本文说明如何使用仓库中的脚本，把 BONES-SEED 的 `soma_uniform/*.bvh` 转成 `SMPL-X` 参数文件。

当前实现对应脚本：

- `tools/convert_bones_soma_bvh_to_smplx.py`
- `soma/bones_smplx.py`

## 1. 转换内容

输入：

- BONES-SEED 的 `soma_uniform/bvh/.../*.bvh`
- `assets/SMPLX/SMPLX_NEUTRAL.npz`

输出：

- `SMPL-X` 参数 `.npz`
- 额外保存 BVH 目标 22 个 body joints、SMPL-X FK 后的 22 个 body joints 和误差统计

当前版本处理的是身体主干 22 个关节：

- pelvis
- left/right hip
- spine1/spine2/spine3
- left/right knee
- left/right ankle
- left/right foot
- neck
- left/right collar
- head
- left/right shoulder
- left/right elbow
- left/right wrist

手指、眼睛、下巴、表情当前默认输出为 0。

## 2. 实现思路

当前实现不依赖 `SOMALayer/Warp`，而是 direct-only：

1. 解析 BVH 层级、通道和 motion。
2. 用 BVH 自身的 `OFFSET + local rotation + local translation` 做 FK，提取 body joints 的位置；同时保留原始 BVH local rotation 通道。
3. 如果传了 `--direct-tpose-frame`，就直接把这个真实 BVH 帧当成标定 T-pose。
4. 如果没有传 `--direct-tpose-frame`，脚本会先自动找一个“最适合估计身体朝向”的参考帧：
   - 优先找最接近“站立且双臂水平展开”的帧
   - 如果实在找不到，就退化为找一个尽量站直、肩宽较明显的帧
5. 在这个参考帧上估计身体的三个基准方向：
   - `up`：`Hips -> Head`
   - `left`：`RightArm -> LeftArm`
   - `forward = up x left`
6. 用这三个方向，自动构造一个 BVH 的“标准 T-pose”用于标定：
   - spine / neck 朝 `up`
   - left arm 朝 `left`
   - right arm 朝 `right`
   - legs 朝 `down`
   - feet 朝 `forward`
   对于 `Head`、`Hand`、`ToeBase` 这类单靠“标准 T-pose”并不能唯一确定 twist 的 leaf joints，保留参考帧里的 joint 轴扭转方向，避免引入过大的额外偏差。
7. 记录这个“真实帧或合成帧”下 BVH body joints 的世界 joint 坐标系朝向，记为 `R_bvh_tpose[j]`。
8. 记录 canonical SMPL-X T-pose 下 body joints 的世界 joint 坐标系朝向，记为 `R_smplx_tpose[j]`。这里的 canonical SMPL-X T-pose 就是标准 SMPL-X zero pose，所以 local rotation 全是单位阵，对应的世界 joint 坐标系也全是单位阵。
9. 计算每个 joint 在 T-pose 下从 BVH joint frame 到 SMPL-X joint frame 的 offset：
   `R_offset[j] = R_bvh_tpose[j]^T @ R_smplx_tpose[j]`
10. 对每一帧，读取 BVH 原始 local rotation，记为 `R_bvh_local[j, t]`。
11. 用标定阶段得到的 parent/joint offset，把这份 BVH local rotation 重新解释到 SMPL-X 的 local joint frame 中：
   `R_smplx_local[j, t] = R_offset[parent(j)]^T @ R_bvh_local[j, t] @ R_offset[j]`
   其中 root 的 `R_offset[parent(root)]` 按单位阵处理。
12. 如果把上式完全展开，它等价于：
   `R_smplx_local[j, t] = R_smplx_tpose[parent(j)]^T @ R_bvh_tpose[parent(j)] @ R_bvh_local[j, t] @ R_bvh_tpose[j]^T @ R_smplx_tpose[j]`
   这个式子的含义很直接：先用 BVH 的标定 T-pose 坐标轴把当前 local rotation 里的“BVH 坐标轴定义”消掉，再放进 SMPL-X 的标定 T-pose 坐标轴里。
13. `betas` 固定为 0，只做 FK，不再做优化拟合。
14. 保存 `SMPL-X` 参数和 body joint 误差；导出时统一转成 Z-up。

这样做的关键点是：默认情况下不再要求 BVH 动作里真的出现一个完美 T-pose，而是先自动构造一个“足够标准”的 BVH T-pose 来做 joint-frame 标定；随后仍然通过 `BVH -> SMPL-X` 的 T-pose offset，把每一帧的 BVH local rotation 换到标准 SMPL-X local frame。

这样做的优点是：

- 不需要 CUDA
- 可以直接在 CPU 环境运行
- 没有优化过程，速度更快
- 导出的 joint frame 更接近标准 SMPL-X 约定

## 3. 运行前提

默认使用仓库虚拟环境：

```bash
.venv/bin/python
```

需要确认以下文件存在：

```bash
assets/SMPLX/SMPLX_NEUTRAL.npz
```

仓库里现在也自带了一个可复用的标定 clip：

```bash
assets/calibration/soma_t_pose.bvh
```

它来自：

- 源文件：`/home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh`
- 截取范围：原始 BVH 的第 `5` 到第 `15` 帧
- 对应关系：原始第 `10` 帧 = clip 内的第 `5` 帧

如果你已经在仓库根目录下，可以先检查：

```bash
ls assets/SMPLX/SMPLX_NEUTRAL.npz
```

## 4. 单个 BVH 转换

示例：

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh \
  convert_output/walk_forward_professional_003__A001_smplx.npz \
  --device cuda:0 \
  --frame-stride 1
```

如果你想强制用项目内置的标定 clip 来计算 offset，可以这样：

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh \
  convert_output/walk_forward_professional_003__A001_smplx.npz \
  --device cpu \
  --calibration-bvh assets/calibration/soma_t_pose.bvh \
  --calibration-bvh-frame 5
```

这里 `--calibration-bvh-frame 5` 的含义是：使用 `assets/calibration/soma_t_pose.bvh` 里的第 `5` 帧，也就是原始大 BVH 的第 `10` 帧，作为标定 pose。

## 5. 批量目录转换

如果输入是目录，脚本会递归查找 `.bvh`，并在输出目录下保留相对目录结构。

示例：

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh \
  /home/zxw/Documents/bones_studio_demo/smplx_npz \
  --device cpu
```

如果你只想先抽样测试前几个文件：

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh \
  /home/zxw/Documents/bones_studio_demo/smplx_npz \
  --device cpu \
  --limit-files 5
```

## 6. 常用参数

### `--device`

默认是：

```bash
--device cpu
```

当前实现已经验证可在 CPU 运行。

### `--frame-stride`

按时间下采样。

例如：

```bash
--frame-stride 2
```

表示每隔 2 帧取 1 帧再做转换。

### `--max-frames`

限制参与转换的帧数，适合先做快速验证。

例如：

```bash
--max-frames 32
```

### `--direct-tpose-frame`

手动指定 BVH 中作为标定 T-pose 的绝对帧号。

传了这个参数后：

- 脚本直接使用这个真实 BVH 帧做标定
- 不再自动构造 synthetic BVH T-pose

例如：

```bash
--direct-tpose-frame 3008
```

如果不传，脚本会先找一个参考帧来估计身体的 `up / left / forward`，然后自动构造一个 synthetic BVH 标准 T-pose 再做标定。
所以即使 BVH 里没有真正的 T-pose，也可以直接转换。

### `--calibration-bvh`

指定一个外部 BVH，只用于计算标定 pose offset，不参与实际动作转换。

典型用途：

- 你想让很多动作都共用同一个标定 pose
- 你不想每个动作都各自自动合成 T-pose
- 你已经人工挑好了一个更可信的标定片段

例如：

```bash
--calibration-bvh assets/calibration/soma_t_pose.bvh
```

### `--calibration-bvh-frame`

当 `--calibration-bvh` 已经传入时，可以再指定使用这个外部 BVH 的哪一帧做标定。

例如：

```bash
--calibration-bvh-frame 5
```

如果不传，脚本会在这个外部 BVH 上先选一个参考帧，再自动构造 synthetic T-pose。

## 7. 输出文件内容

输出 `.npz` 中包含：

- `source_path`
- `fps`
- `frame_indices`
- `betas`
- `global_orient`
- `body_pose`
- `transl`
- `left_hand_pose`
- `right_hand_pose`
- `jaw_pose`
- `leye_pose`
- `reye_pose`
- `expression`
- `bvh_body_joint_names`
- `smplx_body_joint_names`
- `target_body_joints`
- `fitted_body_joints`
- `body_joint_error`
- `body_joint_position_error`
- `bone_direction_error_deg`
- `bone_direction_metric_excluded_joint_names`
- `mean_error_m`
- `p95_error_m`
- `max_error_m`
- `body_joint_position_mean_m`
- `body_joint_position_p95_m`
- `body_joint_position_max_m`
- `bone_direction_mean_deg`
- `bone_direction_p95_deg`
- `bone_direction_max_deg`

主要关心的字段通常是：

- `betas`: `(10,)`
- `global_orient`: `(T, 3)`
- `body_pose`: `(T, 63)`
- `transl`: `(T, 3)`

误差字段说明：

- `body_joint_error` 和 `body_joint_position_error` 是同一份 3D 关节位置误差，单位米；前者是兼容旧字段名，后者是更明确的新字段名
- `mean_error_m / p95_error_m / max_error_m` 也是旧字段名，对应的仍然是 body joint position error
- `bone_direction_error_deg` 是逐帧逐骨的方向夹角误差，单位度
- `bone_direction_mean_deg / bone_direction_p95_deg / bone_direction_max_deg` 的统计默认排除了 `left_collar / right_collar / left_foot / right_foot`

## 8. Python 直接调用

如果不想走 CLI，可以直接在 Python 中调用：

```python
from soma.bones_smplx import convert_bvh_to_smplx_direct, save_conversion_result

result = convert_bvh_to_smplx_direct(
    "/home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh",
    model_path="assets/SMPLX/SMPLX_NEUTRAL.npz",
    device="cpu",
    frame_stride=2,
    max_frames=32,
    calibration_frame=None,
    calibration_bvh_path="assets/calibration/soma_t_pose.bvh",
    calibration_bvh_frame=5,
)

print(result.metrics.mean_error_m)
print(result.metrics.p95_error_m)
print(result.metrics.max_error_m)
print(result.bone_direction_metrics.mean_error_deg)
print(result.bone_direction_metrics.p95_error_deg)
print(result.bone_direction_metrics.max_error_deg)

save_conversion_result(
    result,
    "convert_output/sample_smplx.npz",
    model_path="assets/SMPLX/SMPLX_NEUTRAL.npz",
)
```

## 9. 误差字段

转换脚本不再在终端打印误差统计。

如果你需要看误差，可以直接读取导出的 `.npz` 中这些字段：

- `body_joint_error`
- `body_joint_position_error`
- `mean_error_m`
- `p95_error_m`
- `max_error_m`
- `bone_direction_error_deg`
- `bone_direction_metric_excluded_joint_names`
- `bone_direction_mean_deg`
- `bone_direction_p95_deg`
- `bone_direction_max_deg`

其中：

- `body_joint_error` 和 `body_joint_position_error` 是同一份 3D 关节位置误差，单位米；前者是兼容旧字段名
- `mean_error_m / p95_error_m / max_error_m` 也是旧字段名，对应的仍然是 body joint position error
- `bone_direction_error_deg` 是逐帧逐骨的方向夹角误差，单位度
- `bone_direction_mean_deg / bone_direction_p95_deg / bone_direction_max_deg` 的统计默认排除了 `left_collar / right_collar / left_foot / right_foot`

另外现在也提供了一个专门的评估脚本，可以直接对 `bvh` 和导出的 `npz` 做逐帧 pose error 分析。

示例：

```bash
.venv/bin/python tools/eval_bones_bvh_vs_smplx_pose_error.py \
  --npz-file convert_output/jump_and_land_heavy_001__A001_smplx.npz
```

这个评估会同时输出 4 类误差：

- `raw_global`: 不做任何对齐，直接比较全局关节位置
- `body_frame`: 每帧用骨盆和躯干方向建立局部 body frame 后比较
- `anchor_rigid_all`: 先用锚点做刚体对齐，再统计全部 22 个关节
- `anchor_rigid_non_anchor`: 先用锚点做刚体对齐，再只统计非锚点关节

默认锚点是：

- `Hips`
- `LeftLeg`
- `RightLeg`
- `LeftShoulder`
- `RightShoulder`
- `Neck1`

脚本还会输出逐帧 CSV，默认保存在：

```bash
<npz_file>.pose_error.csv
```

## 10. 当前限制

当前版本有这些边界：

1. 只拟合了 22 个 body joints，没有拟合手指和面部。
2. 输出里 `left_hand_pose/right_hand_pose/jaw_pose/leye_pose/reye_pose/expression` 目前为 0。
3. 这是 joint-based 拟合（位置 + 旋转），不是 mesh-level 拟合，所以局部扭转和细节不会完全恢复。
4. 误差可接受不代表所有动作都同样稳定，极端动作仍建议单独抽样检查。

## 11. 建议使用流程

建议先这样跑：

1. 单个文件，小帧数，验证误差。
2. 小批量目录，观察平均误差分布。
3. 再跑全量目录。

推荐第一步命令：

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh \
  convert_output/walk_forward_professional_003__A001_smplx.npz \
  --device cpu \
  --frame-stride 2 \
  --max-frames 32
```
