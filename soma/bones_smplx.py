from __future__ import annotations

import io

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation

Y_UP_TO_Z_UP_ROTATION_MATRIX = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
)


BODY_JOINT_MAP: list[tuple[str, str]] = [
    ("Hips", "pelvis"),
    ("LeftLeg", "left_hip"),
    ("RightLeg", "right_hip"),
    ("Spine1", "spine1"),
    ("LeftShin", "left_knee"),
    ("RightShin", "right_knee"),
    ("Spine2", "spine2"),
    ("LeftFoot", "left_ankle"),
    ("RightFoot", "right_ankle"),
    ("Chest", "spine3"),
    ("LeftToeBase", "left_foot"),
    ("RightToeBase", "right_foot"),
    ("Neck1", "neck"),
    ("LeftShoulder", "left_collar"),
    ("RightShoulder", "right_collar"),
    ("Head", "head"),
    ("LeftArm", "left_shoulder"),
    ("RightArm", "right_shoulder"),
    ("LeftForeArm", "left_elbow"),
    ("RightForeArm", "right_elbow"),
    ("LeftHand", "left_wrist"),
    ("RightHand", "right_wrist"),
]

SMPLX_BODY_JOINT_NAMES = [smplx_name for _, smplx_name in BODY_JOINT_MAP]
BVH_BODY_JOINT_NAMES = [bvh_name for bvh_name, _ in BODY_JOINT_MAP]
BONE_DIRECTION_METRIC_EXCLUDED_JOINT_NAMES = {
    "left_collar",
    "right_collar",
    "left_foot",
    "right_foot",
}


@dataclass
class FitMetrics:
    mean_error_m: float
    p95_error_m: float
    max_error_m: float


@dataclass
class AngularMetrics:
    mean_error_deg: float
    p95_error_deg: float
    max_error_deg: float


@dataclass
class ConversionResult:
    source_path: Path
    fps: float
    frame_indices: np.ndarray
    betas: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    expression: np.ndarray
    target_body_joints: np.ndarray
    fitted_body_joints: np.ndarray
    body_joint_error: np.ndarray
    metrics: FitMetrics
    bone_direction_error_deg: np.ndarray
    bone_direction_metrics: AngularMetrics


class BvhMotion:
    """Minimal BVH parser tailored to BONES-SEED-style motion files."""

    def __init__(
        self,
        path: Path,
        joint_names: Sequence[str],
        parents: Sequence[int],
        offsets_cm: np.ndarray,
        channels: Sequence[Sequence[str]],
        motion: np.ndarray,
        frame_time: float,
    ) -> None:
        self.path = Path(path)
        self.joint_names = list(joint_names)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.offsets_cm = np.asarray(offsets_cm, dtype=np.float32)
        self.channels = [list(ch) for ch in channels]
        self.motion = np.asarray(motion, dtype=np.float32)
        self.frame_time = float(frame_time)

    @classmethod
    def from_path(cls, path: str | Path) -> "BvhMotion":
        path = Path(path)
        lines = path.read_text().splitlines()

        joint_names: list[str] = []
        parents: list[int] = []
        offsets: list[np.ndarray | None] = []
        channels: list[list[str] | None] = []

        stack: list[int] = []
        active_joint: int | None = None
        pending_end_site = False
        skip_block_depth = 0
        motion_start = None

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue

            token = parts[0]
            if skip_block_depth > 0:
                if token == "{":
                    skip_block_depth += 1
                elif token == "}":
                    skip_block_depth -= 1
                continue

            if token in ("ROOT", "JOINT"):
                idx = len(joint_names)
                joint_names.append(parts[1])
                parents.append(stack[-1] if stack else -1)
                offsets.append(None)
                channels.append(None)
                active_joint = idx
            elif token == "End":
                pending_end_site = True
            elif token == "{":
                if pending_end_site:
                    skip_block_depth = 1
                    pending_end_site = False
                elif active_joint is not None:
                    stack.append(active_joint)
                    active_joint = None
            elif token == "}":
                if stack:
                    stack.pop()
            elif token == "OFFSET":
                offsets[stack[-1]] = np.asarray(parts[1:4], dtype=np.float32)
            elif token == "CHANNELS":
                count = int(parts[1])
                channels[stack[-1]] = parts[2 : 2 + count]
            elif token == "MOTION":
                motion_start = i
                break

        if motion_start is None:
            raise ValueError(f"BVH file has no MOTION section: {path}")

        missing_offsets = [name for name, offset in zip(joint_names, offsets) if offset is None]
        missing_channels = [name for name, ch in zip(joint_names, channels) if ch is None]
        if missing_offsets:
            raise ValueError(f"BVH file has joints without OFFSET: {missing_offsets}")
        if missing_channels:
            raise ValueError(f"BVH file has joints without CHANNELS: {missing_channels}")

        num_frames = int(lines[motion_start + 1].split(":")[1].strip())
        frame_time = float(lines[motion_start + 2].split(":")[1].strip())
        block = '\n'.join(lines[motion_start + 3 : motion_start + 3 + num_frames])
        motion = np.loadtxt(io.StringIO(block), dtype=np.float32)

        return cls(
            path=path,
            joint_names=joint_names,
            parents=parents,
            offsets_cm=np.stack(offsets),  # type: ignore[arg-type]
            channels=channels,  # type: ignore[arg-type]
            motion=motion,
            frame_time=frame_time,
        )

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    def select_frame_indices(
        self,
        *,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> np.ndarray:
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be > 0, got {frame_stride}")
        indices = np.arange(0, self.motion.shape[0], frame_stride, dtype=np.int64)
        if max_frames is not None:
            indices = indices[:max_frames]
        return indices

    def _frame_local_transforms(self, frame_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        frame_values = self.motion[np.asarray(frame_indices, dtype=np.int64)]
        batch_size = frame_values.shape[0]
        num_joints = len(self.joint_names)

        local_t = np.broadcast_to(self.offsets_cm[None], (batch_size, num_joints, 3)).copy()
        local_r = np.broadcast_to(np.eye(3, dtype=np.float32), (batch_size, num_joints, 3, 3)).copy()

        cursor = 0
        for joint_idx, channel_names in enumerate(self.channels):
            num_channels = len(channel_names)
            joint_data = frame_values[:, cursor : cursor + num_channels]
            cursor += num_channels

            has_position_channels = any(name.endswith("position") for name in channel_names)
            if has_position_channels:
                local_t[:, joint_idx] = 0.0
            for axis_idx, position_name in enumerate(("Xposition", "Yposition", "Zposition")):
                if position_name in channel_names:
                    local_t[:, joint_idx, axis_idx] = joint_data[:, channel_names.index(position_name)]

            rotation_channels = [name for name in channel_names if name.endswith("rotation")]
            if len(rotation_channels) != 3:
                raise ValueError(
                    f"Expected 3 rotation channels for joint {self.joint_names[joint_idx]}, got {channel_names}"
                )
            order = "".join(name[0].upper() for name in rotation_channels)
            angles = np.stack([joint_data[:, channel_names.index(name)] for name in rotation_channels], axis=1)
            local_r[:, joint_idx] = Rotation.from_euler(order, angles, degrees=True).as_matrix().astype(
                np.float32
            )

        return local_t, local_r

    def world_transforms(self, frame_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        local_t_cm, local_r = self._frame_local_transforms(frame_indices)
        batch_size, num_joints = local_t_cm.shape[:2]

        world_t_cm = np.zeros((batch_size, num_joints, 3), dtype=np.float32)
        world_r = np.zeros((batch_size, num_joints, 3, 3), dtype=np.float32)

        for joint_idx in range(num_joints):
            parent = self.parents[joint_idx]
            if parent < 0:
                world_t_cm[:, joint_idx] = local_t_cm[:, joint_idx]
                world_r[:, joint_idx] = local_r[:, joint_idx]
            else:
                world_t_cm[:, joint_idx] = world_t_cm[:, parent] + np.einsum(
                    "bij,bj->bi", world_r[:, parent], local_t_cm[:, joint_idx]
                )
                world_r[:, joint_idx] = world_r[:, parent] @ local_r[:, joint_idx]

        return world_t_cm / 100.0, world_r

    def body_joint_positions_m(self, frame_indices: Sequence[int]) -> np.ndarray:
        world_t_m, _ = self.world_transforms(frame_indices)
        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        return np.stack([world_t_m[:, name_to_idx[name], :] for name in BVH_BODY_JOINT_NAMES], axis=1)

    def body_joint_transforms(self, frame_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return body joint positions (B, 22, 3) and world rotations (B, 22, 3, 3)."""
        world_t_m, world_r = self.world_transforms(frame_indices)
        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        indices = [name_to_idx[name] for name in BVH_BODY_JOINT_NAMES]
        positions = np.stack([world_t_m[:, i] for i in indices], axis=1)
        rotations = np.stack([world_r[:, i] for i in indices], axis=1)
        return positions, rotations

    def find_tpose_frame(self) -> int:
        """Find the frame closest to a standing T-pose with arms held horizontally."""

        all_frames = np.arange(self.motion.shape[0], dtype=np.int64)
        world_t_m, _ = self.world_transforms(all_frames)
        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}

        hips = world_t_m[:, name_to_idx["Hips"]]
        head = world_t_m[:, name_to_idx["Head"]]
        left_shoulder = world_t_m[:, name_to_idx["LeftArm"]]
        right_shoulder = world_t_m[:, name_to_idx["RightArm"]]
        left_elbow = world_t_m[:, name_to_idx["LeftForeArm"]]
        right_elbow = world_t_m[:, name_to_idx["RightForeArm"]]
        left_hand = world_t_m[:, name_to_idx["LeftHand"]]
        right_hand = world_t_m[:, name_to_idx["RightHand"]]

        standing = head[:, 1] > hips[:, 1]
        arm_spread = np.linalg.norm(left_hand[:, [0, 2]] - right_hand[:, [0, 2]], axis=1)
        elbow_height_error = np.abs(left_elbow[:, 1] - left_shoulder[:, 1]) + np.abs(
            right_elbow[:, 1] - right_shoulder[:, 1]
        )
        hand_height_error = np.abs(left_hand[:, 1] - left_shoulder[:, 1]) + np.abs(
            right_hand[:, 1] - right_shoulder[:, 1]
        )

        score = arm_spread - 1.5 * elbow_height_error - 2.0 * hand_height_error
        if standing.any():
            score[~standing] = -np.inf

        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            raise ValueError(f"Failed to detect a standing T-pose frame in BVH: {self.path}")
        return best


def _create_smplx_model(model_path: str | Path, batch_size: int, device: torch.device) -> torch.nn.Module:
    model = smplx.create(
        model_type="smplx",
        model_path=str(model_path),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=batch_size,
    )
    return model.to(device)


def _run_smplx_body(
    model: torch.nn.Module,
    *,
    betas: torch.Tensor,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    _zero_cache: dict | None = None,
) -> torch.Tensor:
    batch_size = body_pose.shape[0]
    device = body_pose.device
    cache_key = (batch_size, device)

    if _zero_cache is not None and cache_key in _zero_cache:
        z45, z3, z10 = _zero_cache[cache_key]
    else:
        z45 = torch.zeros(batch_size, 45, dtype=torch.float32, device=device)
        z3 = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        z10 = torch.zeros(batch_size, 10, dtype=torch.float32, device=device)
        if _zero_cache is not None:
            _zero_cache[cache_key] = (z45, z3, z10)

    out = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        left_hand_pose=z45,
        right_hand_pose=z45,
        jaw_pose=z3,
        leye_pose=z3,
        reye_pose=z3,
        expression=z10,
        return_verts=False,
    )
    return out.joints[:, : len(SMPLX_BODY_JOINT_NAMES)]


def compute_fit_metrics(
    target_body_joints_m: np.ndarray | torch.Tensor,
    fitted_body_joints_m: np.ndarray | torch.Tensor,
) -> tuple[FitMetrics, np.ndarray]:
    """Compute 3D joint-position error in meters.

    This metric is affected by both pose mismatch and skeleton-size mismatch.
    """

    target = torch.as_tensor(target_body_joints_m, dtype=torch.float32)
    fitted = torch.as_tensor(fitted_body_joints_m, dtype=torch.float32)
    error = torch.norm(fitted - target, dim=-1)
    metrics = FitMetrics(
        mean_error_m=float(error.mean()),
        p95_error_m=float(torch.quantile(error.flatten(), 0.95)),
        max_error_m=float(error.max()),
    )
    return metrics, error.cpu().numpy()


def compute_bone_direction_metrics(
    target_body_joints_m: np.ndarray | torch.Tensor,
    fitted_body_joints_m: np.ndarray | torch.Tensor,
    parents_list: Sequence[int],
) -> tuple[AngularMetrics, np.ndarray]:
    """Compute parent->child bone direction error in degrees.

    This mostly reflects pose-direction mismatch and is much less sensitive to
    skeleton-size differences than raw 3D joint-position error.

    The per-joint array covers all non-root body joints. The summary metrics
    exclude `left/right_collar` and `left/right_foot`, because those joints have
    the largest semantic-definition mismatch between BONES BVH and SMPL-X.
    """

    target = torch.as_tensor(target_body_joints_m, dtype=torch.float32)
    fitted = torch.as_tensor(fitted_body_joints_m, dtype=torch.float32)

    error_deg = torch.zeros(target.shape[:2], dtype=torch.float32)
    valid_columns: list[torch.Tensor] = []

    for joint_idx, parent_idx in enumerate(parents_list):
        if parent_idx < 0:
            continue
        target_vec = target[:, joint_idx] - target[:, parent_idx]
        fitted_vec = fitted[:, joint_idx] - fitted[:, parent_idx]

        target_norm = torch.linalg.norm(target_vec, dim=-1, keepdim=True).clamp_min(1e-8)
        fitted_norm = torch.linalg.norm(fitted_vec, dim=-1, keepdim=True).clamp_min(1e-8)
        target_dir = target_vec / target_norm
        fitted_dir = fitted_vec / fitted_norm
        cosine = torch.clamp((target_dir * fitted_dir).sum(dim=-1), -1.0, 1.0)
        joint_error = torch.rad2deg(torch.arccos(cosine))
        error_deg[:, joint_idx] = joint_error
        joint_name = SMPLX_BODY_JOINT_NAMES[joint_idx]
        if joint_name not in BONE_DIRECTION_METRIC_EXCLUDED_JOINT_NAMES:
            valid_columns.append(joint_error)

    if not valid_columns:
        raise ValueError("No valid non-root joints found for bone-direction metrics.")

    flat_error = torch.cat(valid_columns, dim=0)
    metrics = AngularMetrics(
        mean_error_deg=float(flat_error.mean()),
        p95_error_deg=float(torch.quantile(flat_error, 0.95)),
        max_error_deg=float(flat_error.max()),
    )
    return metrics, error_deg.cpu().numpy()


def _transform_points_y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    return points @ Y_UP_TO_Z_UP_ROTATION_MATRIX.T


def _unwrap_rotvec(rotvecs: np.ndarray) -> np.ndarray:
    """Ensure temporal continuity of a rotation-vector sequence.

    Converts to quaternions, enforces hemisphere continuity (flip q when
    dot(q_i, q_{i-1}) < 0), then converts back to rotation vectors using a
    custom formula that allows angles beyond π — producing a smooth sequence
    free of the axis-angle π singularity.
    """
    quats = Rotation.from_rotvec(rotvecs).as_quat(canonical=False)  # (N, 4) xyzw
    # Hemisphere continuity: keep each q on the same side as its predecessor
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    # Custom quat→rotvec preserving hemisphere (allows angle > π)
    # scipy xyzw convention: q = [x, y, z, w]
    xyz = quats[:, :3]
    w = quats[:, 3]
    sin_half = np.linalg.norm(xyz, axis=1, keepdims=True)
    half_angle = np.arctan2(sin_half, w[:, None])  # ∈ (-π, π], preserves sign
    angle = 2.0 * half_angle  # full rotation angle, can exceed π
    # axis = xyz / sin(half_angle), avoid division by zero
    safe = (sin_half > 1e-8).squeeze()
    out = np.zeros_like(rotvecs)
    out[safe] = (angle[safe] / sin_half[safe]) * xyz[safe]
    return out.astype(np.float32)


def _transform_root_orient_y_up_to_z_up(root_orient: np.ndarray) -> np.ndarray:
    root_rot_mats = Rotation.from_rotvec(root_orient).as_matrix()
    transformed = Y_UP_TO_Z_UP_ROTATION_MATRIX[None] @ root_rot_mats
    rotvecs = Rotation.from_matrix(transformed).as_rotvec().astype(np.float32)
    return _unwrap_rotvec(rotvecs)


def _normalize_betas(num_betas: np.ndarray | Sequence[float], target_num_betas: int = 10) -> np.ndarray:
    betas = np.asarray(num_betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] >= target_num_betas:
        return betas[:target_num_betas]
    return np.pad(betas, (0, target_num_betas - betas.shape[0]))


def _run_smplx_body_numpy(
    *,
    model_path: str | Path,
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    body_pose = np.asarray(body_pose, dtype=np.float32)
    global_orient = np.asarray(global_orient, dtype=np.float32)
    transl = np.asarray(transl, dtype=np.float32)

    num_frames = body_pose.shape[0]
    model = _create_smplx_model(model_path, batch_size=num_frames, device=device)
    betas_batch = torch.from_numpy(_normalize_betas(betas)[None]).to(device).expand(num_frames, -1)

    with torch.no_grad():
        joints = _run_smplx_body(
            model,
            betas=betas_batch,
            body_pose=torch.from_numpy(body_pose).to(device),
            global_orient=torch.from_numpy(global_orient).to(device),
            transl=torch.from_numpy(transl).to(device),
        )
    return joints.cpu().numpy().astype(np.float32)


def load_smplx_body_joints_from_npz(
    npz_path: str | Path,
    *,
    model_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.load(npz_path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}

    needs_post_fk_y_up_to_z_up = False
    if "pose_body" in data and "root_orient" in data and "trans" in data:
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
    elif "poses" in data and "trans" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 66:
            raise ValueError(f"`poses` must have shape (N, >=66), got {poses.shape}")
        pose_body = poses[:, 3:66]
        root_orient = poses[:, :3]
        trans = np.asarray(data["trans"], dtype=np.float32)
        needs_post_fk_y_up_to_z_up = True
    else:
        raise ValueError(
            "Unsupported NPZ schema: expected (pose_body, root_orient, trans) or (poses, trans)."
        )

    joints = _run_smplx_body_numpy(
        model_path=model_path,
        betas=np.asarray(data.get("betas", np.zeros(10, dtype=np.float32)), dtype=np.float32),
        body_pose=pose_body,
        global_orient=root_orient,
        transl=trans,
        device=device,
    )

    coord_system = str(data.get("coord_system", "unknown"))
    if needs_post_fk_y_up_to_z_up:
        joints = _transform_points_y_up_to_z_up(joints)
        coord_system = "z_up"

    meta: dict[str, object] = {
        "coord_system": coord_system,
        "frame_indices": np.asarray(
            data.get("frame_indices", np.arange(pose_body.shape[0], dtype=np.int32)),
            dtype=np.int32,
        ),
        "fps": float(np.asarray(data.get("mocap_frame_rate", data.get("fps", 30.0)), dtype=np.float32)),
        "source_path": str(data.get("source_path", "")),
        "stored_fitted_body_joints": (
            np.asarray(data["fitted_body_joints"], dtype=np.float32) if "fitted_body_joints" in data else None
        ),
        "stored_target_body_joints": (
            np.asarray(data["target_body_joints"], dtype=np.float32) if "target_body_joints" in data else None
        ),
    }
    return joints.astype(np.float32), meta


def _body_joint_indices(motion: BvhMotion) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
    return [name_to_idx[name] for name in BVH_BODY_JOINT_NAMES]


def _body_local_rotations(motion: BvhMotion, frame_indices: Sequence[int]) -> np.ndarray:
    _, local_r = motion._frame_local_transforms(frame_indices)
    return local_r[:, _body_joint_indices(motion)].astype(np.float32)


def _resolve_reference_frame(motion: BvhMotion, frame_index: int | None) -> int:
    if frame_index is None:
        try:
            frame_index = motion.find_tpose_frame()
        except ValueError:
            all_frames = np.arange(motion.motion.shape[0], dtype=np.int64)
            world_t_m, _ = motion.world_transforms(all_frames)
            name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
            hips = world_t_m[:, name_to_idx["Hips"]]
            head = world_t_m[:, name_to_idx["Head"]]
            left_arm = world_t_m[:, name_to_idx["LeftArm"]]
            right_arm = world_t_m[:, name_to_idx["RightArm"]]
            standing_height = head[:, 1] - hips[:, 1]
            shoulder_width = np.linalg.norm(left_arm - right_arm, axis=1)
            frame_index = int(np.argmax(standing_height + 0.1 * shoulder_width))
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= motion.motion.shape[0]:
        raise ValueError(
            f"reference frame must be in [0, {motion.motion.shape[0] - 1}], got {frame_index}"
        )
    return frame_index


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return (vector / norm).astype(np.float32)


def _project_orthogonal(vector: np.ndarray, primary: np.ndarray) -> np.ndarray:
    return vector - primary * float(np.dot(vector, primary))


def _choose_local_secondary_hint(local_primary: np.ndarray) -> np.ndarray:
    local_primary = np.asarray(local_primary, dtype=np.float32)
    for candidate in (
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ):
        projected = _project_orthogonal(candidate, _normalize_vector(local_primary))
        if np.linalg.norm(projected) > 1e-6:
            return candidate
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _fallback_perpendicular(primary: np.ndarray) -> np.ndarray:
    primary = _normalize_vector(primary)
    for candidate in (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ):
        projected = _project_orthogonal(candidate, primary)
        if np.linalg.norm(projected) > 1e-6:
            return _normalize_vector(projected)
    raise ValueError("Failed to build a perpendicular basis vector.")


def _basis_from_primary_and_secondary(primary: np.ndarray, secondary_hint: np.ndarray) -> np.ndarray:
    x_axis = _normalize_vector(primary)
    y_axis = _project_orthogonal(np.asarray(secondary_hint, dtype=np.float32), x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = _fallback_perpendicular(x_axis)
    else:
        y_axis = _normalize_vector(y_axis)
    z_axis = _normalize_vector(np.cross(x_axis, y_axis))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def _rotation_from_primary_and_secondary(
    *,
    local_primary: np.ndarray,
    target_primary: np.ndarray,
    local_secondary_hint: np.ndarray | None = None,
    target_secondary_hint: np.ndarray | None = None,
    reference_world: np.ndarray | None = None,
) -> np.ndarray:
    if local_secondary_hint is None:
        local_secondary_hint = _choose_local_secondary_hint(local_primary)
    if target_secondary_hint is None:
        if reference_world is None:
            raise ValueError("reference_world is required when target_secondary_hint is not provided.")
        target_secondary_hint = np.asarray(reference_world, dtype=np.float32) @ np.asarray(
            local_secondary_hint, dtype=np.float32
        )
    local_basis = _basis_from_primary_and_secondary(local_primary, local_secondary_hint)
    target_basis = _basis_from_primary_and_secondary(target_primary, target_secondary_hint)
    return (target_basis @ local_basis.T).astype(np.float32)


def _reference_body_axes(motion: BvhMotion, reference_frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, _ = motion.body_joint_transforms([reference_frame])
    positions = positions[0]
    name_to_idx = {name: i for i, name in enumerate(BVH_BODY_JOINT_NAMES)}

    hips = positions[name_to_idx["Hips"]]
    head = positions[name_to_idx["Head"]]
    left_arm = positions[name_to_idx["LeftArm"]]
    right_arm = positions[name_to_idx["RightArm"]]
    left_leg = positions[name_to_idx["LeftLeg"]]
    right_leg = positions[name_to_idx["RightLeg"]]

    up_axis = _normalize_vector(head - hips)
    left_axis_hint = left_arm - right_arm
    left_axis_hint = _project_orthogonal(left_axis_hint, up_axis)
    if np.linalg.norm(left_axis_hint) < 1e-6:
        left_axis_hint = _project_orthogonal(left_leg - right_leg, up_axis)
    left_axis = _normalize_vector(left_axis_hint)
    forward_axis = _normalize_vector(np.cross(up_axis, left_axis))
    left_axis = _normalize_vector(np.cross(forward_axis, up_axis))
    return up_axis, left_axis, forward_axis


def _synthetic_bvh_tpose_world_orientations(motion: BvhMotion, reference_frame: int) -> np.ndarray:
    _, reference_world = motion.body_joint_transforms([reference_frame])
    reference_world = reference_world[0].astype(np.float32)
    up_axis, left_axis, forward_axis = _reference_body_axes(motion, reference_frame)
    right_axis = (-left_axis).astype(np.float32)
    down_axis = (-up_axis).astype(np.float32)

    body_name_to_idx = {name: i for i, name in enumerate(BVH_BODY_JOINT_NAMES)}
    full_name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
    offsets = motion.offsets_cm.astype(np.float32)

    synthetic_world = reference_world.copy()

    synthetic_world[body_name_to_idx["Hips"]] = _rotation_from_primary_and_secondary(
        local_primary=offsets[full_name_to_idx["Spine1"]],
        target_primary=up_axis,
        local_secondary_hint=offsets[full_name_to_idx["LeftLeg"]] - offsets[full_name_to_idx["RightLeg"]],
        target_secondary_hint=left_axis,
    )
    synthetic_world[body_name_to_idx["Chest"]] = _rotation_from_primary_and_secondary(
        local_primary=offsets[full_name_to_idx["Neck1"]],
        target_primary=up_axis,
        local_secondary_hint=offsets[full_name_to_idx["LeftShoulder"]]
        - offsets[full_name_to_idx["RightShoulder"]],
        target_secondary_hint=left_axis,
    )

    single_child_targets: list[tuple[str, str, np.ndarray]] = [
        ("Spine1", "Spine2", up_axis),
        ("Spine2", "Chest", up_axis),
        ("Neck1", "Neck2", up_axis),
        ("LeftShoulder", "LeftArm", left_axis),
        ("LeftArm", "LeftForeArm", left_axis),
        ("LeftForeArm", "LeftHand", left_axis),
        ("RightShoulder", "RightArm", right_axis),
        ("RightArm", "RightForeArm", right_axis),
        ("RightForeArm", "RightHand", right_axis),
        ("LeftLeg", "LeftShin", down_axis),
        ("LeftShin", "LeftFoot", down_axis),
        ("LeftFoot", "LeftToeBase", forward_axis),
        ("RightLeg", "RightShin", down_axis),
        ("RightShin", "RightFoot", down_axis),
        ("RightFoot", "RightToeBase", forward_axis),
    ]
    for joint_name, child_name, target_primary in single_child_targets:
        joint_idx = body_name_to_idx[joint_name]
        synthetic_world[joint_idx] = _rotation_from_primary_and_secondary(
            local_primary=offsets[full_name_to_idx[child_name]],
            target_primary=target_primary,
            reference_world=reference_world[joint_idx],
        )

    return synthetic_world.astype(np.float32)


def _bvh_tpose_world_orientations(
    motion: BvhMotion, calibration_frame: int | None
) -> tuple[np.ndarray, int]:
    """Return BVH T-pose joint frames plus the reference frame index used.

    If `calibration_frame` is provided, that exact BVH frame is treated as the
    calibration T-pose. Otherwise we estimate body axes from a reference frame
    and construct a synthetic upright T-pose for calibration.
    """

    reference_frame = _resolve_reference_frame(motion, calibration_frame)
    if calibration_frame is None:
        return _synthetic_bvh_tpose_world_orientations(motion, reference_frame), reference_frame
    _, calibration_world_r = motion.body_joint_transforms([reference_frame])
    return calibration_world_r[0].astype(np.float32), reference_frame


def _validate_calibration_motion_compatibility(
    motion: BvhMotion,
    calibration_motion: BvhMotion,
) -> None:
    if motion.joint_names != calibration_motion.joint_names:
        raise ValueError("Calibration BVH joint names do not match the input BVH.")
    if motion.parents.shape != calibration_motion.parents.shape or not np.array_equal(
        motion.parents, calibration_motion.parents
    ):
        raise ValueError("Calibration BVH hierarchy does not match the input BVH.")
    if motion.offsets_cm.shape != calibration_motion.offsets_cm.shape or not np.allclose(
        motion.offsets_cm, calibration_motion.offsets_cm, atol=1e-4
    ):
        raise ValueError("Calibration BVH offsets do not match the input BVH skeleton.")
    if motion.channels != calibration_motion.channels:
        raise ValueError("Calibration BVH channel layout does not match the input BVH.")


def _smplx_tpose_world_orientations(parents_list: Sequence[int]) -> np.ndarray:
    """World orientations of SMPL-X joints in canonical zero/T-pose.

    In standard SMPL-X parameter space, zero local rotations define the canonical
    T-pose. Since every local rotation is identity, the corresponding world
    joint frames are also identity through the whole parent chain.
    """

    num_joints = len(parents_list)
    local_identity = np.broadcast_to(np.eye(3, dtype=np.float32), (num_joints, 3, 3)).copy()
    world_identity = np.zeros_like(local_identity)
    for joint_idx, parent_idx in enumerate(parents_list):
        if parent_idx < 0:
            world_identity[joint_idx] = local_identity[joint_idx]
        else:
            world_identity[joint_idx] = world_identity[parent_idx] @ local_identity[joint_idx]
    return world_identity


def _tpose_offset_bvh_to_smplx(
    bvh_tpose_world: np.ndarray,
    smplx_tpose_world: np.ndarray,
) -> np.ndarray:
    """Right-multiplied offset that maps a BVH T-pose joint frame into the SMPL-X T-pose joint frame."""

    return np.einsum("jki,jkl->jil", bvh_tpose_world, smplx_tpose_world)


def _legacy_calibrated_bvh_local_to_smplx_local(
    motion: BvhMotion,
    *,
    frame_indices: Sequence[int],
    parents_list: Sequence[int],
    calibration_frame: int | None = None,
    calibration_motion: BvhMotion | None = None,
) -> tuple[np.ndarray, int]:
    """Reference implementation used to verify the explicit T-pose pipeline.

    This matches the previous implicit formula:
        R_smplx_local = O_bvh_parent_tpose @ R_bvh_local @ O_bvh_joint_tpose^T
    """

    body_local_r = _body_local_rotations(motion, frame_indices)
    calibration_motion = motion if calibration_motion is None else calibration_motion
    bvh_tpose_world, calibration_reference_frame = _bvh_tpose_world_orientations(
        calibration_motion, calibration_frame
    )

    smplx_local = np.zeros_like(body_local_r)
    identity = np.eye(3, dtype=np.float32)
    for joint_idx, parent_idx in enumerate(parents_list):
        orient_parent = identity if parent_idx < 0 else bvh_tpose_world[parent_idx]
        smplx_local[:, joint_idx] = (
            orient_parent[None] @ body_local_r[:, joint_idx] @ bvh_tpose_world[joint_idx].T[None]
        )

    return smplx_local, calibration_reference_frame


def _explicit_tpose_bvh_to_smplx_local(
    motion: BvhMotion,
    *,
    frame_indices: Sequence[int],
    parents_list: Sequence[int],
    calibration_frame: int | None = None,
    calibration_motion: BvhMotion | None = None,
) -> tuple[np.ndarray, int]:
    """Convert BVH local rotations to SMPL-X locals via explicit T-pose offsets.

    Steps:
    1. Use either a real BVH calibration frame or a synthesized BVH standard T-pose.
    2. Record BVH joint frames at that T-pose.
    3. Record SMPL-X joint frames at canonical zero/T-pose.
    4. Compute the T-pose joint-frame offset from BVH to SMPL-X.
    5. For each frame, read the original BVH local rotation.
    6. Use the parent/joint T-pose offsets to reinterpret that local motion in the SMPL-X joint frames.
    """

    # Keep the original BVH local channels here. Some BVHs have extra intermediary joints
    # (for example Neck2), so reconstructing locals from the reduced 22-joint world chain
    # would change the motion definition for joints like Head.
    body_local_r = _body_local_rotations(motion, frame_indices)
    calibration_motion = motion if calibration_motion is None else calibration_motion
    bvh_tpose_world, calibration_reference_frame = _bvh_tpose_world_orientations(
        calibration_motion, calibration_frame
    )
    smplx_tpose_world = _smplx_tpose_world_orientations(parents_list)
    joint_offset = _tpose_offset_bvh_to_smplx(bvh_tpose_world, smplx_tpose_world)

    smplx_local = np.zeros_like(body_local_r)
    identity = np.eye(3, dtype=np.float32)
    for joint_idx, parent_idx in enumerate(parents_list):
        parent_offset_t = identity if parent_idx < 0 else joint_offset[parent_idx].T
        smplx_local[:, joint_idx] = parent_offset_t[None] @ body_local_r[:, joint_idx] @ joint_offset[joint_idx][
            None
        ]
    return smplx_local, calibration_reference_frame


def convert_bvh_to_smplx_direct(
    bvh_path: str | Path,
    *,
    model_path: str | Path,
    device: str | torch.device = "cpu",
    frame_stride: int = 1,
    max_frames: int | None = None,
    calibration_frame: int | None = None,
    calibration_bvh_path: str | Path | None = None,
    calibration_bvh_frame: int | None = None,
) -> ConversionResult:
    """Direct BVH→SMPLX using either a real or synthesized BVH T-pose calibration.

    The motion always comes from `bvh_path`. If `calibration_bvh_path` is given,
    the calibration pose is read from that separate BVH instead.
    """
    motion = BvhMotion.from_path(bvh_path)
    calibration_motion = motion if calibration_bvh_path is None else BvhMotion.from_path(calibration_bvh_path)
    _validate_calibration_motion_compatibility(motion, calibration_motion)
    effective_calibration_frame = calibration_frame
    if calibration_bvh_path is not None:
        effective_calibration_frame = (
            calibration_bvh_frame if calibration_bvh_frame is not None else calibration_frame
        )
    frame_indices = motion.select_frame_indices(frame_stride=frame_stride, max_frames=max_frames)
    num_frames = len(frame_indices)

    device_t = torch.device(device)
    model = _create_smplx_model(model_path, batch_size=1, device=device_t)
    parents_list = model.parents[: len(SMPLX_BODY_JOINT_NAMES)].tolist()
    betas_np = np.zeros(model.num_betas, dtype=np.float32)

    target_body_joints = motion.body_joint_positions_m(frame_indices)
    smplx_local, _ = _explicit_tpose_bvh_to_smplx_local(
        motion,
        frame_indices=frame_indices,
        parents_list=parents_list,
        calibration_frame=effective_calibration_frame,
        calibration_motion=calibration_motion,
    )

    # Convert to axis-angle
    global_orient_rv = Rotation.from_matrix(
        smplx_local[:, 0].reshape(-1, 3, 3)
    ).as_rotvec().astype(np.float32).reshape(num_frames, 3)
    global_orient_rv = _unwrap_rotvec(global_orient_rv)

    body_pose_rv = Rotation.from_matrix(
        smplx_local[:, 1:].reshape(-1, 3, 3)
    ).as_rotvec().astype(np.float32).reshape(num_frames, 21, 3)
    for j in range(21):
        body_pose_rv[:, j] = _unwrap_rotvec(body_pose_rv[:, j])
    body_pose_rv = body_pose_rv.reshape(num_frames, 63)

    # Translation: match BVH pelvis world position
    bvh_pelvis = target_body_joints[:, 0]
    with torch.no_grad():
        z = torch.zeros
        out = model(
            betas=z(1, model.num_betas, dtype=torch.float32, device=device_t),
            body_pose=z(1, 63, dtype=torch.float32, device=device_t),
            global_orient=z(1, 3, dtype=torch.float32, device=device_t),
            transl=z(1, 3, dtype=torch.float32, device=device_t),
            left_hand_pose=z(1, 45, dtype=torch.float32, device=device_t),
            right_hand_pose=z(1, 45, dtype=torch.float32, device=device_t),
            jaw_pose=z(1, 3, dtype=torch.float32, device=device_t),
            leye_pose=z(1, 3, dtype=torch.float32, device=device_t),
            reye_pose=z(1, 3, dtype=torch.float32, device=device_t),
            expression=z(1, 10, dtype=torch.float32, device=device_t),
            return_verts=False,
        )
        rest_pelvis = out.joints[0, 0].cpu().numpy()
    go_mats = Rotation.from_rotvec(global_orient_rv).as_matrix().astype(np.float32)
    transl_np = bvh_pelvis - np.einsum("bij,j->bi", go_mats, rest_pelvis)

    # Compute fitted joints for metrics
    fitted_body_np = _run_smplx_body_numpy(
        model_path=model_path,
        betas=betas_np,
        body_pose=body_pose_rv,
        global_orient=global_orient_rv,
        transl=transl_np,
        device=device,
    )
    metrics, body_joint_error = compute_fit_metrics(target_body_joints, fitted_body_np)
    bone_direction_metrics, bone_direction_error_deg = compute_bone_direction_metrics(
        target_body_joints,
        fitted_body_np,
        parents_list,
    )

    z_np = np.zeros
    return ConversionResult(
        source_path=Path(bvh_path),
        fps=motion.fps / frame_stride,
        frame_indices=frame_indices.astype(np.int32),
        betas=betas_np,
        global_orient=global_orient_rv,
        body_pose=body_pose_rv,
        transl=transl_np.astype(np.float32),
        left_hand_pose=z_np((num_frames, 45), dtype=np.float32),
        right_hand_pose=z_np((num_frames, 45), dtype=np.float32),
        jaw_pose=z_np((num_frames, 3), dtype=np.float32),
        leye_pose=z_np((num_frames, 3), dtype=np.float32),
        reye_pose=z_np((num_frames, 3), dtype=np.float32),
        expression=z_np((num_frames, 10), dtype=np.float32),
        target_body_joints=target_body_joints.astype(np.float32),
        fitted_body_joints=fitted_body_np.astype(np.float32),
        body_joint_error=body_joint_error.astype(np.float32),
        metrics=metrics,
        bone_direction_error_deg=bone_direction_error_deg.astype(np.float32),
        bone_direction_metrics=bone_direction_metrics,
    )


def save_conversion_result(
    result: ConversionResult,
    out_path: str | Path,
    *,
    gender: str = "neutral",
    export_z_up: bool = True,
    model_path: str | Path | None = None,
    export_refine_iters: int = 0,
    export_refine_lr: float = 1e-2,
    export_refine_chunk_size: int = 128,
    device: str | torch.device = "cpu",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    betas_16 = np.pad(result.betas.astype(np.float32), (0, max(0, 16 - result.betas.shape[0])))[:16]

    if export_z_up:
        root_orient = _transform_root_orient_y_up_to_z_up(result.global_orient.astype(np.float32))
        trans = _transform_points_y_up_to_z_up(result.transl.astype(np.float32))
        target_body_joints = _transform_points_y_up_to_z_up(result.target_body_joints.astype(np.float32))
        fitted_body_joints = _transform_points_y_up_to_z_up(result.fitted_body_joints.astype(np.float32))
        if model_path is not None:
            fitted_body_joints = _run_smplx_body_numpy(
                model_path=model_path,
                betas=result.betas.astype(np.float32),
                body_pose=result.body_pose.astype(np.float32),
                global_orient=root_orient,
                transl=trans,
                device=device,
            )
    else:
        root_orient = result.global_orient.astype(np.float32)
        trans = result.transl.astype(np.float32)
        target_body_joints = result.target_body_joints.astype(np.float32)
        fitted_body_joints = result.fitted_body_joints.astype(np.float32)

    pose_body = result.body_pose.astype(np.float32)
    pose_jaw = result.jaw_pose.astype(np.float32)
    pose_eye = np.concatenate([result.leye_pose, result.reye_pose], axis=1).astype(np.float32)
    pose_hand = np.concatenate([result.left_hand_pose, result.right_hand_pose], axis=1).astype(np.float32)
    poses = np.concatenate([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], axis=1).astype(
        np.float32
    )
    mocap_frame_rate = np.array(result.fps, dtype=np.float32)
    mocap_time_length = np.array(result.body_pose.shape[0] / result.fps, dtype=np.float32)

    np.savez_compressed(
        out_path,
        source_path=np.array(str(result.source_path)),
        fps=mocap_frame_rate,
        frame_indices=result.frame_indices,
        betas=betas_16,
        num_betas=np.array(16, dtype=np.int64),
        gender=np.array(gender),
        surface_model_type=np.array("smplx"),
        mocap_frame_rate=mocap_frame_rate,
        mocap_time_length=mocap_time_length,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        pose_hand=pose_hand,
        poses=poses,
        global_orient=root_orient,
        body_pose=pose_body,
        transl=trans,
        left_hand_pose=result.left_hand_pose,
        right_hand_pose=result.right_hand_pose,
        jaw_pose=result.jaw_pose,
        leye_pose=result.leye_pose,
        reye_pose=result.reye_pose,
        expression=result.expression,
        bvh_body_joint_names=np.asarray(BVH_BODY_JOINT_NAMES),
        smplx_body_joint_names=np.asarray(SMPLX_BODY_JOINT_NAMES),
        target_body_joints=target_body_joints,
        fitted_body_joints=fitted_body_joints,
        body_joint_position_error=result.body_joint_error,
        body_joint_position_mean_m=np.array(result.metrics.mean_error_m, dtype=np.float32),
        body_joint_position_p95_m=np.array(result.metrics.p95_error_m, dtype=np.float32),
        body_joint_position_max_m=np.array(result.metrics.max_error_m, dtype=np.float32),
        bone_direction_error_deg=result.bone_direction_error_deg,
        bone_direction_metric_excluded_joint_names=np.asarray(
            sorted(BONE_DIRECTION_METRIC_EXCLUDED_JOINT_NAMES)
        ),
        bone_direction_mean_deg=np.array(result.bone_direction_metrics.mean_error_deg, dtype=np.float32),
        bone_direction_p95_deg=np.array(result.bone_direction_metrics.p95_error_deg, dtype=np.float32),
        bone_direction_max_deg=np.array(result.bone_direction_metrics.max_error_deg, dtype=np.float32),
        body_joint_error=result.body_joint_error,
        mean_error_m=np.array(result.metrics.mean_error_m, dtype=np.float32),
        p95_error_m=np.array(result.metrics.p95_error_m, dtype=np.float32),
        max_error_m=np.array(result.metrics.max_error_m, dtype=np.float32),
        coord_system=np.array("z_up" if export_z_up else "y_up"),
    )


def iter_bvh_files(input_path: str | Path) -> Iterable[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    yield from sorted(input_path.rglob("*.bvh"))


def relative_output_path(source_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root
    rel = source_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".npz")
