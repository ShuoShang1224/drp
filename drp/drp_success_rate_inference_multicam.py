import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from drp.drp_inference import DRPInference
from drp.drp_success_rate_inference import (
    ObstacleScene,
    RolloutVisualizer,
    TrialResult,
    _build_scene_obstacle_pcd,
    _choose_start_goal,
    _eef_error_m,
    _infer_scene_prefix_from_dataset,
    _joint_error_l1,
    _load_scene_obstacle_pcd_from_usd,
    _parse_float_list,
    _resolve_device,
    _scene_clearance_m,
    _summarize,
)
from drp.utils.franka_utils import clamp_to_franka_limits

try:
    import h5py
except ImportError:
    h5py = None

try:
    import geometrout.transform as _geometrout_transform

    _BaseSE3 = _geometrout_transform.SE3
    _SO3 = _geometrout_transform.SO3

    class CompatSE3(_BaseSE3):
        def __init__(self, pos=None, quaternion=None, xyz=None, rpy=None, matrix=None):
            if matrix is not None:
                pose = _BaseSE3.from_matrix(np.asarray(matrix, dtype=np.float64))
                self.pos = pose.pos
                self.so3 = pose.so3
                return
            if xyz is not None:
                pos = xyz
            if pos is None:
                raise TypeError("SE3 requires either pos=/xyz= or matrix=")
            if rpy is not None:
                quaternion = _SO3.from_rpy(*np.asarray(rpy, dtype=np.float64)).q
            if quaternion is None:
                raise TypeError("SE3 requires quaternion= or rpy= when matrix= is not provided")
            super().__init__(
                np.asarray(pos, dtype=np.float64),
                np.asarray(quaternion, dtype=np.float64),
            )

        def __matmul__(self, other):
            return self.__mul__(other)

        def __rmatmul__(self, other):
            return other.__mul__(self)

    _geometrout_transform.SE3 = CompatSE3

    from geometrout.primitive import Cuboid
    from geometrout.transform import SE3
    from robofin.bullet import Bullet
    from robofin.robots import FrankaRobot
except Exception:
    Cuboid = None
    SE3 = None
    Bullet = None
    FrankaRobot = None


DEFAULT_OBSTACLE_POINTS = 2048
DEFAULT_COLLISION_ROBOT_POINTS = 1024
DEFAULT_CAMERA_SOURCE_POINTS = 8192
DEFAULT_D435_DEPTH_WIDTH = 848
DEFAULT_D435_DEPTH_HEIGHT = 480
DEFAULT_D435_DEPTH_HFOV_DEG = 87.0
DEFAULT_D435_DEPTH_NEAR_M = 0.01
DEFAULT_D435_DEPTH_FAR_M = 10.0


class FixedCamera:
    def __init__(
        self,
        position: np.ndarray,
        target: np.ndarray,
        width: int,
        height: int,
        hfov_deg: float,
        near: float,
        far: float,
    ):
        self.position = np.asarray(position, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.width = int(width)
        self.height = int(height)
        self.hfov_deg = float(hfov_deg)
        self.near = float(near)
        self.far = float(far)


def _camera_intrinsics(camera: FixedCamera) -> Tuple[float, float, float, float]:
    fx = 0.5 * camera.width / np.tan(0.5 * np.deg2rad(camera.hfov_deg))
    vfov = 2.0 * np.arctan(camera.height / (2.0 * fx))
    fy = 0.5 * camera.height / np.tan(0.5 * vfov)
    cx = 0.5 * (camera.width - 1)
    cy = 0.5 * (camera.height - 1)
    return float(fx), float(fy), float(cx), float(cy)


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return quat


def _make_fixed_cameras(
    count: int,
    target: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float,
    near: float,
    far: float,
) -> List[FixedCamera]:
    # Default layout keeps the primary cameras a bit lower while also pulling
    # them back so the view is flatter and less top-down over the workspace.
    # Two additional cameras remain available for 3-4 camera experiments.
    positions = [
        np.asarray([-0.28, -0.66, 0.80], dtype=np.float32),
        np.asarray([-0.28, 0.66, 0.80], dtype=np.float32),
        np.asarray([1.36, -0.52, 0.96], dtype=np.float32),
        np.asarray([1.36, 0.52, 0.96], dtype=np.float32),
    ]
    return [
        FixedCamera(position=position, target=target, width=width, height=height, hfov_deg=hfov_deg, near=near, far=far)
        for position in positions[:count]
    ]


def _camera_rotation_world_to_camera(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / np.linalg.norm(forward)

    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return np.stack([right, up, forward], axis=0).astype(np.float32)


def _camera_world_pose_opencv(camera: FixedCamera) -> Tuple[np.ndarray, np.ndarray, float]:
    rotation_wc = _camera_rotation_world_to_camera(camera.position, camera.target)
    right_world = rotation_wc[0]
    up_world = rotation_wc[1]
    forward_world = rotation_wc[2]

    rotation_world_from_camera = np.stack(
        [right_world, -up_world, forward_world],
        axis=1,
    ).astype(np.float32)
    quat_wxyz = _rotation_matrix_to_wxyz(rotation_world_from_camera)

    _, fy, _, _ = _camera_intrinsics(camera)
    vfov = 2.0 * np.arctan(camera.height / (2.0 * fy))
    return quat_wxyz, camera.position.astype(np.float32), float(vfov)


def _fixed_camera_to_bullet_camera_transform(camera: FixedCamera) -> "SE3":
    if SE3 is None:
        raise RuntimeError("geometrout is required for Bullet depth capture")
    rotation_wc = _camera_rotation_world_to_camera(camera.position, camera.target)
    right_world = rotation_wc[0]
    up_world = rotation_wc[1]
    forward_world = rotation_wc[2]
    # Bullet's camera image path expects an OpenGL-style camera pose:
    # x right, y up, and -z forward in the camera frame.
    rotation_world_from_camera = np.stack(
        [right_world, up_world, -forward_world],
        axis=1,
    ).astype(np.float32)
    quat_wxyz = _rotation_matrix_to_wxyz(rotation_world_from_camera)
    return SE3(camera.position.astype(np.float64), quat_wxyz.astype(np.float64)).inverse


def _obstacle_box_to_cuboid(box) -> "Cuboid":
    if Cuboid is None:
        raise RuntimeError("geometrout is required for Bullet depth capture")
    center = box.center.detach().cpu().numpy().astype(np.float64, copy=False)
    dims = (2.0 * box.half_extents.detach().cpu().numpy()).astype(np.float64, copy=False)
    rotation_world_from_box = box.axes.detach().cpu().numpy().T.astype(np.float64, copy=False)
    quat_wxyz = _rotation_matrix_to_wxyz(rotation_world_from_box.astype(np.float32)).astype(np.float64)
    return Cuboid(center=center, dims=dims, quaternion=quat_wxyz)


class BulletDepthPointcloudRenderer:
    def __init__(self, cameras: List[FixedCamera]):
        if Bullet is None or FrankaRobot is None or SE3 is None or Cuboid is None:
            raise RuntimeError(
                "robofin and geometrout are required for --use-depth Bullet camera capture."
            )
        self.cameras = cameras
        self.sim = Bullet(gui=False)
        self.robot = self.sim.load_robot(FrankaRobot)
        self.last_capture_counts: List[int] = []
        self.last_used_global_fallback = False

    def capture(
        self,
        obstacle_scene: ObstacleScene,
        robot_q: np.ndarray,
        obstacle_points: int,
        depth_voxel_size: float,
        allow_global_fallback: bool,
        include_robot_in_depth: bool,
        robot_pcd_for_removal: Optional[torch.Tensor],
        robot_removal_radius: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        obstacle_primitives = [_obstacle_box_to_cuboid(box) for box in obstacle_scene.boxes]
        self.robot.marionette(np.asarray(robot_q, dtype=np.float64).reshape(-1).tolist())
        self.sim.clear_all_obstacles()
        self.sim.load_primitives(obstacle_primitives)
        try:
            merged_parts: List[np.ndarray] = []
            capture_counts: List[int] = []
            for camera in self.cameras:
                fx, fy, cx, cy = _camera_intrinsics(camera)
                camera_transform = _fixed_camera_to_bullet_camera_transform(camera)
                camera_points = self.sim.get_pointcloud_from_camera(
                    camera_transform,
                    width=camera.width,
                    height=camera.height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    near=camera.near,
                    far=camera.far,
                    remove_robot=None if include_robot_in_depth else self.robot,
                    finite_depth=True,
                )
                capture_counts.append(int(camera_points.shape[0]))
                if camera_points.size > 0:
                    merged_parts.append(camera_points[:, :3].astype(np.float32, copy=False))
        finally:
            self.sim.clear_all_obstacles()

        self.last_capture_counts = capture_counts
        self.last_used_global_fallback = False
        if merged_parts:
            merged = np.concatenate(merged_parts, axis=0)
        else:
            if not allow_global_fallback:
                raise RuntimeError(
                    "Bullet depth capture returned no obstacle points from any camera. "
                    f"Per-camera counts: {capture_counts}. "
                    "Adjust camera poses / target / near-far or rerun with --allow-depth-fallback-to-global "
                    "if you intentionally want to fall back to the global obstacle pointcloud."
                )
            self.last_used_global_fallback = True
            merged = obstacle_scene.obstacle_pcd.detach().cpu().numpy().astype(np.float32, copy=False)

        if depth_voxel_size > 0 and merged.shape[0] > 0:
            voxels = np.round(merged / depth_voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxels, axis=0, return_index=True)
            merged = merged[np.sort(unique_indices)]

        if (
            include_robot_in_depth
            and robot_pcd_for_removal is not None
            and robot_removal_radius > 0
            and merged.shape[0] > 0
        ):
            merged_t = torch.as_tensor(
                merged,
                device=robot_pcd_for_removal.device,
                dtype=torch.float32,
            )
            min_dists = torch.cdist(
                merged_t.unsqueeze(0),
                robot_pcd_for_removal.unsqueeze(0),
            ).amin(dim=2)[0]
            keep_mask = min_dists > robot_removal_radius
            merged = (
                merged_t[keep_mask]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        sampled = _subsample_points(merged, obstacle_points, rng)
        return torch.as_tensor(sampled, device=obstacle_scene.obstacle_pcd.device, dtype=torch.float32)


class MultiCamRolloutVisualizer(RolloutVisualizer):
    def __init__(self, policy: DRPInference, robot_points: int, cameras: List[FixedCamera]):
        super().__init__(policy=policy, robot_points=robot_points)
        self._camera_handles = []
        self._draw_cameras(cameras)

    def _draw_cameras(self, cameras: List[FixedCamera]) -> None:
        for idx, camera in enumerate(cameras, start=1):
            quat_wxyz, position, vfov = _camera_world_pose_opencv(camera)
            aspect = float(camera.width) / float(camera.height)
            color = (190, 40 + 35 * ((idx - 1) % 4), 40 + 45 * ((idx - 1) % 3))
            workspace_depth = float(np.linalg.norm(camera.target - camera.position))
            frustum_scale = min(
                camera.far,
                max(camera.near + 0.15, min(workspace_depth * 1.15, 1.8)),
            )

            self.server.scene.add_frame(
                f"/cameras/cam_{idx}/axes",
                position=position,
                wxyz=quat_wxyz,
                show_axes=True,
                axes_length=0.10,
                axes_radius=0.004,
                origin_radius=0.008,
            )
            self.server.scene.add_box(
                f"/cameras/cam_{idx}/body",
                dimensions=(0.09, 0.025, 0.025),
                color=color,
                position=position,
                wxyz=quat_wxyz,
            )
            self._camera_handles.append(
                self.server.scene.add_camera_frustum(
                    f"/cameras/cam_{idx}/frustum",
                    fov=vfov,
                    aspect=aspect,
                    scale=frustum_scale,
                    line_width=2.0,
                    color=color,
                    position=position,
                    wxyz=quat_wxyz,
                    visible=True,
                    variant="wireframe",
                )
            )

def _subsample_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if points.shape[0] >= num_points:
        indices = rng.choice(points.shape[0], size=num_points, replace=False)
        return points[indices].astype(np.float32, copy=False)
    indices = rng.choice(points.shape[0], size=num_points, replace=True)
    return points[indices].astype(np.float32, copy=False)


def _make_policy_obstacle_pcd(
    obstacle_scene: ObstacleScene,
    use_depth: bool,
    depth_renderer: Optional[BulletDepthPointcloudRenderer],
    start_q_np: np.ndarray,
    obstacle_points: int,
    depth_voxel_size: float,
    allow_depth_fallback_to_global: bool,
    include_robot_in_depth: bool,
    robot_pcd_for_removal: Optional[torch.Tensor],
    robot_removal_radius: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    if not use_depth:
        points = obstacle_scene.obstacle_pcd.detach().cpu().numpy().astype(np.float32, copy=False)
        sampled = _subsample_points(points, obstacle_points, rng)
        return torch.as_tensor(sampled, device=obstacle_scene.obstacle_pcd.device, dtype=torch.float32)

    if depth_renderer is None:
        raise RuntimeError("Depth rendering was requested but no Bullet depth renderer is available.")
    return depth_renderer.capture(
        obstacle_scene=obstacle_scene,
        robot_q=start_q_np,
        obstacle_points=obstacle_points,
        depth_voxel_size=depth_voxel_size,
        allow_global_fallback=allow_depth_fallback_to_global,
        include_robot_in_depth=include_robot_in_depth,
        robot_pcd_for_removal=robot_pcd_for_removal,
        robot_removal_radius=robot_removal_radius,
        rng=rng,
    )


def _run_trial(
    policy: DRPInference,
    start_q_np: np.ndarray,
    goal_q_np: np.ndarray,
    obstacle_scene: ObstacleScene,
    policy_obstacle_pcd: torch.Tensor,
    args: argparse.Namespace,
    trial_id: int,
    direction_tag: str,
    visualizer: Optional[RolloutVisualizer],
) -> TrialResult:
    device = policy.device
    joint_pos = torch.as_tensor(start_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    goal_joint_pos = torch.as_tensor(goal_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    max_step = torch.full_like(joint_pos, args.max_joint_step)

    start_time = time.perf_counter()
    min_clearance = float("inf")

    with torch.inference_mode():
        start_robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
        clearance = _scene_clearance_m(start_robot_pcd, obstacle_scene)
        min_clearance = min(min_clearance, clearance)

        if visualizer is not None:
            visualizer.update(
                trial_id=trial_id,
                direction_tag=direction_tag,
                step=0,
                joint_pos=joint_pos,
                goal_joint_pos=goal_joint_pos,
                obstacle_pcd=policy_obstacle_pcd,
                obstacle_scene=obstacle_scene,
                current_robot_pcd=start_robot_pcd,
                clearance=clearance,
                status="start",
            )
            time.sleep(args.visual_step_sleep)

        if clearance < args.collision_threshold:
            return TrialResult(
                trial_id=trial_id,
                direction=direction_tag,
                success=False,
                reason="start_in_collision",
                steps=0,
                final_joint_error_l1=_joint_error_l1(joint_pos, goal_joint_pos),
                final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
                min_clearance_m=min_clearance,
                runtime_s=time.perf_counter() - start_time,
            )

        for step in range(1, args.max_steps + 1):
            env_obs = {
                "joint_pos": joint_pos,
                "goal_joint_pos": goal_joint_pos,
                "combined_obstacle_pcd": [policy_obstacle_pcd],
            }
            next_q = policy.get_actions(env_obs)
            joint_pos = joint_pos + torch.clamp(next_q - joint_pos, -max_step, max_step)
            joint_pos, _ = clamp_to_franka_limits(joint_pos)

            robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
            clearance = _scene_clearance_m(robot_pcd, obstacle_scene)
            min_clearance = min(min_clearance, clearance)

            if visualizer is not None:
                visualizer.update(
                    trial_id=trial_id,
                    direction_tag=direction_tag,
                    step=step,
                    joint_pos=joint_pos,
                    goal_joint_pos=goal_joint_pos,
                    obstacle_pcd=policy_obstacle_pcd,
                    obstacle_scene=obstacle_scene,
                    current_robot_pcd=robot_pcd,
                    clearance=clearance,
                    status="running",
                )
                time.sleep(args.visual_step_sleep)

            if clearance < args.collision_threshold:
                return TrialResult(
                    trial_id=trial_id,
                    direction=direction_tag,
                    success=False,
                    reason="collision",
                    steps=step,
                    final_joint_error_l1=_joint_error_l1(joint_pos, goal_joint_pos),
                    final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
                    min_clearance_m=min_clearance,
                    runtime_s=time.perf_counter() - start_time,
                )

            joint_err = _joint_error_l1(joint_pos, goal_joint_pos)
            if joint_err <= args.goal_tolerance:
                return TrialResult(
                    trial_id=trial_id,
                    direction=direction_tag,
                    success=True,
                    reason="success",
                    steps=step,
                    final_joint_error_l1=joint_err,
                    final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
                    min_clearance_m=min_clearance,
                    runtime_s=time.perf_counter() - start_time,
                )

    return TrialResult(
        trial_id=trial_id,
        direction=direction_tag,
        success=False,
        reason="timeout",
        steps=args.max_steps,
        final_joint_error_l1=_joint_error_l1(joint_pos, goal_joint_pos),
        final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
        min_clearance_m=min_clearance,
        runtime_s=time.perf_counter() - start_time,
    )


def _run_synthetic_trials(
    policy: DRPInference,
    args: argparse.Namespace,
    rng: np.random.Generator,
    depth_renderer: Optional[BulletDepthPointcloudRenderer],
    config_1: np.ndarray,
    config_2: np.ndarray,
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
    visualizer: Optional[RolloutVisualizer],
) -> List[TrialResult]:
    results: List[TrialResult] = []

    for trial in range(1, args.num_trials + 1):
        start_q, goal_q, direction_tag = _choose_start_goal(
            direction_mode=args.direction,
            config_1=config_1,
            config_2=config_2,
            trial_id=trial,
            rng=rng,
        )
        if args.start_noise_std > 0:
            start_q = start_q + rng.normal(0.0, args.start_noise_std, size=7).astype(np.float32)
            start_q, _ = clamp_to_franka_limits(start_q)
        if args.goal_noise_std > 0:
            goal_q = goal_q + rng.normal(0.0, args.goal_noise_std, size=7).astype(np.float32)
            goal_q, _ = clamp_to_franka_limits(goal_q)

        obstacle_scene = _build_scene_obstacle_pcd(
            device=policy.device,
            table_points=args.table_points,
            box_points=args.box_points,
            box_center=box_center,
            box_half_extents=box_half_extents,
            box_jitter_xy=args.box_jitter_xy,
            rng=rng,
        )
        policy_obstacle_pcd = _make_policy_obstacle_pcd(
            obstacle_scene=obstacle_scene,
            use_depth=args.use_depth,
            depth_renderer=depth_renderer,
            start_q_np=np.asarray(start_q, dtype=np.float32),
            obstacle_points=args.obstacle_points,
            depth_voxel_size=args.depth_voxel_size,
            allow_depth_fallback_to_global=args.allow_depth_fallback_to_global,
            rng=rng,
        )
        if args.use_depth and depth_renderer is not None:
            print(
                f"[depth] trial={trial} per_camera_points={depth_renderer.last_capture_counts} "
                f"fallback_to_global={depth_renderer.last_used_global_fallback}"
            )

        result = _run_trial(
            policy=policy,
            start_q_np=np.asarray(start_q, dtype=np.float32),
            goal_q_np=np.asarray(goal_q, dtype=np.float32),
            obstacle_scene=obstacle_scene,
            policy_obstacle_pcd=policy_obstacle_pcd,
            args=args,
            trial_id=trial,
            direction_tag=direction_tag,
            visualizer=visualizer,
        )
        results.append(result)

        if visualizer is not None:
            time.sleep(args.visual_final_sleep)

        if args.print_every > 0 and (trial % args.print_every == 0 or trial == args.num_trials):
            success_so_far = sum(r.success for r in results)
            print(
                f"[trial {trial:>4d}/{args.num_trials}] "
                f"success={success_so_far}/{trial} ({100.0 * success_so_far / trial:.1f}%) "
                f"last={result.reason:<18s} "
                f"steps={result.steps:>3d} "
                f"clearance={result.min_clearance_m:.4f}m"
            )

    return results


def _run_dataset_trials(
    policy: DRPInference,
    args: argparse.Namespace,
    rng: np.random.Generator,
    depth_renderer: Optional[BulletDepthPointcloudRenderer],
    dataset_hdf5: Path,
    saved_scenes_dir: Path,
    scene_prefix: str,
    visualizer: Optional[RolloutVisualizer],
) -> Tuple[List[TrialResult], Dict[str, int]]:
    if h5py is None:
        raise RuntimeError("h5py is required for --dataset-hdf5 mode")
    if not dataset_hdf5.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_hdf5}")
    if not saved_scenes_dir.is_dir():
        raise FileNotFoundError(f"Saved scenes directory not found: {saved_scenes_dir}")

    results: List[TrialResult] = []
    scene_cache: Dict[int, ObstacleScene] = {}
    missing_scene_count = 0
    parse_failed_scene_count = 0

    with h5py.File(dataset_hdf5, "r") as f:
        if "data" not in f:
            raise KeyError("Expected group 'data' in dataset hdf5")

        demo_keys = sorted(list(f["data"].keys()), key=lambda k: int(k.split("_")[-1]))
        if args.max_demos > 0:
            demo_keys = demo_keys[: args.max_demos]

        total_pairs = 0
        for demo_key in demo_keys:
            group = f["data"][demo_key]
            pair_count = min(group["start_positions"].shape[0], group["goal_positions"].shape[0])
            if args.max_pairs_per_demo > 0:
                pair_count = min(pair_count, args.max_pairs_per_demo)
            total_pairs += pair_count

        print("=" * 80)
        print("DRP success-rate evaluation")
        print(f"Dataset hdf5          : {dataset_hdf5}")
        print(f"Saved scenes dir      : {saved_scenes_dir}")
        print(f"Scene prefix          : {scene_prefix}")
        print(f"Demo groups           : {len(demo_keys)}")
        print(f"Total start-goal pairs: {total_pairs}")
        print(f"Obstacle input mode   : {'fixed_camera_depth' if args.use_depth else 'global_pointcloud'}")
        if args.use_depth:
            print(f"Camera count          : {args.camera_count}")
            print(f"Camera target         : {np.round(_parse_float_list(args.camera_target, 3, '--camera-target'), 4).tolist()}")
        print(f"Obstacle points       : {args.obstacle_points}")
        print(f"Robot points          : {args.collision_robot_points}")
        print(f"Max steps / trial     : {args.max_steps}")
        print(f"Goal tolerance (L1)   : {args.goal_tolerance:.4f} rad")
        print(f"SDF collision thresh  : {args.collision_threshold:.4f} m")
        print("=" * 80)

        trial_id = 1
        for demo_key in demo_keys:
            demo_idx = int(demo_key.split("_")[-1])
            scene_path = saved_scenes_dir / f"{scene_prefix}_demo_{demo_idx}.usd"
            if not scene_path.is_file():
                missing_scene_count += 1
                if args.print_every > 0:
                    print(f"[warn] missing scene for {demo_key}: {scene_path.name}")
                continue

            if demo_idx not in scene_cache:
                try:
                    load_points = args.depth_source_points if args.use_depth else args.obstacle_points
                    scene_cache[demo_idx] = _load_scene_obstacle_pcd_from_usd(
                        scene_path=scene_path,
                        device=policy.device,
                        total_points=load_points,
                        rng=rng,
                    )
                except Exception as exc:
                    parse_failed_scene_count += 1
                    if args.print_every > 0:
                        print(f"[warn] failed to parse {scene_path.name}: {exc}")
                    continue

            group = f["data"][demo_key]
            starts = np.asarray(group["start_positions"], dtype=np.float32)
            goals = np.asarray(group["goal_positions"], dtype=np.float32)
            pair_count = min(starts.shape[0], goals.shape[0])
            if args.max_pairs_per_demo > 0:
                pair_count = min(pair_count, args.max_pairs_per_demo)

            for pair_idx in range(pair_count):
                start_q, _ = clamp_to_franka_limits(starts[pair_idx, :7])
                goal_q, _ = clamp_to_franka_limits(goals[pair_idx, :7])
                policy_obstacle_pcd = _make_policy_obstacle_pcd(
                    obstacle_scene=scene_cache[demo_idx],
                    use_depth=args.use_depth,
                    depth_renderer=depth_renderer,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    obstacle_points=args.obstacle_points,
                    depth_voxel_size=args.depth_voxel_size,
                    allow_depth_fallback_to_global=args.allow_depth_fallback_to_global,
                    rng=rng,
                )
                '''
                if args.use_depth and depth_renderer is not None:
                    print(
                        f"[depth] demo={demo_key} pair={pair_idx} "
                        f"per_camera_points={depth_renderer.last_capture_counts} "
                        f"fallback_to_global={depth_renderer.last_used_global_fallback}"
                    )
                '''
                result = _run_trial(
                    policy=policy,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    goal_q_np=np.asarray(goal_q, dtype=np.float32),
                    obstacle_scene=scene_cache[demo_idx],
                    policy_obstacle_pcd=policy_obstacle_pcd,
                    args=args,
                    trial_id=trial_id,
                    direction_tag=f"{demo_key}/pair_{pair_idx}",
                    visualizer=visualizer,
                )
                results.append(result)

                if visualizer is not None:
                    time.sleep(args.visual_final_sleep)

                if args.print_every > 0 and len(results) % args.print_every == 0:
                    success_so_far = sum(r.success for r in results)
                    print(
                        f"[pair {len(results):>4d}/{total_pairs}] "
                        f"demo={demo_key:<10s} pair={pair_idx:<2d} "
                        f"success={success_so_far}/{len(results)} ({100.0 * success_so_far / len(results):.1f}%) "
                        f"last={result.reason:<18s} "
                        f"steps={result.steps:>3d}"
                    )
                trial_id += 1

    return results, {
        "missing_scenes": missing_scene_count,
        "parse_failed_scenes": parse_failed_scene_count,
        "cached_scenes": len(scene_cache),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DRP success-rate evaluation with global pointcloud or fixed-camera obstacle input."
    )
    parser.add_argument("--dataset-hdf5", type=str, default="", help="Optional dataset HDF5 path.")
    parser.add_argument(
        "--saved-scenes-dir",
        type=str,
        default="test_data/saved_scenes",
        help="Directory containing USD scene files named <scene_prefix>_demo_<id>.usd.",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default="",
        help="Scene filename prefix. If empty, inferred from dataset filename.",
    )
    parser.add_argument("--max-demos", type=int, default=0, help="Limit number of demo groups; 0 means all.")
    parser.add_argument(
        "--max-pairs-per-demo",
        type=int,
        default=0,
        help="Limit number of start/goal pairs used per demo; 0 means all.",
    )
    parser.add_argument("--num-trials", type=int, default=100, help="Number of rollout trials.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum policy steps per trial.")
    parser.add_argument("--max-joint-step", type=float, default=0.08, help="Max per-joint step size (rad).")
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.08,
        help="Success threshold on joint L1 distance to goal (rad).",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.008,
        help="Collision threshold in meters using obstacle SDF clearance.",
    )
    parser.add_argument(
        "--obstacle-points",
        type=int,
        default=DEFAULT_OBSTACLE_POINTS,
        help="Number of obstacle points fed into DRP.",
    )
    parser.add_argument(
        "--collision-robot-points",
        type=int,
        default=DEFAULT_COLLISION_ROBOT_POINTS,
        help="Number of robot points sampled for SDF collision checking.",
    )
    parser.add_argument(
        "--direction",
        choices=["config1_to_config2", "config2_to_config1", "alternate", "random"],
        default="alternate",
        help="Direction policy for choosing start/goal configs each trial.",
    )
    parser.add_argument("--config1", type=str, default="-0.7,0.5,0.0,-2.0,0.0,2.5,0.0")
    parser.add_argument("--config2", type=str, default="0.7,0.8,0.0,-1.7,0.0,3.0,0.0")
    parser.add_argument("--start-noise-std", type=float, default=0.0)
    parser.add_argument("--goal-noise-std", type=float, default=0.0)
    parser.add_argument("--table-points", type=int, default=4096)
    parser.add_argument("--box-points", type=int, default=4096)
    parser.add_argument("--box-center", type=str, default="0.5,0.0,0.2")
    parser.add_argument("--box-half-extents", type=str, default="0.2,0.1,0.2")
    parser.add_argument("--box-jitter-xy", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Use fixed-camera partial obstacle pointcloud input instead of the global obstacle pointcloud.",
    )
    parser.add_argument(
        "--allow-depth-fallback-to-global",
        action="store_true",
        help="If Bullet depth capture returns zero points from all cameras, fall back to the global obstacle pointcloud.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Run in headless mode without visualization.",
    )
    parser.add_argument("--visual-step-sleep", type=float, default=1.0 / 12.0)
    parser.add_argument("--visual-final-sleep", type=float, default=0.4)
    parser.add_argument("--camera-count", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument(
        "--camera-target",
        type=str,
        default="0.60,0.00,0.5",
        help="Look-at target for the fixed cameras in world coordinates (x,y,z).",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_D435_DEPTH_WIDTH,
        help="Depth image width. Default matches a common RealSense D435 depth mode.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_D435_DEPTH_HEIGHT,
        help="Depth image height. Default matches a common RealSense D435 depth mode.",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=DEFAULT_D435_DEPTH_HFOV_DEG,
        help="Horizontal field of view in degrees. Default matches D435 depth FOV.",
    )
    parser.add_argument(
        "--camera-near",
        type=float,
        default=DEFAULT_D435_DEPTH_NEAR_M,
        help="Near plane in meters for Bullet depth rendering. Default is kept very small to avoid workspace clipping.",
    )
    parser.add_argument(
        "--camera-far",
        type=float,
        default=DEFAULT_D435_DEPTH_FAR_M,
        help="Far plane in meters for Bullet depth rendering. Default is large so visible obstacles are not cropped by range.",
    )
    parser.add_argument(
        "--depth-source-points",
        type=int,
        default=DEFAULT_CAMERA_SOURCE_POINTS,
        help="Legacy fallback sample count for global obstacle points if Bullet depth capture returns no points.",
    )
    parser.add_argument(
        "--depth-voxel-size",
        type=float,
        default=0.003,
        help="Voxel size in meters for downsampling merged Bullet camera pointclouds.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.max_demos < 0:
        raise ValueError("--max-demos must be >= 0")
    if args.max_pairs_per_demo < 0:
        raise ValueError("--max-pairs-per-demo must be >= 0")
    if args.obstacle_points <= 0:
        raise ValueError("--obstacle-points must be > 0")
    if args.collision_robot_points <= 0:
        raise ValueError("--collision-robot-points must be > 0")
    if args.visual_step_sleep < 0 or args.visual_final_sleep < 0:
        raise ValueError("visual sleep parameters must be >= 0")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("--camera-width and --camera-height must be > 0")
    if args.camera_near <= 0 or args.camera_far <= args.camera_near:
        raise ValueError("--camera-far must be > --camera-near > 0")
    if args.depth_source_points <= 0:
        raise ValueError("--depth-source-points must be > 0")

    use_dataset_mode = bool(args.dataset_hdf5)
    if not use_dataset_mode and args.num_trials <= 0:
        raise ValueError("--num-trials must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "DRP inference requires CUDA because pointnet2_ops only supports CUDA tensors in this setup. "
            "Please run on a GPU machine and pass --device cuda if needed."
        )
    policy = DRPInference(device=device)

    camera_target = _parse_float_list(args.camera_target, expected_len=3, name="--camera-target")
    cameras = _make_fixed_cameras(
        count=args.camera_count,
        target=camera_target,
        width=args.camera_width,
        height=args.camera_height,
        hfov_deg=args.camera_hfov_deg,
        near=args.camera_near,
        far=args.camera_far,
    )
    depth_renderer = BulletDepthPointcloudRenderer(cameras) if args.use_depth else None
    visualizer = None
    if not args.skip_visuals:
        visualizer = MultiCamRolloutVisualizer(
            policy=policy,
            robot_points=args.collision_robot_points,
            cameras=cameras,
        )

    print("=" * 80)
    print("DRP success-rate evaluation")
    print(f"Device                : {device}")
    print(f"Mode                  : {'dataset_usd' if use_dataset_mode else 'synthetic'}")
    print(f"Obstacle input mode   : {'fixed_camera_depth' if args.use_depth else 'global_pointcloud'}")
    if args.use_depth:
        print(f"Camera count          : {args.camera_count}")
        print(f"Camera target         : {np.round(camera_target, 4).tolist()}")
    print(f"Obstacle points       : {args.obstacle_points}")
    print(f"Robot points          : {args.collision_robot_points}")
    print(f"SDF collision thresh  : {args.collision_threshold:.4f} m")
    print(f"Goal tolerance (L1)   : {args.goal_tolerance:.4f} rad")
    print(f"Visuals               : {'off' if args.skip_visuals else 'on'}")
    print("=" * 80)

    begin = time.perf_counter()
    dataset_meta: Optional[Dict[str, int]] = None
    if use_dataset_mode:
        dataset_hdf5 = Path(args.dataset_hdf5)
        saved_scenes_dir = Path(args.saved_scenes_dir)
        scene_prefix = args.scene_prefix.strip() or _infer_scene_prefix_from_dataset(dataset_hdf5)
        results, dataset_meta = _run_dataset_trials(
            policy=policy,
            args=args,
            rng=rng,
            depth_renderer=depth_renderer,
            dataset_hdf5=dataset_hdf5,
            saved_scenes_dir=saved_scenes_dir,
            scene_prefix=scene_prefix,
            visualizer=visualizer,
        )
    else:
        config_1 = _parse_float_list(args.config1, expected_len=7, name="--config1")
        config_2 = _parse_float_list(args.config2, expected_len=7, name="--config2")
        box_center = _parse_float_list(args.box_center, expected_len=3, name="--box-center")
        box_half_extents = _parse_float_list(args.box_half_extents, expected_len=3, name="--box-half-extents")
        results = _run_synthetic_trials(
            policy=policy,
            args=args,
            rng=rng,
            depth_renderer=depth_renderer,
            config_1=config_1,
            config_2=config_2,
            box_center=box_center,
            box_half_extents=box_half_extents,
            visualizer=visualizer,
        )

    total_runtime = time.perf_counter() - begin
    if not results:
        raise RuntimeError("No valid trials were executed. Please check dataset paths and scene files.")

    summary = _summarize(results)
    summary["total_runtime_s"] = total_runtime
    summary["use_depth"] = args.use_depth
    summary["camera_count"] = args.camera_count if args.use_depth else 0
    summary["obstacle_points"] = args.obstacle_points
    summary["collision_robot_points"] = args.collision_robot_points
    if dataset_meta is not None:
        summary.update(dataset_meta)

    print("\n" + "=" * 80)
    print("Evaluation finished")
    print(f"Success rate          : {summary['success_rate'] * 100.0:.2f}%")
    print(f"Success / Total       : {summary['success_count']} / {summary['total_trials']}")
    print(f"Failure breakdown     : {summary['failure_breakdown']}")
    print(f"Avg success steps     : {summary['avg_steps_success']}")
    print(f"Avg final joint err   : {summary['avg_final_joint_error_l1']:.4f} rad")
    print(f"Avg final eef err     : {summary['avg_final_eef_error_m']:.4f} m")
    print(f"Avg min clearance     : {summary['avg_min_clearance_m']:.4f} m")
    print(f"Min clearance         : {summary['min_clearance_m']:.4f} m")
    print(f"Total runtime         : {summary['total_runtime_s']:.2f} s")
    print("=" * 80)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
