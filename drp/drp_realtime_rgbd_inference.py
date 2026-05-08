import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from drp.drp_inference import DRPInference
from drp.drp_success_rate_inference import RolloutVisualizer, _joint_error_l1, _parse_float_list, _resolve_device
from drp.utils.franka_utils import clamp_to_franka_limits

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


DEFAULT_D435_DEPTH_WIDTH = 848
DEFAULT_D435_DEPTH_HEIGHT = 480
DEFAULT_D435_FPS = 30


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float32)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _pose_matrix(position: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _rpy_to_matrix(rpy)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32)
    return matrix


def _load_camera_to_world(
    extrinsics_json: str,
    camera_position: np.ndarray,
    camera_rpy: np.ndarray,
) -> np.ndarray:
    if not extrinsics_json:
        return _pose_matrix(camera_position, camera_rpy)

    payload = json.loads(Path(extrinsics_json).read_text(encoding="utf-8"))
    if "camera_to_world" in payload:
        matrix = np.asarray(payload["camera_to_world"], dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError("camera_to_world must be a 4x4 matrix")
        return matrix

    if "position" in payload and "rpy" in payload:
        return _pose_matrix(
            np.asarray(payload["position"], dtype=np.float32),
            np.asarray(payload["rpy"], dtype=np.float32),
        )

    raise ValueError(
        "Extrinsics JSON must contain either 'camera_to_world' or both 'position' and 'rpy'"
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    rotated = points @ transform[:3, :3].T
    translated = rotated + transform[:3, 3]
    return translated.astype(np.float32, copy=False)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices].astype(np.float32, copy=False)


def _crop_workspace(points: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    return points[mask]


def _remove_robot_points(
    points: np.ndarray,
    robot_pcd: torch.Tensor,
    threshold_m: float,
    device: torch.device,
) -> np.ndarray:
    if threshold_m <= 0 or points.shape[0] == 0:
        return points
    points_t = torch.as_tensor(points, device=device, dtype=torch.float32)
    min_dists = torch.cdist(points_t.unsqueeze(0), robot_pcd.unsqueeze(0)).amin(dim=2)[0]
    keep = min_dists > threshold_m
    return points_t[keep].detach().cpu().numpy().astype(np.float32, copy=False)


def _load_joint_vector(payload: dict, keys: Tuple[str, ...], fallback: np.ndarray) -> np.ndarray:
    for key in keys:
        if key in payload:
            values = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            if values.shape[0] < 7:
                raise ValueError(f"Key '{key}' must contain at least 7 values")
            return values[:7]
    return fallback


def _load_runtime_state(
    joint_state_file: str,
    goal_state_file: str,
    current_q: np.ndarray,
    goal_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    current = current_q.copy()
    goal = goal_q.copy()

    if joint_state_file:
        payload = json.loads(Path(joint_state_file).read_text(encoding="utf-8"))
        current = _load_joint_vector(
            payload,
            keys=("joint_pos", "current_joint_pos", "current_angles", "q"),
            fallback=current,
        )
        goal = _load_joint_vector(
            payload,
            keys=("goal_joint_pos", "goal_angles", "goal_q"),
            fallback=goal,
        )

    if goal_state_file:
        payload = json.loads(Path(goal_state_file).read_text(encoding="utf-8"))
        goal = _load_joint_vector(
            payload,
            keys=("goal_joint_pos", "goal_angles", "goal_q", "joint_pos"),
            fallback=goal,
        )

    current, _ = clamp_to_franka_limits(current.astype(np.float32))
    goal, _ = clamp_to_franka_limits(goal.astype(np.float32))
    return np.asarray(current, dtype=np.float32), np.asarray(goal, dtype=np.float32)


def _write_command_output(
    output_file: str,
    predicted_q: np.ndarray,
    commanded_q: np.ndarray,
    loop_idx: int,
    point_count: int,
) -> None:
    if not output_file:
        return
    payload = {
        "loop_idx": int(loop_idx),
        "timestamp_s": time.time(),
        "predicted_joint_target": np.asarray(predicted_q, dtype=np.float32).tolist(),
        "command_joint_target": np.asarray(commanded_q, dtype=np.float32).tolist(),
        "point_count": int(point_count),
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class RealSensePointcloudSource:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str,
        min_depth_m: float,
        max_depth_m: float,
        depth_stride: int,
        warmup_frames: int,
    ):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required for realtime RGB-D input. Please install librealsense and pyrealsense2."
            )

        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_stride = max(1, int(depth_stride))

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(max(0, int(warmup_frames))):
            self.pipeline.wait_for_frames()

    def capture(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Failed to acquire aligned depth frame from RealSense")

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        if self.depth_stride > 1:
            depth = depth[:: self.depth_stride, :: self.depth_stride]

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx = float(intrinsics.fx) / self.depth_stride
        fy = float(intrinsics.fy) / self.depth_stride
        cx = float(intrinsics.ppx) / self.depth_stride
        cy = float(intrinsics.ppy) / self.depth_stride

        valid = np.isfinite(depth)
        valid &= depth > self.min_depth_m
        valid &= depth < self.max_depth_m
        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        z = depth[ys, xs]
        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy
        return np.stack((x, y, z), axis=1).astype(np.float32, copy=False)

    def close(self) -> None:
        self.pipeline.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime DRP inference driven by RGB-D camera pointcloud input."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop-hz", type=float, default=10.0, help="Target control loop frequency.")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--goal-tolerance", type=float, default=0.08)
    parser.add_argument("--collision-robot-points", type=int, default=1024)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--skip-visual", action="store_true", help="Run in headless mode.")
    parser.add_argument("--visualize-step-sleep", type=float, default=0.0)

    parser.add_argument(
        "--joint-state-file",
        type=str,
        default="",
        help="Optional JSON file with joint_pos and optionally goal_joint_pos, reloaded every loop.",
    )
    parser.add_argument(
        "--goal-state-file",
        type=str,
        default="",
        help="Optional JSON file with goal_joint_pos, reloaded every loop.",
    )
    parser.add_argument(
        "--start-config",
        type=str,
        default="-0.7,0.5,0.0,-2.0,0.0,2.5,0.0",
        help="Fallback current joint configuration if no runtime joint source is provided.",
    )
    parser.add_argument(
        "--goal-config",
        type=str,
        default="0.7,0.8,0.0,-1.7,0.0,3.0,0.0",
        help="Fallback goal joint configuration if no runtime goal source is provided.",
    )
    parser.add_argument(
        "--command-output-file",
        type=str,
        default="",
        help="Optional JSON file to write the commanded joint target after each inference step.",
    )

    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=DEFAULT_D435_DEPTH_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_D435_DEPTH_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_D435_FPS)
    parser.add_argument("--camera-min-depth", type=float, default=0.15)
    parser.add_argument("--camera-max-depth", type=float, default=2.0)
    parser.add_argument("--depth-stride", type=int, default=2, help="Use every Nth depth pixel for faster pointcloud conversion.")
    parser.add_argument("--warmup-frames", type=int, default=15)

    parser.add_argument(
        "--camera-extrinsics-json",
        type=str,
        default="",
        help="Optional JSON file containing camera_to_world or {position, rpy}.",
    )
    parser.add_argument(
        "--camera-position",
        type=str,
        default="0.0,0.0,0.0",
        help="Camera position in world coordinates used when --camera-extrinsics-json is not set.",
    )
    parser.add_argument(
        "--camera-rpy",
        type=str,
        default="0.0,0.0,0.0",
        help="Camera roll,pitch,yaw in radians used when --camera-extrinsics-json is not set.",
    )
    parser.add_argument(
        "--workspace-min",
        type=str,
        default="0.15,-0.80,-0.05",
        help="Crop lower bound in world frame.",
    )
    parser.add_argument(
        "--workspace-max",
        type=str,
        default="1.10,0.80,1.20",
        help="Crop upper bound in world frame.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument(
        "--max-pointcloud-points",
        type=int,
        default=12000,
        help="Max number of obstacle points kept before policy subsampling.",
    )
    parser.add_argument(
        "--min-pointcloud-points",
        type=int,
        default=128,
        help="Skip inference if the filtered pointcloud has fewer points than this.",
    )
    parser.add_argument(
        "--remove-robot-from-depth",
        action="store_true",
        help="Remove depth points close to the current robot pointcloud before inference.",
    )
    parser.add_argument(
        "--robot-removal-threshold",
        type=float,
        default=0.03,
        help="Distance threshold in meters used by --remove-robot-from-depth.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.loop_hz <= 0:
        raise ValueError("--loop-hz must be > 0")
    if args.max_iterations < 0:
        raise ValueError("--max-iterations must be >= 0")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        raise ValueError("camera width/height/fps must be > 0")
    if args.camera_min_depth <= 0 or args.camera_max_depth <= args.camera_min_depth:
        raise ValueError("--camera-max-depth must be > --camera-min-depth > 0")
    if args.max_pointcloud_points <= 0:
        raise ValueError("--max-pointcloud-points must be > 0")
    if args.min_pointcloud_points < 0:
        raise ValueError("--min-pointcloud-points must be >= 0")
    if args.collision_robot_points <= 0:
        raise ValueError("--collision-robot-points must be > 0")
    if args.visualize_step_sleep < 0:
        raise ValueError("--visualize-step-sleep must be >= 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    start_q = _parse_float_list(args.start_config, expected_len=7, name="--start-config")
    goal_q = _parse_float_list(args.goal_config, expected_len=7, name="--goal-config")
    camera_position = _parse_float_list(args.camera_position, expected_len=3, name="--camera-position")
    camera_rpy = _parse_float_list(args.camera_rpy, expected_len=3, name="--camera-rpy")
    workspace_min = _parse_float_list(args.workspace_min, expected_len=3, name="--workspace-min")
    workspace_max = _parse_float_list(args.workspace_max, expected_len=3, name="--workspace-max")
    camera_to_world = _load_camera_to_world(args.camera_extrinsics_json, camera_position, camera_rpy)

    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "DRP realtime inference requires CUDA in this repository because pointnet2_ops expects CUDA tensors."
        )

    policy = DRPInference(device=device)
    visualizer = None if args.skip_visual else RolloutVisualizer(policy=policy, robot_points=args.collision_robot_points)
    pointcloud_source = RealSensePointcloudSource(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        serial=args.camera_serial.strip(),
        min_depth_m=args.camera_min_depth,
        max_depth_m=args.camera_max_depth,
        depth_stride=args.depth_stride,
        warmup_frames=args.warmup_frames,
    )

    print("=" * 80)
    print("DRP realtime RGB-D inference")
    print(f"Device                : {device}")
    print(f"RealSense serial      : {args.camera_serial or '<auto>'}")
    print(f"Depth mode            : {args.camera_width}x{args.camera_height}@{args.camera_fps}")
    print(f"Loop Hz               : {args.loop_hz}")
    print(f"Visualize             : {not args.skip_visual}")
    print(f"Joint source          : {'file' if args.joint_state_file else 'internal_closed_loop'}")
    print(f"Goal source           : {'goal file' if args.goal_state_file else 'joint file/static config'}")
    print(f"Workspace min/max     : {np.round(workspace_min, 3).tolist()} / {np.round(workspace_max, 3).tolist()}")
    print("=" * 80)

    current_q = np.asarray(start_q, dtype=np.float32)
    goal_q = np.asarray(goal_q, dtype=np.float32)
    max_step = float(args.max_joint_step)
    loop_period = 1.0 / float(args.loop_hz)

    try:
        loop_idx = 0
        while args.max_iterations == 0 or loop_idx < args.max_iterations:
            tic = time.perf_counter()
            current_q, goal_q = _load_runtime_state(
                joint_state_file=args.joint_state_file,
                goal_state_file=args.goal_state_file,
                current_q=current_q,
                goal_q=goal_q,
            )

            raw_camera_points = pointcloud_source.capture()
            obstacle_points = _transform_points(raw_camera_points, camera_to_world)
            obstacle_points = _crop_workspace(obstacle_points, workspace_min, workspace_max)
            obstacle_points = _voxel_downsample(obstacle_points, args.voxel_size)
            obstacle_points = _subsample_points(obstacle_points, args.max_pointcloud_points, rng)

            joint_pos = torch.as_tensor(current_q, device=device, dtype=torch.float32).unsqueeze(0)
            goal_joint_pos = torch.as_tensor(goal_q, device=device, dtype=torch.float32).unsqueeze(0)
            robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]

            if args.remove_robot_from_depth:
                obstacle_points = _remove_robot_points(
                    points=obstacle_points,
                    robot_pcd=robot_pcd,
                    threshold_m=args.robot_removal_threshold,
                    device=device,
                )

            if obstacle_points.shape[0] < args.min_pointcloud_points:
                if args.print_every > 0 and (loop_idx % args.print_every == 0):
                    print(
                        f"[loop {loop_idx:>5d}] skipped: only {obstacle_points.shape[0]} points after filtering"
                    )
                loop_idx += 1
                sleep_time = loop_period - (time.perf_counter() - tic)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            obstacle_pcd = torch.as_tensor(obstacle_points, device=device, dtype=torch.float32)
            env_obs = {
                "joint_pos": joint_pos,
                "goal_joint_pos": goal_joint_pos,
                "combined_obstacle_pcd": [obstacle_pcd],
            }

            predicted_q = policy.get_actions(env_obs)
            commanded_q = joint_pos + torch.clamp(predicted_q - joint_pos, -max_step, max_step)
            commanded_q, _ = clamp_to_franka_limits(commanded_q)

            predicted_q_np = predicted_q[0].detach().cpu().numpy().astype(np.float32, copy=False)
            commanded_q_np = commanded_q[0].detach().cpu().numpy().astype(np.float32, copy=False)
            joint_err = _joint_error_l1(commanded_q, goal_joint_pos)

            _write_command_output(
                output_file=args.command_output_file,
                predicted_q=predicted_q_np,
                commanded_q=commanded_q_np,
                loop_idx=loop_idx,
                point_count=obstacle_points.shape[0],
            )

            if visualizer is not None:
                visualizer.update(
                    trial_id=loop_idx,
                    direction_tag="realtime_rgbd",
                    step=loop_idx,
                    joint_pos=joint_pos,
                    goal_joint_pos=goal_joint_pos,
                    obstacle_pcd=obstacle_pcd,
                    current_robot_pcd=robot_pcd,
                    clearance=float("inf"),
                    status="running",
                )
                if args.visualize_step_sleep > 0:
                    time.sleep(args.visualize_step_sleep)

            if not args.joint_state_file:
                current_q = commanded_q_np.copy()

            elapsed_ms = 1000.0 * (time.perf_counter() - tic)
            if args.print_every > 0 and (loop_idx % args.print_every == 0):
                print(
                    f"[loop {loop_idx:>5d}] "
                    f"points={obstacle_points.shape[0]:>5d} "
                    f"joint_err={joint_err:.4f} "
                    f"latency={elapsed_ms:.1f}ms"
                )

            loop_idx += 1
            sleep_time = loop_period - (time.perf_counter() - tic)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        pointcloud_source.close()


if __name__ == "__main__":
    main()
