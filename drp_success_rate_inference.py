import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from drp.drp_inference import DRPInference
from drp.utils.franka_utils import clamp_to_franka_limits

try:
    import h5py
except ImportError:
    h5py = None

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None
    UsdGeom = None


@dataclass
class TrialResult:
    trial_id: int
    direction: str
    success: bool
    reason: str
    steps: int
    final_joint_error_l2: float
    final_eef_error_m: float
    min_clearance_m: float
    runtime_s: float


def _parse_float_list(text: str, expected_len: int, name: str) -> np.ndarray:
    try:
        values = [float(v.strip()) for v in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated floats") from exc
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"{name} must have exactly {expected_len} values, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float32)


def _box_surface_points(center: np.ndarray, half_extents: np.ndarray, n: int, device: torch.device) -> torch.Tensor:
    center_t = torch.as_tensor(center, device=device, dtype=torch.float32)
    half_t = torch.as_tensor(half_extents, device=device, dtype=torch.float32)
    pts = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * half_t
    faces = torch.randint(0, 6, (n,), device=device)
    axes = faces // 2
    signs = faces.remainder(2).float() * 2.0 - 1.0
    pts[torch.arange(n, device=device), axes] = signs * half_t[axes]
    return pts + center_t


def _build_scene_obstacle_pcd(
    device: torch.device,
    table_points: int,
    box_points: int,
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
    box_jitter_xy: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    center = box_center.copy()
    if box_jitter_xy > 0:
        center[:2] += rng.uniform(-box_jitter_xy, box_jitter_xy, size=2).astype(np.float32)

    table = torch.empty(table_points, 3, device=device)
    table[:, 0] = torch.rand(table_points, device=device) * 1.0 + 0.2
    table[:, 1] = torch.rand(table_points, device=device) * 1.2 - 0.6
    table[:, 2] = 0.02

    box = _box_surface_points(center, box_half_extents, box_points, device)
    return torch.cat((table, box), dim=0)


def _demo_index_from_key(demo_key: str) -> int:
    if not demo_key.startswith("demo_"):
        raise ValueError(f"Invalid demo key format: {demo_key}")
    return int(demo_key.split("_")[-1])


def _infer_scene_prefix_from_dataset(dataset_hdf5: Path) -> str:
    stem = dataset_hdf5.stem
    return stem[:-8] if stem.endswith("_targets") else stem


def _sample_cube_surface_points_np(size: float, num_points: int, rng: np.random.Generator) -> np.ndarray:
    half = 0.5 * float(size)
    pts = (rng.random((num_points, 3), dtype=np.float32) * 2.0 - 1.0) * half
    faces = rng.integers(0, 6, size=num_points)
    axes = faces // 2
    signs = (faces % 2) * 2 - 1
    pts[np.arange(num_points), axes] = signs.astype(np.float32) * half
    return pts


def _load_scene_obstacle_pcd_from_usd(
    scene_path: Path,
    device: torch.device,
    total_points: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    if Usd is None or UsdGeom is None:
        raise RuntimeError("pxr is required for USD parsing but is not available")

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {scene_path}")

    obstacle_cubes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube):
            continue
        if not str(prim.GetPath()).startswith("/World/Obstacles"):
            continue
        obstacle_cubes.append(prim)

    if not obstacle_cubes:
        raise RuntimeError(f"No obstacle cubes found in scene: {scene_path}")

    xform_cache = UsdGeom.XformCache()
    points_per_cube = max(64, int(np.ceil(total_points / len(obstacle_cubes))))

    parts = []
    for prim in obstacle_cubes:
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get()
        if size is None:
            size = 2.0

        local_pts = _sample_cube_surface_points_np(size=float(size), num_points=points_per_cube, rng=rng)
        matrix = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)
        world_pts = local_pts @ matrix[:3, :3] + matrix[3, :3]
        parts.append(world_pts)

    obstacle_pts = np.concatenate(parts, axis=0)

    if obstacle_pts.shape[0] >= total_points:
        indices = rng.choice(obstacle_pts.shape[0], size=total_points, replace=False)
        obstacle_pts = obstacle_pts[indices]
    else:
        indices = rng.choice(obstacle_pts.shape[0], size=total_points, replace=True)
        obstacle_pts = obstacle_pts[indices]

    return torch.as_tensor(obstacle_pts, device=device, dtype=torch.float32)


def _joint_error_l2(q: torch.Tensor, q_goal: torch.Tensor) -> float:
    return float(torch.linalg.norm(q - q_goal, dim=1).item())


def _eef_error_m(policy: DRPInference, q: torch.Tensor, q_goal: torch.Tensor) -> float:
    eef = policy._fk_sampler.end_effector_pose(q)[0, :3, 3]
    eef_goal = policy._fk_sampler.end_effector_pose(q_goal)[0, :3, 3]
    return float(torch.linalg.norm(eef - eef_goal).item())


def _pointcloud_clearance_m(robot_pcd: torch.Tensor, obstacle_pcd: torch.Tensor) -> float:
    dists = torch.cdist(robot_pcd.unsqueeze(0), obstacle_pcd.unsqueeze(0))
    return float(dists.amin().item())


def _choose_start_goal(
    direction_mode: str,
    config_1: np.ndarray,
    config_2: np.ndarray,
    trial_id: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if direction_mode == "config1_to_config2":
        go_forward = True
    elif direction_mode == "config2_to_config1":
        go_forward = False
    elif direction_mode == "alternate":
        go_forward = trial_id % 2 == 1
    elif direction_mode == "random":
        go_forward = bool(rng.integers(0, 2))
    else:
        raise ValueError(f"Unknown direction mode: {direction_mode}")

    if go_forward:
        return config_1.copy(), config_2.copy(), "config1->config2"
    return config_2.copy(), config_1.copy(), "config2->config1"


def _run_trial(
    policy: DRPInference,
    start_q_np: np.ndarray,
    goal_q_np: np.ndarray,
    obstacle_pcd: torch.Tensor,
    args: argparse.Namespace,
    trial_id: int,
    direction_tag: str,
) -> TrialResult:
    device = policy.device
    joint_pos = torch.as_tensor(start_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    goal_joint_pos = torch.as_tensor(goal_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    max_step = torch.full_like(joint_pos, args.max_joint_step)

    start_time = time.perf_counter()
    min_clearance = float("inf")

    with torch.inference_mode():
        start_robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
        clearance = _pointcloud_clearance_m(start_robot_pcd, obstacle_pcd)
        min_clearance = min(min_clearance, clearance)
        if clearance < args.collision_threshold:
            return TrialResult(
                trial_id=trial_id,
                direction=direction_tag,
                success=False,
                reason="start_in_collision",
                steps=0,
                final_joint_error_l2=_joint_error_l2(joint_pos, goal_joint_pos),
                final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
                min_clearance_m=min_clearance,
                runtime_s=time.perf_counter() - start_time,
            )

        for step in range(1, args.max_steps + 1):
            env_obs = {
                "joint_pos": joint_pos,
                "goal_joint_pos": goal_joint_pos,
                "combined_obstacle_pcd": [obstacle_pcd],
            }
            next_q = policy.get_actions(env_obs)
            joint_pos = joint_pos + torch.clamp(next_q - joint_pos, -max_step, max_step)
            joint_pos, _ = clamp_to_franka_limits(joint_pos)

            robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
            clearance = _pointcloud_clearance_m(robot_pcd, obstacle_pcd)
            min_clearance = min(min_clearance, clearance)

            if clearance < args.collision_threshold:
                return TrialResult(
                    trial_id=trial_id,
                    direction=direction_tag,
                    success=False,
                    reason="collision",
                    steps=step,
                    final_joint_error_l2=_joint_error_l2(joint_pos, goal_joint_pos),
                    final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
                    min_clearance_m=min_clearance,
                    runtime_s=time.perf_counter() - start_time,
                )

            joint_err = _joint_error_l2(joint_pos, goal_joint_pos)
            if joint_err <= args.goal_tolerance:
                return TrialResult(
                    trial_id=trial_id,
                    direction=direction_tag,
                    success=True,
                    reason="success",
                    steps=step,
                    final_joint_error_l2=joint_err,
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
        final_joint_error_l2=_joint_error_l2(joint_pos, goal_joint_pos),
        final_eef_error_m=_eef_error_m(policy, joint_pos, goal_joint_pos),
        min_clearance_m=min_clearance,
        runtime_s=time.perf_counter() - start_time,
    )


def _run_synthetic_trials(
    policy: DRPInference,
    args: argparse.Namespace,
    rng: np.random.Generator,
    config_1: np.ndarray,
    config_2: np.ndarray,
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
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

        obstacle_pcd = _build_scene_obstacle_pcd(
            device=policy.device,
            table_points=args.table_points,
            box_points=args.box_points,
            box_center=box_center,
            box_half_extents=box_half_extents,
            box_jitter_xy=args.box_jitter_xy,
            rng=rng,
        )

        result = _run_trial(
            policy=policy,
            start_q_np=np.asarray(start_q, dtype=np.float32),
            goal_q_np=np.asarray(goal_q, dtype=np.float32),
            obstacle_pcd=obstacle_pcd,
            args=args,
            trial_id=trial,
            direction_tag=direction_tag,
        )
        results.append(result)

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
    dataset_hdf5: Path,
    saved_scenes_dir: Path,
    scene_prefix: str,
) -> Tuple[List[TrialResult], Dict[str, int]]:
    if h5py is None:
        raise RuntimeError("h5py is required for --dataset-hdf5 mode")

    if not dataset_hdf5.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_hdf5}")
    if not saved_scenes_dir.is_dir():
        raise FileNotFoundError(f"Saved scenes directory not found: {saved_scenes_dir}")

    results: List[TrialResult] = []
    scene_cache: Dict[int, torch.Tensor] = {}
    missing_scene_count = 0
    parse_failed_scene_count = 0

    with h5py.File(dataset_hdf5, "r") as f:
        if "data" not in f:
            raise KeyError("Expected group 'data' in dataset hdf5")

        demo_keys = sorted(
            list(f["data"].keys()),
            key=lambda k: _demo_index_from_key(k),
        )
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
        print("DRP dataset success-rate evaluation")
        print(f"Dataset hdf5          : {dataset_hdf5}")
        print(f"Saved scenes dir      : {saved_scenes_dir}")
        print(f"Scene prefix          : {scene_prefix}")
        print(f"Demo groups           : {len(demo_keys)}")
        print(f"Total start-goal pairs: {total_pairs}")
        print(f"Max steps / trial     : {args.max_steps}")
        print(f"Goal tolerance (L2)   : {args.goal_tolerance:.4f} rad")
        print(f"Collision threshold   : {args.collision_threshold:.4f} m")
        print(f"USD obstacle points   : {args.usd_obstacle_points}")
        print("=" * 80)

        trial_id = 1
        for demo_idx_pos, demo_key in enumerate(demo_keys, start=1):
            demo_idx = _demo_index_from_key(demo_key)
            scene_path = saved_scenes_dir / f"{scene_prefix}_demo_{demo_idx}.usd"

            if not scene_path.is_file():
                missing_scene_count += 1
                if args.print_every > 0:
                    print(f"[warn] missing scene for {demo_key}: {scene_path.name}")
                continue

            if demo_idx not in scene_cache:
                try:
                    scene_cache[demo_idx] = _load_scene_obstacle_pcd_from_usd(
                        scene_path=scene_path,
                        device=policy.device,
                        total_points=args.usd_obstacle_points,
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

                result = _run_trial(
                    policy=policy,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    goal_q_np=np.asarray(goal_q, dtype=np.float32),
                    obstacle_pcd=scene_cache[demo_idx],
                    args=args,
                    trial_id=trial_id,
                    direction_tag=f"{demo_key}/pair_{pair_idx}",
                )
                results.append(result)
                trial_id += 1

                if args.print_every > 0 and len(results) % args.print_every == 0:
                    success_so_far = sum(r.success for r in results)
                    print(
                        f"[pair {len(results):>4d}/{total_pairs}] "
                        f"demo={demo_key:<10s} pair={pair_idx:<2d} "
                        f"success={success_so_far}/{len(results)} ({100.0 * success_so_far / len(results):.1f}%) "
                        f"last={result.reason:<18s} "
                        f"steps={result.steps:>3d}"
                    )

            if args.print_every > 0 and demo_idx_pos % max(args.print_every, 1) == 0:
                print(f"[demo {demo_idx_pos:>4d}/{len(demo_keys)}] processed")

    return results, {
        "missing_scenes": missing_scene_count,
        "parse_failed_scenes": parse_failed_scene_count,
        "cached_scenes": len(scene_cache),
    }


def _summarize(results: List[TrialResult]) -> dict:
    if not results:
        return {}

    total = len(results)
    success_results = [r for r in results if r.success]
    fail_results = [r for r in results if not r.success]
    reasons = Counter(r.reason for r in fail_results)

    summary = {
        "total_trials": total,
        "success_count": len(success_results),
        "success_rate": len(success_results) / total,
        "failure_count": len(fail_results),
        "failure_breakdown": dict(reasons),
        "avg_steps_success": float(np.mean([r.steps for r in success_results])) if success_results else None,
        "avg_runtime_s": float(np.mean([r.runtime_s for r in results])),
        "avg_final_joint_error_l2": float(np.mean([r.final_joint_error_l2 for r in results])),
        "avg_final_eef_error_m": float(np.mean([r.final_eef_error_m for r in results])),
        "avg_min_clearance_m": float(np.mean([r.min_clearance_m for r in results])),
        "min_clearance_m": float(np.min([r.min_clearance_m for r in results])),
    }
    return summary


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch DRP inference evaluator for success-rate measurement."
    )
    parser.add_argument(
        "--dataset-hdf5",
        type=str,
        default="",
        help="Optional dataset HDF5 path. If provided, runs dataset/USD evaluation mode.",
    )
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
    parser.add_argument(
        "--max-demos",
        type=int,
        default=0,
        help="Limit number of demo groups loaded from dataset; 0 means all.",
    )
    parser.add_argument(
        "--max-pairs-per-demo",
        type=int,
        default=0,
        help="Limit number of start/goal pairs used per demo; 0 means all.",
    )
    parser.add_argument(
        "--usd-obstacle-points",
        type=int,
        default=6000,
        help="Number of obstacle points sampled from each USD scene for collision checks.",
    )
    parser.add_argument("--num-trials", type=int, default=100, help="Number of rollout trials.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum policy steps per trial.")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.08,
        help="Max per-joint step size (rad) for each policy action.",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.08,
        help="Success threshold on joint L2 distance to goal (rad).",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.008,
        help="Collision threshold in meters using pointcloud clearance.",
    )
    parser.add_argument(
        "--collision-robot-points",
        type=int,
        default=512,
        help="Number of robot points sampled for pointcloud collision checking.",
    )
    parser.add_argument(
        "--direction",
        choices=["config1_to_config2", "config2_to_config1", "alternate", "random"],
        default="alternate",
        help="Direction policy for choosing start/goal configs each trial.",
    )
    parser.add_argument(
        "--config1",
        type=str,
        default="-0.7,0.5,0.0,-2.0,0.0,2.5,0.0",
        help="Start/goal config 1 as 7 comma-separated radians.",
    )
    parser.add_argument(
        "--config2",
        type=str,
        default="0.7,0.8,0.0,-1.7,0.0,3.0,0.0",
        help="Start/goal config 2 as 7 comma-separated radians.",
    )
    parser.add_argument(
        "--start-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std (rad) added to start config each trial.",
    )
    parser.add_argument(
        "--goal-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std (rad) added to goal config each trial.",
    )
    parser.add_argument(
        "--table-points",
        type=int,
        default=1024,
        help="Number of sampled points on the table plane.",
    )
    parser.add_argument(
        "--box-points",
        type=int,
        default=1024,
        help="Number of sampled points on the obstacle box surface.",
    )
    parser.add_argument(
        "--box-center",
        type=str,
        default="0.5,0.0,0.2",
        help="Obstacle box center as x,y,z.",
    )
    parser.add_argument(
        "--box-half-extents",
        type=str,
        default="0.2,0.1,0.2",
        help="Obstacle box half extents as hx,hy,hz.",
    )
    parser.add_argument(
        "--box-jitter-xy",
        type=float,
        default=0.0,
        help="Uniform random jitter range in meters for box center x/y per trial.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N trials.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output JSON path for summary and per-trial results.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.collision_robot_points <= 0:
        raise ValueError("--collision-robot-points must be > 0")
    if args.usd_obstacle_points <= 0:
        raise ValueError("--usd-obstacle-points must be > 0")
    if args.max_demos < 0:
        raise ValueError("--max-demos must be >= 0")
    if args.max_pairs_per_demo < 0:
        raise ValueError("--max-pairs-per-demo must be >= 0")

    use_dataset_mode = bool(args.dataset_hdf5)
    if not use_dataset_mode and args.num_trials <= 0:
        raise ValueError("--num-trials must be > 0")

    if args.table_points <= 0 or args.box_points <= 0:
        raise ValueError("--table-points and --box-points must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    device = _resolve_device(args.device)
    policy = DRPInference(device=device)

    print("=" * 80)
    print("DRP batch success-rate evaluation")
    print(f"Device                : {device}")
    print(f"Max steps / trial     : {args.max_steps}")
    print(f"Goal tolerance (L2)   : {args.goal_tolerance:.4f} rad")
    print(f"Collision threshold   : {args.collision_threshold:.4f} m")
    print(f"Seed                  : {args.seed}")
    print(f"Mode                  : {'dataset_usd' if use_dataset_mode else 'synthetic'}")
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
            dataset_hdf5=dataset_hdf5,
            saved_scenes_dir=saved_scenes_dir,
            scene_prefix=scene_prefix,
        )
    else:
        config_1 = _parse_float_list(args.config1, expected_len=7, name="--config1")
        config_2 = _parse_float_list(args.config2, expected_len=7, name="--config2")
        box_center = _parse_float_list(args.box_center, expected_len=3, name="--box-center")
        box_half_extents = _parse_float_list(
            args.box_half_extents, expected_len=3, name="--box-half-extents"
        )

        print(f"Trials                : {args.num_trials}")
        print(f"Direction mode        : {args.direction}")

        results = _run_synthetic_trials(
            policy=policy,
            args=args,
            rng=rng,
            config_1=config_1,
            config_2=config_2,
            box_center=box_center,
            box_half_extents=box_half_extents,
        )

    total_runtime = time.perf_counter() - begin

    if not results:
        raise RuntimeError("No valid trials were executed. Please check dataset paths and scene files.")

    summary = _summarize(results)
    summary["total_runtime_s"] = total_runtime
    if dataset_meta is not None:
        summary.update(dataset_meta)

    print("\n" + "=" * 80)
    print("Evaluation finished")
    print(f"Success rate          : {summary['success_rate'] * 100.0:.2f}%")
    print(f"Success / Total       : {summary['success_count']} / {summary['total_trials']}")
    print(f"Failure breakdown     : {summary['failure_breakdown']}")
    print(f"Avg success steps     : {summary['avg_steps_success']}")
    print(f"Avg final joint err   : {summary['avg_final_joint_error_l2']:.4f} rad")
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
