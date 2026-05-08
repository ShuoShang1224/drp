import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from drp.drp_inference import DRPInference
from drp.drp_success_rate_inference_multicam import (
    DEFAULT_CAMERA_SOURCE_POINTS,
    DEFAULT_D435_DEPTH_FAR_M,
    DEFAULT_D435_DEPTH_HEIGHT,
    DEFAULT_D435_DEPTH_HFOV_DEG,
    DEFAULT_D435_DEPTH_NEAR_M,
    DEFAULT_D435_DEPTH_WIDTH,
    DEFAULT_OBSTACLE_POINTS,
    BulletDepthPointcloudRenderer,
    FixedCamera,
    MultiCamRolloutVisualizer,
    _make_policy_obstacle_pcd,
)
from drp.drp_success_rate_inference import (
    RolloutVisualizer,
    TrialResult,
    _eef_error_m,
    _joint_error_l1,
    _load_scene_obstacle_pcd_from_usd,
    _resolve_device,
    _scene_clearance_m,
    _summarize,
)
from drp.utils.franka_utils import clamp_to_franka_limits

try:
    import h5py
except ImportError:
    h5py = None


def _decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _demo_index_from_key(demo_key: str) -> int:
    if not demo_key.startswith("demo_"):
        raise ValueError(f"Invalid demo key format: {demo_key}")
    return int(demo_key.split("_")[-1])


def _default_dataset_hdf5() -> str:
    candidates = [
        Path("drp_data/drp_scene_all_test_merged_targets.hdf5"),
        Path("test_data/drp_scene_all_test_merged_targets.hdf5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _default_saved_scenes_dir() -> str:
    candidates = [
        Path("drp_data/saved_scenes"),
        Path("test_data/saved_scenes"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])


def _default_camera_extrinsics() -> List[str]:
    return [
        "cam/T_cam2base.npy",
        "cam/T_cam2base_cam2.npy",
        "cam/T_cam2base_cam3.npy",
        "cam/T_cam2base_cam4.npy",
    ]


def _load_cameras_from_cam2base_files(
    camera_paths: List[str],
    width: int,
    height: int,
    hfov_deg: float,
    near: float,
    far: float,
) -> List[FixedCamera]:
    cameras: List[FixedCamera] = []
    for camera_path_str in camera_paths:
        camera_path = Path(camera_path_str)
        if not camera_path.is_file():
            raise FileNotFoundError(f"Camera extrinsics file not found: {camera_path}")
        transform = np.load(camera_path)
        if transform.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform in {camera_path}, got {transform.shape}")

        position = transform[:3, 3].astype(np.float32, copy=False)
        forward = transform[:3, 2].astype(np.float32, copy=False)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-8:
            raise ValueError(f"Degenerate camera forward axis in {camera_path}")
        forward = forward / forward_norm
        target = position + 0.5 * forward
        cameras.append(
            FixedCamera(
                position=position,
                target=target.astype(np.float32, copy=False),
                width=width,
                height=height,
                hfov_deg=hfov_deg,
                near=near,
                far=far,
            )
        )
    return cameras


def _collect_demo_entries(
    f: "h5py.File",
    split_filter: str,
    scene_name_filter: str,
    max_demos: int,
) -> List[Tuple[str, str, str, int]]:
    entries: List[Tuple[str, str, str, int]] = []

    if "index" in f and "entries" in f["index"]:
        index_entries = f["index"]["entries"]
        for raw_entry in index_entries:
            split = _decode_h5_string(raw_entry["split"])
            scene_name = _decode_h5_string(raw_entry["scene_name"])
            demo_key = _decode_h5_string(raw_entry["demo_key"])
            if split_filter and split != split_filter:
                continue
            if scene_name_filter and scene_name != scene_name_filter:
                continue
            entries.append((split, scene_name, demo_key, int(raw_entry["num_pairs"])))
    else:
        if "data" not in f:
            raise KeyError("Expected group 'data' in dataset hdf5")
        for split, split_group in f["data"].items():
            if split_filter and split != split_filter:
                continue
            for scene_name, scene_group in split_group.items():
                if scene_name_filter and scene_name != scene_name_filter:
                    continue
                for demo_key, demo_group in scene_group.items():
                    pair_count = min(
                        demo_group["start_positions"].shape[0],
                        demo_group["goal_positions"].shape[0],
                    )
                    entries.append((split, scene_name, demo_key, int(pair_count)))

    entries.sort(key=lambda item: (item[0], item[1], _demo_index_from_key(item[2])))
    if max_demos > 0:
        entries = entries[:max_demos]
    return entries


def _eef_orientation_error_deg(policy: DRPInference, q: torch.Tensor, q_goal: torch.Tensor) -> float:
    pose = policy._fk_sampler.end_effector_pose(q)[0, :3, :3]
    goal_pose = policy._fk_sampler.end_effector_pose(q_goal)[0, :3, :3]
    relative = pose.T @ goal_pose
    trace = torch.trace(relative)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos_theta)).item())


def _run_trial_with_policy_obstacle(
    policy: DRPInference,
    start_q_np: np.ndarray,
    goal_q_np: np.ndarray,
    obstacle_scene,
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
                status="running",
            )
            time.sleep(args.visual_step_sleep)

        if clearance < args.collision_threshold:
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
                    status="collusion",
                )
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

        start_pos_err = _eef_error_m(policy, joint_pos, goal_joint_pos)
        start_ori_err = _eef_orientation_error_deg(policy, joint_pos, goal_joint_pos)
        if (
            start_pos_err <= args.goal_pos_tolerance_m
            and start_ori_err <= args.goal_ori_tolerance_deg
        ):
            return TrialResult(
                trial_id=trial_id,
                direction=direction_tag,
                success=True,
                reason="success",
                steps=0,
                final_joint_error_l1=_joint_error_l1(joint_pos, goal_joint_pos),
                final_eef_error_m=start_pos_err,
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
                        status="collusion",
                    )
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

            pos_err = _eef_error_m(policy, joint_pos, goal_joint_pos)
            ori_err = _eef_orientation_error_deg(policy, joint_pos, goal_joint_pos)
            if pos_err <= args.goal_pos_tolerance_m and ori_err <= args.goal_ori_tolerance_deg:
                return TrialResult(
                    trial_id=trial_id,
                    direction=direction_tag,
                    success=True,
                    reason="success",
                    steps=step,
                    final_joint_error_l1=_joint_error_l1(joint_pos, goal_joint_pos),
                    final_eef_error_m=pos_err,
                    min_clearance_m=min_clearance,
                    runtime_s=time.perf_counter() - start_time,
                )

    if visualizer is not None:
        final_robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
        final_clearance = _scene_clearance_m(final_robot_pcd, obstacle_scene)
        visualizer.update(
            trial_id=trial_id,
            direction_tag=direction_tag,
            step=args.max_steps,
            joint_pos=joint_pos,
            goal_joint_pos=goal_joint_pos,
            obstacle_pcd=policy_obstacle_pcd,
            obstacle_scene=obstacle_scene,
            current_robot_pcd=final_robot_pcd,
            clearance=final_clearance,
            status="timeout",
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


def _summarize_by_split(results: List[TrialResult]) -> Dict[str, dict]:
    buckets: Dict[str, List[TrialResult]] = defaultdict(list)
    for result in results:
        split = result.direction.split("/", 1)[0].strip().lower()
        buckets[split].append(result)
    return {split: _summarize(split_results) for split, split_results in buckets.items()}


def _run_merged_dataset_trials(
    policy: DRPInference,
    args: argparse.Namespace,
    rng: np.random.Generator,
    depth_renderer: Optional[BulletDepthPointcloudRenderer],
    dataset_hdf5: Path,
    saved_scenes_dir: Path,
    visualizer: Optional[RolloutVisualizer],
) -> Tuple[List[TrialResult], Dict[str, int]]:
    if h5py is None:
        raise RuntimeError("h5py is required for merged dataset inference mode")
    if not dataset_hdf5.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_hdf5}")
    if not saved_scenes_dir.is_dir():
        raise FileNotFoundError(f"Saved scenes directory not found: {saved_scenes_dir}")

    results: List[TrialResult] = []
    scene_cache: Dict[str, object] = {}
    missing_scene_count = 0
    parse_failed_scene_count = 0

    with h5py.File(dataset_hdf5, "r") as f:
        entries = _collect_demo_entries(
            f=f,
            split_filter=args.split.strip(),
            scene_name_filter=args.scene_name.strip(),
            max_demos=args.max_demos,
        )

        total_pairs = 0
        for split, scene_name, demo_key, indexed_pair_count in entries:
            group = f["data"][split][scene_name][demo_key]
            pair_count = min(
                indexed_pair_count,
                group["start_positions"].shape[0],
                group["goal_positions"].shape[0],
            )
            if args.max_pairs_per_demo > 0:
                pair_count = min(pair_count, args.max_pairs_per_demo)
            total_pairs += pair_count

        print("=" * 80)
        print("DRP merged-dataset success-rate evaluation")
        print(f"Dataset hdf5          : {dataset_hdf5}")
        print(f"Saved scenes dir      : {saved_scenes_dir}")
        print(f"Split filter          : {args.split or '<all>'}")
        print(f"Scene filter          : {args.scene_name or '<all>'}")
        print(f"Demo groups           : {len(entries)}")
        print(f"Total start-goal pairs: {total_pairs}")
        print(f"Obstacle input mode   : {args.obstacle_input_mode}")
        if args.obstacle_input_mode == "sim_camera":
            print(f"Camera count          : {len(args.camera_extrinsics)}")
            print(f"Include robot in depth: {args.include_robot_in_depth}")
        print(f"Max steps / trial     : {args.max_steps}")
        print(f"Success pos tol       : {args.goal_pos_tolerance_m * 100.0:.2f} cm")
        print(f"Success ori tol       : {args.goal_ori_tolerance_deg:.2f} deg")
        print(f"SDF collision thresh  : {args.collision_threshold:.4f} m")
        print(f"USD obstacle points   : {args.usd_obstacle_points}")
        print(f"Policy obstacle pts   : {args.obstacle_points}")
        print("=" * 80)

        trial_id = 1
        for demo_pos, (split, scene_name, demo_key, _) in enumerate(entries, start=1):
            scene_file_stem = f"{scene_name}_{demo_key}"
            scene_path = saved_scenes_dir / f"{scene_file_stem}.usd"

            if not scene_path.is_file():
                missing_scene_count += 1
                if args.print_every > 0:
                    print(
                        f"[warn] missing scene for {split}/{scene_name}/{demo_key}: "
                        f"{scene_path.name}"
                    )
                continue

            if scene_file_stem not in scene_cache:
                try:
                    scene_cache[scene_file_stem] = _load_scene_obstacle_pcd_from_usd(
                        scene_path=scene_path,
                        device=policy.device,
                        total_points=(
                            args.depth_source_points
                            if args.obstacle_input_mode == "sim_camera"
                            else args.usd_obstacle_points
                        ),
                        rng=rng,
                    )
                except Exception as exc:
                    parse_failed_scene_count += 1
                    if args.print_every > 0:
                        print(f"[warn] failed to parse {scene_path.name}: {exc}")
                    continue

            group = f["data"][split][scene_name][demo_key]
            starts = np.asarray(group["start_positions"], dtype=np.float32)
            goals = np.asarray(group["goal_positions"], dtype=np.float32)

            pair_count = min(starts.shape[0], goals.shape[0])
            if args.max_pairs_per_demo > 0:
                pair_count = min(pair_count, args.max_pairs_per_demo)

            for pair_idx in range(pair_count):
                start_q, _ = clamp_to_franka_limits(starts[pair_idx, :7])
                goal_q, _ = clamp_to_franka_limits(goals[pair_idx, :7])
                start_joint_pos = torch.as_tensor(
                    start_q,
                    device=policy.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                start_robot_pcd = policy._fk_sampler.sample(
                    start_joint_pos,
                    args.collision_robot_points,
                )[0]
                policy_obstacle_pcd = _make_policy_obstacle_pcd(
                    obstacle_scene=scene_cache[scene_file_stem],
                    use_depth=(args.obstacle_input_mode == "sim_camera"),
                    depth_renderer=depth_renderer,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    obstacle_points=args.obstacle_points,
                    depth_voxel_size=args.depth_voxel_size,
                    allow_depth_fallback_to_global=args.allow_depth_fallback_to_global,
                    include_robot_in_depth=args.include_robot_in_depth,
                    robot_pcd_for_removal=start_robot_pcd if args.include_robot_in_depth else None,
                    robot_removal_radius=args.robot_removal_radius,
                    rng=rng,
                )

                result = _run_trial_with_policy_obstacle(
                    policy=policy,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    goal_q_np=np.asarray(goal_q, dtype=np.float32),
                    obstacle_scene=scene_cache[scene_file_stem],
                    policy_obstacle_pcd=policy_obstacle_pcd,
                    args=args,
                    trial_id=trial_id,
                    direction_tag=f"{split}/{scene_name}/{demo_key}/pair_{pair_idx}",
                    visualizer=visualizer,
                )
                results.append(result)
                trial_id += 1

                if args.print_every > 0 and len(results) % args.print_every == 0:
                    success_so_far = sum(r.success for r in results)
                    print(
                        f"[pair {len(results):>4d}/{total_pairs}] "
                        f"split={split:<4s} scene={scene_name:<18s} demo={demo_key:<7s} "
                        f"pair={pair_idx:<2d} "
                        f"success={success_so_far}/{len(results)} "
                        f"({100.0 * success_so_far / len(results):.1f}%) "
                        f"last={result.reason:<18s} "
                        f"steps={result.steps:>3d}"
                    )

            if args.print_every > 0 and demo_pos % max(args.print_every, 1) == 0:
                print(f"[demo {demo_pos:>4d}/{len(entries)}] processed")

        meta: Dict[str, int] = {
            "missing_scenes": missing_scene_count,
            "parse_failed_scenes": parse_failed_scene_count,
            "cached_scenes": len(scene_cache),
            "demo_groups": len(entries),
            "total_pairs": total_pairs,
        }
        return results, meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch DRP inference evaluator for merged final-test HDF5."
    )
    parser.add_argument(
        "--difficulty",
        choices=("easy", "hard", "all"),
        default="all",
        help="Run only easy scenes, only hard scenes, or all scenes.",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Disable Viser visualization and run in headless mode.",
    )
    parser.add_argument(
        "--gpd",
        action="store_true",
        help="Use global pointcloud input instead of simulated multi-camera depth pointclouds.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.dataset_hdf5 = _default_dataset_hdf5()
    args.saved_scenes_dir = _default_saved_scenes_dir()
    args.split = "" if args.difficulty == "all" else args.difficulty
    args.scene_name = ""
    args.max_demos = 0
    args.max_pairs_per_demo = 0
    args.usd_obstacle_points = 2048
    args.obstacle_input_mode = "global_pointcloud" if args.gpd else "sim_camera"
    args.obstacle_points = DEFAULT_OBSTACLE_POINTS
    args.allow_depth_fallback_to_global = False
    args.camera_extrinsics = _default_camera_extrinsics()
    args.camera_width = DEFAULT_D435_DEPTH_WIDTH
    args.camera_height = DEFAULT_D435_DEPTH_HEIGHT
    args.camera_hfov_deg = DEFAULT_D435_DEPTH_HFOV_DEG
    args.camera_near = DEFAULT_D435_DEPTH_NEAR_M
    args.camera_far = DEFAULT_D435_DEPTH_FAR_M
    args.depth_source_points = DEFAULT_CAMERA_SOURCE_POINTS
    args.depth_voxel_size = 0.003
    args.include_robot_in_depth = True
    args.robot_removal_radius = 0.03
    args.max_steps = 300
    args.max_joint_step = 0.08
    args.goal_pos_tolerance_m = 0.01
    args.goal_ori_tolerance_deg = 15.0
    args.collision_threshold = 0.008
    args.collision_robot_points = 1024
    args.seed = 0
    args.device = "cuda"
    args.disable_torch_compile = False
    args.print_every = 10
    args.save_json = ""
    args.visualize_step_sleep = 0.05
    args.visualize_hold = False

    # Keep compatibility with the rollout helper reused from
    # drp_success_rate_inference_multicam.py.
    args.visual_step_sleep = args.visualize_step_sleep
    args.visual_final_sleep = 0.0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    device = _resolve_device(args.device)
    policy = DRPInference(device=device, compile_model=not args.disable_torch_compile)
    cameras = _load_cameras_from_cam2base_files(
        camera_paths=[str(Path(path)) for path in args.camera_extrinsics],
        width=args.camera_width,
        height=args.camera_height,
        hfov_deg=args.camera_hfov_deg,
        near=args.camera_near,
        far=args.camera_far,
    )
    depth_renderer = (
        BulletDepthPointcloudRenderer(cameras)
        if args.obstacle_input_mode == "sim_camera"
        else None
    )
    use_visualizer = not args.skip_visual
    visualizer = (
        MultiCamRolloutVisualizer(
            policy=policy,
            robot_points=args.collision_robot_points,
            cameras=cameras,
        )
        if use_visualizer
        else None
    )

    print("=" * 80)
    print("DRP merged final-test evaluation")
    print(f"Device                : {device}")
    print(f"Dataset hdf5          : {args.dataset_hdf5}")
    print(f"Saved scenes dir      : {args.saved_scenes_dir}")
    print(f"Seed                  : {args.seed}")
    print(f"Obstacle input mode   : {args.obstacle_input_mode}")
    print(f"Camera refresh        : {'start-only' if args.obstacle_input_mode == 'sim_camera' else 'static'}")
    print(f"Camera count          : {len(cameras)}")
    print(f"Include robot in depth: {args.include_robot_in_depth}")
    print(f"Torch compile         : {not args.disable_torch_compile}")
    print(f"Visualize             : {use_visualizer}")
    print("=" * 80)

    begin = time.perf_counter()
    results, dataset_meta = _run_merged_dataset_trials(
        policy=policy,
        args=args,
        rng=rng,
        depth_renderer=depth_renderer,
        dataset_hdf5=Path(args.dataset_hdf5),
        saved_scenes_dir=Path(args.saved_scenes_dir),
        visualizer=visualizer,
    )
    total_runtime = time.perf_counter() - begin

    if not results:
        raise RuntimeError("No valid trials were executed. Please check dataset paths and scene files.")

    summary = _summarize(results)
    summary["total_runtime_s"] = total_runtime
    summary["obstacle_input_mode"] = args.obstacle_input_mode
    summary["camera_count"] = len(cameras)
    summary.update(dataset_meta)
    split_summaries = _summarize_by_split(results)

    print("\n" + "=" * 80)
    print("Evaluation finished")
    easy_summary = split_summaries.get("easy")
    hard_summary = split_summaries.get("hard")
    if easy_summary:
        print(
            f"Easy success rate     : {easy_summary['success_rate'] * 100.0:.2f}% "
            f"({easy_summary['success_count']} / {easy_summary['total_trials']})"
        )
    if hard_summary:
        print(
            f"Hard success rate     : {hard_summary['success_rate'] * 100.0:.2f}% "
            f"({hard_summary['success_count']} / {hard_summary['total_trials']})"
        )
    print(
        f"Total success rate    : {summary['success_rate'] * 100.0:.2f}% "
        f"({summary['success_count']} / {summary['total_trials']})"
    )
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
            "split_summaries": split_summaries,
            "results": [asdict(r) for r in results],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved detailed results to: {output_path}")

    if args.visualize_hold and visualizer is not None:
        print("Visualizer hold is enabled. Press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
