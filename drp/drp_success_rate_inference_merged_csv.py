import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from drp.drp_inference import DRPInference
from drp.drp_success_rate_inference import (
    RolloutVisualizer,
    _load_scene_obstacle_pcd_from_usd,
    _resolve_device,
    _scene_clearance_m,
    _summarize,
)
from drp.drp_success_rate_inference_merged import (
    _collect_demo_entries,
    _default_camera_extrinsics,
    _default_dataset_hdf5,
    _default_saved_scenes_dir,
    _eef_orientation_error_deg,
    _load_cameras_from_cam2base_files,
    _summarize_by_split,
)
from drp.drp_success_rate_inference_multicam import (
    DEFAULT_CAMERA_SOURCE_POINTS,
    DEFAULT_D435_DEPTH_FAR_M,
    DEFAULT_D435_DEPTH_HEIGHT,
    DEFAULT_D435_DEPTH_HFOV_DEG,
    DEFAULT_D435_DEPTH_NEAR_M,
    DEFAULT_D435_DEPTH_WIDTH,
    DEFAULT_OBSTACLE_POINTS,
    BulletDepthPointcloudRenderer,
    MultiCamRolloutVisualizer,
    _make_policy_obstacle_pcd,
)
from drp.utils.franka_utils import clamp_to_franka_limits

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class DetailedTrialResult:
    trial_id: int
    split: str
    scene_name: str
    demo_key: str
    pair_index: int
    direction: str
    status: str
    success: bool
    steps: int
    runtime_s: float
    start_in_collision: bool
    collided_during_rollout: bool
    timed_out: bool
    reached_pos_1cm: bool
    reached_ori_3deg: bool
    reached_ori_15deg: bool
    reached_success_1cm_15deg: bool
    final_joint_error_l1_rad: float
    final_eef_pos_error_m: float
    final_eef_pos_error_cm: float
    final_eef_ori_error_deg: float
    min_clearance_m: float
    final_clearance_m: float
    obstacle_input_mode: str


def _eef_pos_error_m(policy: DRPInference, q: torch.Tensor, q_goal: torch.Tensor) -> float:
    eef = policy._fk_sampler.end_effector_pose(q)[0, :3, 3]
    eef_goal = policy._fk_sampler.end_effector_pose(q_goal)[0, :3, 3]
    return float(torch.linalg.norm(eef - eef_goal).item())


def _joint_error_l1(q: torch.Tensor, q_goal: torch.Tensor) -> float:
    return float(torch.abs(q - q_goal).sum(dim=1).item())


def _build_detailed_result(
    *,
    policy: DRPInference,
    joint_pos: torch.Tensor,
    goal_joint_pos: torch.Tensor,
    split: str,
    scene_name: str,
    demo_key: str,
    pair_index: int,
    trial_id: int,
    direction: str,
    status: str,
    steps: int,
    runtime_s: float,
    min_clearance_m: float,
    final_clearance_m: float,
    obstacle_input_mode: str,
) -> DetailedTrialResult:
    final_pos_error_m = _eef_pos_error_m(policy, joint_pos, goal_joint_pos)
    final_ori_error_deg = _eef_orientation_error_deg(policy, joint_pos, goal_joint_pos)
    reached_pos_1cm = final_pos_error_m <= 0.01
    reached_ori_3deg = final_ori_error_deg <= 3.0
    reached_ori_15deg = final_ori_error_deg <= 15.0
    return DetailedTrialResult(
        trial_id=trial_id,
        split=split,
        scene_name=scene_name,
        demo_key=demo_key,
        pair_index=pair_index,
        direction=direction,
        status=status,
        success=(status == "success"),
        steps=steps,
        runtime_s=runtime_s,
        start_in_collision=(status == "start_in_collision"),
        collided_during_rollout=(status == "collision"),
        timed_out=(status == "timeout"),
        reached_pos_1cm=reached_pos_1cm,
        reached_ori_3deg=reached_ori_3deg,
        reached_ori_15deg=reached_ori_15deg,
        reached_success_1cm_15deg=(reached_pos_1cm and reached_ori_15deg),
        final_joint_error_l1_rad=_joint_error_l1(joint_pos, goal_joint_pos),
        final_eef_pos_error_m=final_pos_error_m,
        final_eef_pos_error_cm=final_pos_error_m * 100.0,
        final_eef_ori_error_deg=final_ori_error_deg,
        min_clearance_m=min_clearance_m,
        final_clearance_m=final_clearance_m,
        obstacle_input_mode=obstacle_input_mode,
    )


def _run_trial_detailed(
    *,
    policy: DRPInference,
    start_q_np: np.ndarray,
    goal_q_np: np.ndarray,
    obstacle_scene,
    policy_obstacle_pcd: torch.Tensor,
    args: argparse.Namespace,
    trial_id: int,
    split: str,
    scene_name: str,
    demo_key: str,
    pair_index: int,
    direction_tag: str,
    visualizer: Optional[RolloutVisualizer],
) -> DetailedTrialResult:
    device = policy.device
    joint_pos = torch.as_tensor(start_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    goal_joint_pos = torch.as_tensor(goal_q_np, device=device, dtype=torch.float32).unsqueeze(0)
    max_step = torch.full_like(joint_pos, args.max_joint_step)

    start_time = time.perf_counter()
    min_clearance = float("inf")
    final_clearance = float("inf")

    with torch.inference_mode():
        start_robot_pcd = policy._fk_sampler.sample(joint_pos, args.collision_robot_points)[0]
        clearance = _scene_clearance_m(start_robot_pcd, obstacle_scene)
        min_clearance = min(min_clearance, clearance)
        final_clearance = clearance

        if visualizer is not None:
            visualizer.update(
                trial_id=trial_id,
                direction_tag=direction_tag,
                step=0,
                joint_pos=joint_pos,
                goal_joint_pos=goal_joint_pos,
                obstacle_pcd=policy_obstacle_pcd,
                current_robot_pcd=start_robot_pcd,
                clearance=clearance,
                status="start",
            )
            time.sleep(args.visual_step_sleep)

        if clearance < args.collision_threshold:
            return _build_detailed_result(
                policy=policy,
                joint_pos=joint_pos,
                goal_joint_pos=goal_joint_pos,
                split=split,
                scene_name=scene_name,
                demo_key=demo_key,
                pair_index=pair_index,
                trial_id=trial_id,
                direction=direction_tag,
                status="start_in_collision",
                steps=0,
                runtime_s=time.perf_counter() - start_time,
                min_clearance_m=min_clearance,
                final_clearance_m=final_clearance,
                obstacle_input_mode=args.obstacle_input_mode,
            )

        start_pos_err = _eef_pos_error_m(policy, joint_pos, goal_joint_pos)
        start_ori_err = _eef_orientation_error_deg(policy, joint_pos, goal_joint_pos)
        if start_pos_err <= args.goal_pos_tolerance_m and start_ori_err <= args.goal_ori_tolerance_deg:
            return _build_detailed_result(
                policy=policy,
                joint_pos=joint_pos,
                goal_joint_pos=goal_joint_pos,
                split=split,
                scene_name=scene_name,
                demo_key=demo_key,
                pair_index=pair_index,
                trial_id=trial_id,
                direction=direction_tag,
                status="success",
                steps=0,
                runtime_s=time.perf_counter() - start_time,
                min_clearance_m=min_clearance,
                final_clearance_m=final_clearance,
                obstacle_input_mode=args.obstacle_input_mode,
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
            final_clearance = clearance

            if visualizer is not None:
                visualizer.update(
                    trial_id=trial_id,
                    direction_tag=direction_tag,
                    step=step,
                    joint_pos=joint_pos,
                    goal_joint_pos=goal_joint_pos,
                    obstacle_pcd=policy_obstacle_pcd,
                    current_robot_pcd=robot_pcd,
                    clearance=clearance,
                    status="running",
                )
                time.sleep(args.visual_step_sleep)

            if clearance < args.collision_threshold:
                return _build_detailed_result(
                    policy=policy,
                    joint_pos=joint_pos,
                    goal_joint_pos=goal_joint_pos,
                    split=split,
                    scene_name=scene_name,
                    demo_key=demo_key,
                    pair_index=pair_index,
                    trial_id=trial_id,
                    direction=direction_tag,
                    status="collision",
                    steps=step,
                    runtime_s=time.perf_counter() - start_time,
                    min_clearance_m=min_clearance,
                    final_clearance_m=final_clearance,
                    obstacle_input_mode=args.obstacle_input_mode,
                )

            pos_err = _eef_pos_error_m(policy, joint_pos, goal_joint_pos)
            ori_err = _eef_orientation_error_deg(policy, joint_pos, goal_joint_pos)
            if pos_err <= args.goal_pos_tolerance_m and ori_err <= args.goal_ori_tolerance_deg:
                return _build_detailed_result(
                    policy=policy,
                    joint_pos=joint_pos,
                    goal_joint_pos=goal_joint_pos,
                    split=split,
                    scene_name=scene_name,
                    demo_key=demo_key,
                    pair_index=pair_index,
                    trial_id=trial_id,
                    direction=direction_tag,
                    status="success",
                    steps=step,
                    runtime_s=time.perf_counter() - start_time,
                    min_clearance_m=min_clearance,
                    final_clearance_m=final_clearance,
                    obstacle_input_mode=args.obstacle_input_mode,
                )

    return _build_detailed_result(
        policy=policy,
        joint_pos=joint_pos,
        goal_joint_pos=goal_joint_pos,
        split=split,
        scene_name=scene_name,
        demo_key=demo_key,
        pair_index=pair_index,
        trial_id=trial_id,
        direction=direction_tag,
        status="timeout",
        steps=args.max_steps,
        runtime_s=time.perf_counter() - start_time,
        min_clearance_m=min_clearance,
        final_clearance_m=final_clearance,
        obstacle_input_mode=args.obstacle_input_mode,
    )


def _write_csv(output_path: Path, rows: List[DetailedTrialResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _run_merged_dataset_trials(
    policy: DRPInference,
    args: argparse.Namespace,
    rng: np.random.Generator,
    depth_renderer: Optional[BulletDepthPointcloudRenderer],
    dataset_hdf5: Path,
    saved_scenes_dir: Path,
    visualizer: Optional[RolloutVisualizer],
) -> Tuple[List[DetailedTrialResult], Dict[str, int]]:
    if h5py is None:
        raise RuntimeError("h5py is required for merged dataset inference mode")
    if not dataset_hdf5.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_hdf5}")
    if not saved_scenes_dir.is_dir():
        raise FileNotFoundError(f"Saved scenes directory not found: {saved_scenes_dir}")

    results: List[DetailedTrialResult] = []
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
        print("DRP merged-dataset CSV evaluation")
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
        print(f"CSV output            : {args.csv_output}")
        print("=" * 80)

        trial_id = 1
        for demo_pos, (split, scene_name, demo_key, _) in enumerate(entries, start=1):
            scene_file_stem = f"{scene_name}_{demo_key}"
            scene_path = saved_scenes_dir / f"{scene_file_stem}.usd"

            if not scene_path.is_file():
                missing_scene_count += 1
                if args.print_every > 0:
                    print(f"[warn] missing scene for {split}/{scene_name}/{demo_key}: {scene_path.name}")
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
                    start_q, device=policy.device, dtype=torch.float32
                ).unsqueeze(0)
                start_robot_pcd = policy._fk_sampler.sample(
                    start_joint_pos, args.collision_robot_points
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

                result = _run_trial_detailed(
                    policy=policy,
                    start_q_np=np.asarray(start_q, dtype=np.float32),
                    goal_q_np=np.asarray(goal_q, dtype=np.float32),
                    obstacle_scene=scene_cache[scene_file_stem],
                    policy_obstacle_pcd=policy_obstacle_pcd,
                    args=args,
                    trial_id=trial_id,
                    split=split,
                    scene_name=scene_name,
                    demo_key=demo_key,
                    pair_index=pair_idx,
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
                        f"pair={pair_idx:<2d} success={success_so_far}/{len(results)} "
                        f"({100.0 * success_so_far / len(results):.1f}%) "
                        f"last={result.status:<18s} steps={result.steps:>3d}"
                    )

            if args.print_every > 0 and demo_pos % max(args.print_every, 1) == 0:
                print(f"[demo {demo_pos:>4d}/{len(entries)}] processed")

        meta = {
            "missing_scenes": missing_scene_count,
            "parse_failed_scenes": parse_failed_scene_count,
            "cached_scenes": len(scene_cache),
            "demo_groups": len(entries),
            "total_pairs": total_pairs,
        }
        return results, meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch DRP merged-dataset evaluator that exports per-pair CSV diagnostics."
    )
    parser.add_argument("--skip-visual", action="store_true", help="Disable Viser visualization.")
    parser.add_argument(
        "--gpd",
        action="store_true",
        help="Use global pointcloud input instead of simulated multi-camera depth pointclouds.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="outputs/drp_merged_pair_metrics_3.csv",
        help="Path to the CSV file with one row per start-goal pair.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.dataset_hdf5 = _default_dataset_hdf5()
    args.saved_scenes_dir = _default_saved_scenes_dir()
    args.split = ""
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
    args.goal_ori_tolerance_deg = 3.0
    args.collision_threshold = 0.008
    args.collision_robot_points = 1024
    args.seed = 0
    args.device = "cuda"
    args.disable_torch_compile = False
    args.print_every = 10
    args.visualize_step_sleep = 0.05
    args.visualize_hold = False
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
    print("DRP merged CSV diagnostics evaluation")
    print(f"Device                : {device}")
    print(f"Dataset hdf5          : {args.dataset_hdf5}")
    print(f"Saved scenes dir      : {args.saved_scenes_dir}")
    print(f"Obstacle input mode   : {args.obstacle_input_mode}")
    print(f"Camera refresh        : {'start-only' if args.obstacle_input_mode == 'sim_camera' else 'static'}")
    print(f"CSV output            : {args.csv_output}")
    print(f"Visualize             : {use_visualizer}")
    print("=" * 80)

    begin = time.perf_counter()
    detailed_results, dataset_meta = _run_merged_dataset_trials(
        policy=policy,
        args=args,
        rng=rng,
        depth_renderer=depth_renderer,
        dataset_hdf5=Path(args.dataset_hdf5),
        saved_scenes_dir=Path(args.saved_scenes_dir),
        visualizer=visualizer,
    )
    total_runtime = time.perf_counter() - begin

    if not detailed_results:
        raise RuntimeError("No valid trials were executed. Please check dataset paths and scene files.")

    csv_path = Path(args.csv_output)
    _write_csv(csv_path, detailed_results)

    summary_input = [
        type(
            "SummaryTrialResult",
            (),
            {
                "trial_id": row.trial_id,
                "direction": row.direction,
                "success": row.success,
                "reason": row.status,
                "steps": row.steps,
                "final_joint_error_l1": row.final_joint_error_l1_rad,
                "final_eef_error_m": row.final_eef_pos_error_m,
                "min_clearance_m": row.min_clearance_m,
                "runtime_s": row.runtime_s,
            },
        )()
        for row in detailed_results
    ]
    summary = _summarize(summary_input)
    summary["total_runtime_s"] = total_runtime
    summary.update(dataset_meta)
    split_summaries = _summarize_by_split(summary_input)

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
    print(f"Avg final eef err     : {summary['avg_final_eef_error_m']:.4f} m")
    print(f"CSV saved             : {csv_path}")
    print("=" * 80)

    if args.visualize_hold and visualizer is not None:
        print("Visualizer hold is enabled. Press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
