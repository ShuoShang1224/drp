import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from drp.drp_inference import DRPInference
from drp.utils.franka_utils import clamp_to_franka_limits


def _load_joint_vector(payload: dict, keys: Tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in payload:
            values = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            if values.shape[0] < 7:
                raise ValueError(f"Key '{key}' must contain at least 7 values")
            return values[:7]
    raise KeyError(f"None of the keys {keys} found in payload")


def _load_runtime_state(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    joint_pos = _load_joint_vector(payload, ("joint_pos", "current_joint_pos", "current_angles", "q"))
    goal_pos = _load_joint_vector(payload, ("goal_joint_pos", "goal_angles", "goal_q"))
    joint_pos, _ = clamp_to_franka_limits(joint_pos.astype(np.float32))
    goal_pos, _ = clamp_to_franka_limits(goal_pos.astype(np.float32))
    return np.asarray(joint_pos, dtype=np.float32), np.asarray(goal_pos, dtype=np.float32)


def _write_command_output(
    output_path: Path,
    predicted_q: np.ndarray,
    commanded_q: np.ndarray,
    source_frame_idx: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_s": time.time(),
        "source_frame_idx": int(source_frame_idx),
        "predicted_joint_target": np.asarray(predicted_q, dtype=np.float32).tolist(),
        "command_joint_target": np.asarray(commanded_q, dtype=np.float32).tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DRP inference on externally provided pointcloud input."
    )
    parser.add_argument("--pointcloud", type=str, default="outputs/realsense_merged_pointcloud.npz")
    parser.add_argument("--joint-state", type=str, required=True, help="JSON file containing current and goal joint states.")
    parser.add_argument("--output", type=str, default="outputs/drp_command.json")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--min-points", type=int, default=128)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.loop_hz <= 0:
        raise ValueError("--loop-hz must be > 0")
    if args.max_iterations < 0:
        raise ValueError("--max-iterations must be >= 0")
    if args.min_points < 0:
        raise ValueError("--min-points must be >= 0")

    pointcloud_path = Path(args.pointcloud)
    joint_state_path = Path(args.joint_state)
    output_path = Path(args.output)
    device = _resolve_device(args.device)
    policy = DRPInference(device=device, compile_model=not args.disable_torch_compile)
    loop_period = 1.0 / float(args.loop_hz)

    print("=" * 80)
    print("DRP pointcloud inference server")
    print(f"Pointcloud input      : {pointcloud_path}")
    print(f"Joint state input     : {joint_state_path}")
    print(f"Command output        : {output_path}")
    print(f"Device                : {device}")
    print("=" * 80)

    loop_idx = 0
    last_frame_idx = -1
    while args.max_iterations == 0 or loop_idx < args.max_iterations:
        tic = time.perf_counter()
        if not pointcloud_path.is_file() or not joint_state_path.is_file():
            time.sleep(min(loop_period, 0.1))
            continue

        payload = np.load(pointcloud_path)
        points = np.asarray(payload["points"], dtype=np.float32)
        frame_idx = int(np.asarray(payload["frame_idx"]).item())

        if frame_idx == last_frame_idx:
            time.sleep(min(loop_period, 0.02))
            continue
        last_frame_idx = frame_idx

        if points.shape[0] < args.min_points:
            if args.print_every > 0 and loop_idx % args.print_every == 0:
                print(f"[loop {loop_idx:>5d}] skipped: only {points.shape[0]} point(s)")
            loop_idx += 1
            continue

        joint_pos_np, goal_pos_np = _load_runtime_state(joint_state_path)
        joint_pos = torch.as_tensor(joint_pos_np, device=device, dtype=torch.float32).unsqueeze(0)
        goal_joint_pos = torch.as_tensor(goal_pos_np, device=device, dtype=torch.float32).unsqueeze(0)
        obstacle_pcd = torch.as_tensor(points, device=device, dtype=torch.float32)

        env_obs = {
            "joint_pos": joint_pos,
            "goal_joint_pos": goal_joint_pos,
            "combined_obstacle_pcd": [obstacle_pcd],
        }
        predicted_q = policy.get_actions(env_obs)
        max_step = torch.full_like(joint_pos, args.max_joint_step)
        commanded_q = joint_pos + torch.clamp(predicted_q - joint_pos, -max_step, max_step)
        commanded_q, _ = clamp_to_franka_limits(commanded_q)

        _write_command_output(
            output_path=output_path,
            predicted_q=predicted_q[0].detach().cpu().numpy().astype(np.float32, copy=False),
            commanded_q=commanded_q[0].detach().cpu().numpy().astype(np.float32, copy=False),
            source_frame_idx=frame_idx,
        )

        if args.print_every > 0 and loop_idx % args.print_every == 0:
            print(
                f"[loop {loop_idx:>5d}] frame={frame_idx:>5d} "
                f"points={points.shape[0]:>5d}"
            )

        loop_idx += 1
        sleep_time = loop_period - (time.perf_counter() - tic)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
