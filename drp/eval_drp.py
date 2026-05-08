"""
Real-world DRP evaluation entrypoint, following the NeuralMP deployment flow.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from robofin.robots import FrankaRobot

from drp.drp_inference import DRPInference
from drp.utils.franka_utils import clamp_to_franka_limits


def _resolve_neural_mp_root(explicit_root: Optional[str]) -> Path:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    candidates.append(Path("/home/hanyu/neuralmotionplanner"))
    candidates.append(Path(__file__).resolve().parents[2] / "neuralmotionplanner")

    for candidate in candidates:
        if (candidate / "neural_mp").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the neuralmotionplanner repo. "
        "Pass --neural-mp-root /path/to/neuralmotionplanner."
    )


def _import_neural_mp_modules(neural_mp_root: Path):
    root_str = str(neural_mp_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from neural_mp.envs.franka_real_env import FrankaRealEnvManimo
    from neural_mp.real_evals.eval_base import rw_eval

    return FrankaRealEnvManimo, rw_eval


class DRPRealEvalAgent:
    def __init__(
        self,
        env,
        *,
        cache_dir: Path,
        device: torch.device,
        disable_torch_compile: bool,
        max_rollout_len: int,
        max_joint_step: float,
        goal_pos_tolerance_m: float,
        goal_ori_tolerance_deg: float,
    ):
        self.env = env
        self.cache_dir = cache_dir
        self.device = device
        self.max_rollout_len = int(max_rollout_len)
        self.max_joint_step = float(max_joint_step)
        self.goal_pos_tolerance_m = float(goal_pos_tolerance_m)
        self.goal_ori_tolerance_deg = float(goal_ori_tolerance_deg)
        self.policy = DRPInference(device=device, compile_model=not disable_torch_compile)
        self.goal_config = None
        self.goal_pose = None

    def setup_configs(self, start_config=None, goal_config=None) -> None:
        self.start_config = None if start_config is None else np.asarray(start_config, dtype=np.float32)
        self.goal_config = None if goal_config is None else np.asarray(goal_config, dtype=np.float32)
        self.goal_pose = (
            None
            if self.goal_config is None
            else FrankaRobot.fk(self.goal_config, eff_frame="right_gripper")
        )

    def get_scene_pcd(
        self,
        *,
        use_cache: bool = False,
        cache_name: Optional[str] = None,
        debug_raw_pcd: bool = False,
        debug_combined_pcd: bool = False,
        save_pcd: bool = False,
        save_file_name: str = "combined",
        denoise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input("press Enter to collect pcd...")

        if use_cache:
            if not cache_name:
                raise ValueError("--cache-name is required when --use-cache is set")
            points_path = self.cache_dir / f"{cache_name}_pcd.npy"
            colors_path = self.cache_dir / f"{cache_name}_rgb.npy"
            points = np.load(points_path)
            colors = np.load(colors_path)
            return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)

        least_occlusion_config = np.array([0.0, -0.45, 0.0, -1.0, 0.0, 1.9, 0.7], dtype=np.float32)
        if getattr(self.env, "ctrl_hz", None) is not None:
            self.env.move_robot_to_joint_state(joint_state=least_occlusion_config, time_to_go=4)

        points, colors = self.env.get_scene_pcd(
            debug_raw_pcd=debug_raw_pcd,
            debug_combined_pcd=debug_combined_pcd,
            save_pcd=save_pcd,
            save_file_name=save_file_name,
            denoise=denoise,
        )

        if getattr(self.env, "ctrl_hz", None) is not None:
            self.env.reset()

        return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)

    @staticmethod
    def _pose_errors(config: np.ndarray, goal_pose) -> Tuple[float, float]:
        eff_pose = FrankaRobot.fk(config, eff_frame="right_gripper")
        pos_err = float(np.linalg.norm(eff_pose._xyz - goal_pose._xyz))
        ori_err = float(
            np.abs(np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians))
        )
        return pos_err, ori_err

    @torch.inference_mode()
    def motion_plan(self, start_config, goal_config, points, colors):
        del colors

        start_config = np.asarray(start_config, dtype=np.float32).reshape(7)
        goal_config = np.asarray(goal_config, dtype=np.float32).reshape(7)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        start_config, _ = clamp_to_franka_limits(start_config)
        goal_config, _ = clamp_to_franka_limits(goal_config)
        self.setup_configs(start_config=start_config, goal_config=goal_config)

        joint_pos = torch.as_tensor(start_config, device=self.device, dtype=torch.float32).unsqueeze(0)
        goal_joint_pos = torch.as_tensor(goal_config, device=self.device, dtype=torch.float32).unsqueeze(0)
        obstacle_pcd = torch.as_tensor(points, device=self.device, dtype=torch.float32)

        planning_success = False
        trajectory = [start_config.copy()]
        rollout_steps = 0

        tic = time.perf_counter()
        for step_idx in range(self.max_rollout_len):
            env_obs = {
                "joint_pos": joint_pos,
                "goal_joint_pos": goal_joint_pos,
                "combined_obstacle_pcd": [obstacle_pcd],
            }
            predicted_q = self.policy.get_actions(env_obs)
            max_step = torch.full_like(joint_pos, self.max_joint_step)
            joint_pos = joint_pos + torch.clamp(predicted_q - joint_pos, -max_step, max_step)
            joint_pos, _ = clamp_to_franka_limits(joint_pos)

            q_np = joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=False)
            trajectory.append(q_np.copy())
            rollout_steps = step_idx + 1

            pos_err, ori_err = self._pose_errors(q_np, self.goal_pose)
            if pos_err < self.goal_pos_tolerance_m and ori_err < self.goal_ori_tolerance_deg:
                planning_success = True
                break
        toc = time.perf_counter()

        if rollout_steps == 0:
            rollout_steps = 1
        ave_rollout_time = (toc - tic) / rollout_steps
        final_pos_err, final_ori_err = self._pose_errors(trajectory[-1], self.goal_pose)
        print(
            "sim results:\n"
            f"step: {rollout_steps}\n"
            f"pos_err: {final_pos_err * 100.0:.3f} cm\n"
            f"ori_err: {final_ori_err:.3f} deg"
        )

        return np.asarray(trajectory, dtype=np.float32), planning_success, ave_rollout_time

    def motion_plan_with_tto(self, start_config, goal_config, points, colors, batch_size=100):
        del batch_size
        print("DRP does not use test-time optimization; falling back to single rollout.")
        return self.motion_plan(start_config, goal_config, points, colors)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DRP on the real Franka deployment stack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m drp.eval_drp --cfg-set scene1_l1\n"
            "  python -m drp.eval_drp --cfg-set scene1_l1 --use-cache --cache-name scene1_single_blcok\n"
            "  python -m drp.eval_drp --cfg-set scene1_l1 --device cpu --cam-only\n\n"
            "Flow:\n"
            "  1. Launch the neural_mp real-world Franka environment.\n"
            "  2. Collect or load a multi-camera scene point cloud.\n"
            "  3. Roll out DRP in joint space for each start/goal pair.\n"
            "  4. Reuse neural_mp's interactive real-world execution and CSV logging loop."
        ),
    )
    parser.add_argument(
        "--neural-mp-root",
        type=str,
        default="/home/hanyu/neuralmotionplanner",
        help="Path to the neuralmotionplanner repo that provides the real-world env and eval loop.",
    )
    parser.add_argument(
        "--cfg-set",
        type=str,
        default="scene1_l1",
        help="Name of the collected config set under real_world_test_set/collected_configs.",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="scene1_single_blcok",
        help="Base name of cached pcd/rgb files under real_world_test_set/collected_pcds.",
    )
    parser.add_argument(
        "--cam-only",
        action="store_true",
        help="Launch only the camera stack without connecting the robot arm.",
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help="Launch the arm without the gripper, matching neural_mp's arm-only eval mode.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached point cloud files instead of triggering live multi-camera capture.",
    )
    parser.add_argument(
        "--debug-raw-pcd",
        action="store_true",
        help="Show/debug each camera's raw point cloud inside the neural_mp environment pipeline.",
    )
    parser.add_argument(
        "--debug-combined-pcd",
        action="store_true",
        help="Show/debug the merged point cloud after camera fusion and robot removal.",
    )
    parser.add_argument(
        "--denoise-pcd",
        action="store_true",
        help="Enable the neural_mp environment's point cloud denoising during capture.",
    )
    parser.add_argument(
        "--save-pcd",
        action="store_true",
        help="Save the captured merged point cloud under real_world_test_set/collected_pcds.",
    )
    parser.add_argument(
        "-l",
        "--log-name",
        type=str,
        default="rw_eval_drp",
        help="CSV log name written under real_world_test_set/evals.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Inference device for DRP.",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile for lower startup and memory overhead.",
    )
    parser.add_argument(
        "--max-rollout-len",
        type=int,
        default=120,
        help="Maximum number of DRP policy steps per start/goal planning query.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.08,
        help="Clamp each DRP joint update to this maximum absolute step size in radians.",
    )
    parser.add_argument(
        "--goal-pos-tol-cm",
        type=float,
        default=1.0,
        help="Planning success threshold on end-effector position error in centimeters.",
    )
    parser.add_argument(
        "--goal-ori-tol-deg",
        type=float,
        default=15.0,
        help="Planning success threshold on end-effector orientation error in degrees.",
    )
    parser.add_argument(
        "--tto",
        action="store_true",
        help="Kept for compatibility with rw_eval; DRP falls back to the normal rollout.",
    )
    return parser


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    neural_mp_root = _resolve_neural_mp_root(args.neural_mp_root)
    FrankaRealEnvManimo, rw_eval = _import_neural_mp_modules(neural_mp_root)

    config_path = neural_mp_root / "real_world_test_set" / "collected_configs" / f"{args.cfg_set}.npy"
    cache_dir = neural_mp_root / "real_world_test_set" / "collected_pcds"
    eval_dir = neural_mp_root / "real_world_test_set" / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config set not found: {config_path}")

    device = _resolve_device(args.device)
    env = FrankaRealEnvManimo(cam_only=args.cam_only, arm_only=args.arm_only)
    eval_agent = DRPRealEvalAgent(
        env=env,
        cache_dir=cache_dir,
        device=device,
        disable_torch_compile=args.disable_torch_compile,
        max_rollout_len=args.max_rollout_len,
        max_joint_step=args.max_joint_step,
        goal_pos_tolerance_m=args.goal_pos_tol_cm / 100.0,
        goal_ori_tolerance_deg=args.goal_ori_tol_deg,
    )

    print("=" * 80)
    print("DRP real-world evaluation")
    print(f"neural_mp root        : {neural_mp_root}")
    print(f"Config set            : {config_path}")
    print(f"Use cached pcd        : {args.use_cache}")
    print(f"Device                : {device}")
    print(f"Max rollout len       : {args.max_rollout_len}")
    print(f"Max joint step        : {args.max_joint_step:.4f} rad")
    print("=" * 80)

    points, colors = eval_agent.get_scene_pcd(
        use_cache=args.use_cache,
        cache_name=args.cache_name,
        debug_raw_pcd=args.debug_raw_pcd,
        debug_combined_pcd=args.debug_combined_pcd,
        save_pcd=args.save_pcd,
        save_file_name=args.log_name,
        denoise=args.denoise_pcd,
    )
    config_set = np.load(config_path)

    rw_eval(
        eval_agent=eval_agent,
        env=env,
        points=points,
        colors=np.asarray(colors, dtype=np.float32)[:, [2, 1, 0]],
        config_set=config_set,
        arm_only=args.arm_only,
        args=args,
        file_path=str(eval_dir / f"{args.log_name}.csv"),
    )


if __name__ == "__main__":
    main()
