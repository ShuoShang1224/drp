import torch
from drp.model.impact import IMPACT
from drp.utils.pcd_utils import FrankaSampler


class DRPInference():
    def __init__(self, device):
        self.device = device
        self._num_robot_points = 256
        self._num_obstacle_points = 2048
        self._fk_sampler = FrankaSampler(self.device, use_cache=True, default_prismatic_value=0.04)
        self.set_up_policy()

    @property
    def num_robot_points(self):
        return self._num_robot_points

    @property
    def num_obstacle_points(self):
        return self._num_obstacle_points

    def set_up_policy(self):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.suppress_errors = True

        model = IMPACT.from_pretrained('jimyoung6709/DRP')

        self.model = torch.compile(model).to(self.device).float()

    @torch.inference_mode()
    def get_actions(self, env_obs_dict):
        joint_pos = env_obs_dict["joint_pos"]
        goal_joint_pos = env_obs_dict["goal_joint_pos"]

        current_robot_pcd = self._fk_sampler.sample(joint_pos, self.num_robot_points)

        obstacle_pcd_list = env_obs_dict["combined_obstacle_pcd"]
        subsampled_pcd_list = []
        for pcd in obstacle_pcd_list:
            pcd = torch.as_tensor(pcd, device=self.device, dtype=torch.float32)
            num_points = pcd.shape[0]
            if num_points == 0:
                sampled = torch.zeros(self.num_obstacle_points, 3, device=pcd.device)
            elif num_points >= self.num_obstacle_points:
                indices = torch.randperm(num_points, device=pcd.device)[0:self.num_obstacle_points]
                sampled = pcd[indices]
            else:
                indices = torch.randint(0, num_points, (self.num_obstacle_points,))
                sampled = pcd[indices]
            subsampled_pcd_list.append(sampled)

        # Stack into final tensor of shape (num_envs, num_obstacle_points, 3)
        subsampled_pcd = torch.stack(subsampled_pcd_list, dim=0)

        obs_dict = self.prepare_act_observation(
            joint_pos=joint_pos, 
            goal_joint_pos=goal_joint_pos, 
            obstacle_pcd=subsampled_pcd, 
            current_robot_pcd=current_robot_pcd, 
        )
        # max is 10
        n_action = 0 #5
        # (num_envs, S, 7)
        action_chunk = self.model.get_action(obs_dict)
        # absolute joint position target (num_envs, 7)
        absolute_joint_pos_action = action_chunk[:, min(n_action, action_chunk.shape[1])]
        joint_pos_target = absolute_joint_pos_action
        # delta_action = joint_pos_target - joint_pos
        # joint_pos_target = joint_pos + delta_action*4
        return joint_pos_target

    def prepare_act_observation(self, joint_pos, goal_joint_pos, obstacle_pcd, current_robot_pcd):
        obs_dict = dict()
        obs_dict["current_angles"] = joint_pos.float()
        obs_dict["goal_angles"] = goal_joint_pos.float()
        obs_dict["scene_pcd"] = obstacle_pcd.float()
        obs_dict["robot_pcd"] = current_robot_pcd.float()
        return obs_dict

    def reset(self):
        pass



if __name__ == "__main__":
    import time
    from pathlib import Path

    import numpy as np

    from drp.utils.franka_utils import (
        FRANKA_LOWER_LIMITS,
        FRANKA_UPPER_LIMITS,
        clamp_to_franka_limits,
    )
    from drp.utils.viser_visualizer import ViserVisualizer

    def _box_surface_points(center, half_extents, n, device):
        center = torch.tensor(center, device=device, dtype=torch.float32)
        half_extents = torch.tensor(half_extents, device=device, dtype=torch.float32)
        pts = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * half_extents
        faces = torch.randint(0, 6, (n,), device=device)
        axes = faces // 2
        signs = faces.remainder(2).float() * 2.0 - 1.0
        pts[torch.arange(n, device=device), axes] = signs * half_extents[axes]
        return pts + center

    def _solve_position_ik(fk_sampler, target_pos, seed_q, max_iters=16):
        lower = torch.tensor(FRANKA_LOWER_LIMITS, device=seed_q.device, dtype=torch.float32)
        upper = torch.tensor(FRANKA_UPPER_LIMITS, device=seed_q.device, dtype=torch.float32)
        target_pos = torch.as_tensor(target_pos, device=seed_q.device, dtype=torch.float32)
        q = seed_q.detach().clone().float()

        def fk_pos(q_in):
            return fk_sampler.end_effector_pose(q_in.unsqueeze(0))[0, :3, 3]

        for _ in range(max_iters):
            q = q.detach().requires_grad_(True)
            pos = fk_pos(q)
            err = target_pos - pos
            if torch.linalg.norm(err) < 0.005:
                break
            jac = torch.autograd.functional.jacobian(fk_pos, q)
            lhs = jac @ jac.T + 1e-4 * torch.eye(3, device=q.device)
            dq = jac.T @ torch.linalg.solve(lhs, err)
            q = torch.clamp(q + torch.clamp(dq, -0.12, 0.12), lower, upper)
        return q.detach()

    urdf_path = (
        Path(__file__).resolve().parent
        / "assets/urdf/franka_description/robots/franka_panda_gripper.urdf"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DRPInference(device)
    visualizer = ViserVisualizer(urdf_path)

    config_1 = np.array([-0.7, 0.5, 0.0, -2.0, 0.0, 2.5, 0.0], dtype=np.float32)
    config_2 = np.array([0.7, 0.8, 0., -1.7, 0.0, 3.0, 0.0], dtype=np.float32)
    joint_pos = torch.tensor(config_1, device=device).unsqueeze(0)
    goal_joint_pos = torch.tensor(config_2, device=device).unsqueeze(0)
    target_pos = policy._fk_sampler.end_effector_pose(goal_joint_pos)[0, :3, 3].detach().cpu().numpy()
    eef_target = visualizer.server.scene.add_transform_controls(
        "/eef_target",
        scale=0.18,
        position=target_pos,
        disable_rotations=True,
        translation_limits=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
    )
    last_target_pos = target_pos.copy()

    def _set_target_config(config_q):
        global goal_joint_pos, last_target_pos
        goal_joint_pos = torch.as_tensor(config_q, device=device, dtype=torch.float32).unsqueeze(0)
        pos = policy._fk_sampler.end_effector_pose(goal_joint_pos)[0, :3, 3].detach().cpu().numpy()
        last_target_pos = pos.copy()
        eef_target.position = tuple(float(v) for v in pos)

    visualizer.server.gui.add_button("config 1").on_click(lambda _: _set_target_config(config_1))
    visualizer.server.gui.add_button("config 2").on_click(lambda _: _set_target_config(config_2))

    table = torch.empty(1024, 3, device=device)
    table[:, 0] = torch.rand(1024, device=device) * 1.0 + 0.2
    table[:, 1] = torch.rand(1024, device=device) * 1.2 - 0.6
    table[:, 2] = 0.02
    obstacle_pcd = torch.cat(
        (
            table,
            _box_surface_points((0.5, 0.0, 0.2), (0.2, 0.1, 0.2), 1024, device),
        ),
        dim=0,
    )
    visualizer.update_point_cloud(
        "scene_pcd",
        obstacle_pcd.cpu().numpy(),
    )

    print("Viser DRPACT demo is running. Drag /eef_target to change the target.")
    while True:
        target_pos = np.asarray(eef_target.position, dtype=np.float32)
        if not np.allclose(target_pos, last_target_pos, atol=1e-4):
            goal_joint_pos = _solve_position_ik(
                policy._fk_sampler, target_pos, goal_joint_pos[0]
            ).unsqueeze(0)
            last_target_pos = target_pos.copy()
        env_obs = {
            "joint_pos": joint_pos,
            "goal_joint_pos": goal_joint_pos,
            "combined_obstacle_pcd": [obstacle_pcd],
        }

        next_q = policy.get_actions(env_obs)
        max_step = torch.full_like(joint_pos, 0.08)
        joint_pos = joint_pos + torch.clamp(next_q - joint_pos, -max_step, max_step)
        joint_pos, _ = clamp_to_franka_limits(joint_pos)

        q_np = joint_pos[0].detach().cpu().numpy()
        robot_pcd = policy._fk_sampler.sample(joint_pos, 512)[0].detach().cpu().numpy()
        visualizer.set_joint_positions(q_np)
        visualizer.update_point_cloud("robot_pcd", robot_pcd)
        time.sleep(1.0 / 15.0)
