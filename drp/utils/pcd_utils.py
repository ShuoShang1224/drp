from pathlib import Path

import numpy as np
import torch


_FRANKA_VISUAL_LINKS = (
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)


def _matrix_from_xyz_rpy(xyz, rpy):
    x, y, z = xyz
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )
    mat[:3, 3] = (x, y, z)
    return mat


_FRANKA_JOINT_ORIGINS = (
    _matrix_from_xyz_rpy((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (-1.57079632679, 0.0, 0.0)),
    _matrix_from_xyz_rpy((0.0, -0.316, 0.0), (1.57079632679, 0.0, 0.0)),
    _matrix_from_xyz_rpy((0.0825, 0.0, 0.0), (1.57079632679, 0.0, 0.0)),
    _matrix_from_xyz_rpy((-0.0825, 0.384, 0.0), (-1.57079632679, 0.0, 0.0)),
    _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (1.57079632679, 0.0, 0.0)),
    _matrix_from_xyz_rpy((0.088, 0.0, 0.0), (1.57079632679, 0.0, 0.0)),
)
_FRANKA_LINK8_ORIGIN = _matrix_from_xyz_rpy((0.0, 0.0, 0.107), (0.0, 0.0, 0.0))
_FRANKA_HAND_ORIGIN = _matrix_from_xyz_rpy(
    (0.0, 0.0, 0.0), (0.0, 0.0, -0.785398163397)
)
_FRANKA_RIGHT_GRIPPER_ORIGIN = _matrix_from_xyz_rpy(
    (0.0, 0.0, 0.1), (0.0, 0.0, 2.35619449019)
)
_FRANKA_RIGHT_FINGER_VISUAL_ORIGIN = _matrix_from_xyz_rpy(
    (0.0, 0.0, 0.0), (0.0, 0.0, np.pi)
)


def _batched_z_rotation(angles):
    batch = angles.shape[0]
    rot = torch.eye(4, device=angles.device, dtype=angles.dtype).repeat(batch, 1, 1)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = -sin
    rot[:, 1, 0] = sin
    rot[:, 1, 1] = cos
    return rot


def _transform_points(points, transforms):
    rotation = transforms[:, :3, :3].transpose(1, 2)
    translation = transforms[:, :3, 3].unsqueeze(1)
    return points.expand(transforms.shape[0], -1, -1).matmul(rotation) + translation


def transform_pointcloud(pc, transformation_matrix, in_place=True):
    xyz = pc[..., :3]
    if pc.ndim == 2:
        transformed = _transform_points(
            xyz.unsqueeze(0), transformation_matrix.unsqueeze(0)
        )[0]
    elif pc.ndim == 3:
        transformed = _transform_points(xyz, transformation_matrix)
    else:
        raise ValueError("Pointcloud must have shape Nx3 or BxNx3")

    if in_place:
        pc[..., :3] = transformed
        return pc
    return torch.cat((transformed, pc[..., 3:]), dim=-1)


class FrankaSampler:
    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        default_prismatic_value=0.025,
        with_base_link=True,
    ):
        self.device = torch.device(device)
        self.num_fixed_points = num_fixed_points
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
        self._load_points()
        self._init_fk_constants()

    def _load_points(self):
        cache_path = Path(__file__).resolve().parent / "pcd_cache" / "franka"
        if self.num_fixed_points is None:
            cache_path = cache_path / "full_point_cloud.npy"
            self._fixed_cache_loaded = False
        else:
            fixed_path = cache_path / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            self._fixed_cache_loaded = fixed_path.is_file()
            cache_path = (
                fixed_path
                if self._fixed_cache_loaded
                else cache_path / "full_point_cloud.npy"
            )
        points = np.load(cache_path, allow_pickle=True).item()
        self.points = {
            name: torch.as_tensor(
                points[name], device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            for name in _FRANKA_VISUAL_LINKS
        }
        self.links = [
            name
            for name in _FRANKA_VISUAL_LINKS
            if self.with_base_link or name != "panda_link0"
        ]

    def _init_fk_constants(self):
        self._joint_origins = tuple(
            torch.as_tensor(mat, device=self.device, dtype=torch.float32)
            for mat in _FRANKA_JOINT_ORIGINS
        )
        self._link8_origin = torch.as_tensor(
            _FRANKA_LINK8_ORIGIN, device=self.device, dtype=torch.float32
        )
        self._hand_origin = torch.as_tensor(
            _FRANKA_HAND_ORIGIN, device=self.device, dtype=torch.float32
        )
        self._right_gripper_origin = torch.as_tensor(
            _FRANKA_RIGHT_GRIPPER_ORIGIN, device=self.device, dtype=torch.float32
        )
        self._right_finger_visual_origin = torch.as_tensor(
            _FRANKA_RIGHT_FINGER_VISUAL_ORIGIN, device=self.device, dtype=torch.float32
        )

    def _origin_batch(self, origin, batch):
        return origin.unsqueeze(0).expand(batch, -1, -1)

    def _finger_transform(self, gripper_width, direction):
        batch = gripper_width.shape[0]
        transform = torch.eye(
            4, device=gripper_width.device, dtype=gripper_width.dtype
        ).repeat(batch, 1, 1)
        transform[:, 1, 3] = direction * gripper_width
        transform[:, 2, 3] = 0.0584
        return transform

    def _link_fk(self, config):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        config = config.to(device=self.device, dtype=torch.float32)
        batch = config.shape[0]

        link_transforms = {
            "panda_link0": torch.eye(
                4, device=self.device, dtype=torch.float32
            ).repeat(batch, 1, 1)
        }
        current = link_transforms["panda_link0"]
        for idx, origin in enumerate(self._joint_origins):
            joint_pose = self._origin_batch(origin, batch).matmul(
                _batched_z_rotation(config[:, idx])
            )
            current = current.matmul(joint_pose)
            link_transforms[f"panda_link{idx + 1}"] = current

        link8 = current.matmul(self._origin_batch(self._link8_origin, batch))
        hand = link8.matmul(self._origin_batch(self._hand_origin, batch))

        if config.shape[-1] >= 8:
            gripper_width = config[:, 7]
        else:
            gripper_width = torch.full(
                (batch,),
                self.default_prismatic_value,
                device=self.device,
                dtype=torch.float32,
            )

        link_transforms["panda_hand"] = hand
        link_transforms["panda_leftfinger"] = hand.matmul(
            self._finger_transform(gripper_width, 1.0)
        )
        link_transforms["panda_rightfinger"] = hand.matmul(
            self._finger_transform(gripper_width, -1.0)
        )
        link_transforms["right_gripper"] = link8.matmul(
            self._origin_batch(self._right_gripper_origin, batch)
        )
        return link_transforms

    def _visual_fk(self, config):
        link_transforms = self._link_fk(config)
        batch = link_transforms["panda_rightfinger"].shape[0]
        link_transforms["panda_rightfinger"] = link_transforms[
            "panda_rightfinger"
        ].matmul(
            self._origin_batch(self._right_finger_visual_origin, batch)
        )
        return link_transforms

    def end_effector_pose(self, config, frame="right_gripper"):
        if frame != "right_gripper":
            raise ValueError("Only the right_gripper frame is supported")
        return self._link_fk(config)[frame]

    def sample(self, config, num_points=None):
        if (self.num_fixed_points is None) == (num_points is None):
            raise ValueError("Specify exactly one of num_fixed_points or num_points")

        sample_count = self.num_fixed_points if num_points is None else num_points
        fk = self._visual_fk(config)
        pc = torch.cat(
            [_transform_points(self.points[link], fk[link]) for link in self.links],
            dim=1,
        )
        if (
            num_points is None
            and self.num_fixed_points is not None
            and self._fixed_cache_loaded
        ):
            return pc
        if sample_count > pc.shape[1]:
            raise ValueError(
                f"Cannot sample {sample_count} points from {pc.shape[1]} points"
            )
        indices = torch.randperm(pc.shape[1], device=pc.device)[:sample_count]
        return pc[:, indices, :]
