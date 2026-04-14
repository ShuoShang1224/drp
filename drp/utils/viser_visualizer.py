import numpy as np
from pathlib import Path

import viser
from viser.extras import ViserUrdf


class ViserVisualizer:
    def __init__(self, urdf_path):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(control_width="medium")  # small | medium | large

        self.server.scene.add_frame(
            "/WorldAxes", show_axes=True, axes_length=0.15, axes_radius=0.01, visible=True
        )
        self.server.scene.add_frame("/robot", show_axes=False)
        self.server.scene.add_grid(
            "/grid", width=10, height=10, position=(0.0, 0.0, 0.0), shadow_opacity=0.1
        )
        self._urdf_vis = ViserUrdf(
            self.server,
            urdf_or_path=Path(urdf_path),
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )

        self.joint_names = self._urdf_vis.get_actuated_joint_names()
        self._name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self._q = np.zeros(len(self.joint_names), dtype=float)
        default_q = [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.0]
        for i, value in enumerate(default_q, start=1):
            idx = self._name_to_idx.get(f"panda_joint{i}")
            if idx is not None:
                self._q[idx] = value
        if "panda_finger_joint1" in self._name_to_idx:
            self._q[self._name_to_idx["panda_finger_joint1"]] = 0.04
        self._urdf_vis.update_cfg(self._q)

        self._point_cloud_handle = {
            "scene_pcd": self.server.scene.add_point_cloud(
                name="/scene_pcd",
                points=np.zeros((0, 3), dtype=np.float16),
                colors=(60, 170, 80),
                point_size=0.006,
                precision="float16",
                visible=True,
            ),
            "robot_pcd": self.server.scene.add_point_cloud(
                name="/robot_pcd",
                points=np.zeros((0, 3), dtype=np.float16),
                colors=(0, 80, 255),
                point_size=0.005,
                precision="float16",
                visible=True,
            ),
        }

    def set_joint_positions(self, q, gripper_width=0.04):
        q = np.asarray(q, dtype=float).reshape(7)
        cfg = self._q.copy()
        for i, value in enumerate(q, start=1):
            idx = self._name_to_idx.get(f"panda_joint{i}")
            if idx is None:
                raise KeyError(f"Missing panda_joint{i}. Known joints: {list(self.joint_names)}")
            cfg[idx] = value
        if "panda_finger_joint1" in self._name_to_idx:
            cfg[self._name_to_idx["panda_finger_joint1"]] = gripper_width
        self._q[:] = cfg
        self._urdf_vis.update_cfg(self._q)

    def update_point_cloud(self, point_cloud_type, point_cloud):
        point_cloud_handle = self._point_cloud_handle[point_cloud_type]
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if getattr(point_cloud_handle, "precision", "float16") == "float16":
            point_cloud = point_cloud.astype(np.float16, copy=False)
        point_cloud_handle.points = point_cloud
