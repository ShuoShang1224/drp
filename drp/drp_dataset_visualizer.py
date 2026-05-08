import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from drp.utils.franka_utils import clamp_to_franka_limits
from drp.utils.pcd_utils import FrankaSampler

try:
    import h5py
except ImportError:
    h5py = None

try:
    import viser
    from viser.extras import ViserUrdf
except ImportError:
    viser = None
    ViserUrdf = None

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None
    UsdGeom = None


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


def _box_surface_points(
    center: np.ndarray,
    half_extents: np.ndarray,
    n: int,
    device: torch.device,
) -> torch.Tensor:
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
) -> torch.Tensor:
    table = torch.empty(table_points, 3, device=device)
    table[:, 0] = torch.rand(table_points, device=device) * 1.0 + 0.2
    table[:, 1] = torch.rand(table_points, device=device) * 1.2 - 0.6
    table[:, 2] = 0.02

    box = _box_surface_points(box_center, box_half_extents, box_points, device)
    return torch.cat((table, box), dim=0)


def _sample_cube_surface_points_np(
    size: float,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
        local_pts = _sample_cube_surface_points_np(float(size), points_per_cube, rng)
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


def _infer_scene_prefix_from_dataset(dataset_hdf5: Path) -> str:
    stem = dataset_hdf5.stem
    return stem[:-8] if stem.endswith("_targets") else stem


def _load_dataset_sample(
    dataset_hdf5: Path,
    saved_scenes_dir: Path,
    scene_prefix: str,
    demo_index: int,
    pair_index: int,
    device: torch.device,
    obstacle_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, Path]:
    if h5py is None:
        raise RuntimeError("h5py is required for dataset visualization mode")
    if not dataset_hdf5.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_hdf5}")
    if not saved_scenes_dir.is_dir():
        raise FileNotFoundError(f"Saved scenes directory not found: {saved_scenes_dir}")

    demo_key = f"demo_{demo_index}"
    with h5py.File(dataset_hdf5, "r") as f:
        if "data" not in f or demo_key not in f["data"]:
            raise KeyError(f"Could not find data/{demo_key} in {dataset_hdf5}")
        group = f["data"][demo_key]
        starts = np.asarray(group["start_positions"], dtype=np.float32)
        goals = np.asarray(group["goal_positions"], dtype=np.float32)
        pair_count = min(starts.shape[0], goals.shape[0])
        if pair_count == 0:
            raise RuntimeError(f"No valid start/goal pairs found in {demo_key}")
        if pair_index < 0 or pair_index >= pair_count:
            raise IndexError(f"pair_index {pair_index} out of range [0, {pair_count - 1}]")

        start_q, _ = clamp_to_franka_limits(starts[pair_index, :7])
        goal_q, _ = clamp_to_franka_limits(goals[pair_index, :7])

    scene_path = saved_scenes_dir / f"{scene_prefix}_demo_{demo_index}.usd"
    obstacle_pcd = _load_scene_obstacle_pcd_from_usd(
        scene_path=scene_path,
        device=device,
        total_points=obstacle_points,
        rng=rng,
    )
    return (
        np.asarray(start_q, dtype=np.float32),
        np.asarray(goal_q, dtype=np.float32),
        obstacle_pcd,
        scene_path,
    )


def _make_vis_cfg(joint_names, q: np.ndarray, gripper_width: float = 0.04) -> np.ndarray:
    cfg = np.zeros(len(joint_names), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    for i, value in enumerate(q.reshape(7), start=1):
        idx = name_to_idx.get(f"panda_joint{i}")
        if idx is not None:
            cfg[idx] = float(value)
    if "panda_finger_joint1" in name_to_idx:
        cfg[name_to_idx["panda_finger_joint1"]] = gripper_width
    return cfg


def _pointcloud_clearance_m(robot_pcd: torch.Tensor, obstacle_pcd: torch.Tensor) -> float:
    dists = torch.cdist(robot_pcd.unsqueeze(0), obstacle_pcd.unsqueeze(0))
    return float(dists.amin().item())


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
        description="Visualize DRP obstacle point cloud, start pose, and goal pose."
    )
    parser.add_argument(
        "--dataset-hdf5",
        type=str,
        default="",
        help="Optional dataset HDF5 path. If set, visualize one dataset/USD sample.",
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
        "--demo-index",
        type=int,
        default=0,
        help="Dataset demo index to visualize when --dataset-hdf5 is set.",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Start/goal pair index within the chosen demo.",
    )
    parser.add_argument(
        "--usd-obstacle-points",
        type=int,
        default=6000,
        help="Number of obstacle points sampled from the USD scene.",
    )
    parser.add_argument(
        "--config1",
        type=str,
        default="-0.7,0.5,0.0,-2.0,0.0,2.5,0.0",
        help="Synthetic start config as 7 comma-separated radians.",
    )
    parser.add_argument(
        "--config2",
        type=str,
        default="0.7,0.8,0.0,-1.7,0.0,3.0,0.0",
        help="Synthetic goal config as 7 comma-separated radians.",
    )
    parser.add_argument(
        "--table-points",
        type=int,
        default=1024,
        help="Number of sampled points on the table plane in synthetic mode.",
    )
    parser.add_argument(
        "--box-points",
        type=int,
        default=1024,
        help="Number of sampled points on the obstacle box surface in synthetic mode.",
    )
    parser.add_argument(
        "--box-center",
        type=str,
        default="0.5,0.0,0.2",
        help="Synthetic obstacle box center as x,y,z.",
    )
    parser.add_argument(
        "--box-half-extents",
        type=str,
        default="0.2,0.1,0.2",
        help="Synthetic obstacle box half extents as hx,hy,hz.",
    )
    parser.add_argument(
        "--robot-points",
        type=int,
        default=2048,
        help="Number of sampled robot points for start/goal visualization.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Sampling device. CPU is usually enough for visualization.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if viser is None or ViserUrdf is None:
        raise RuntimeError("viser is required for visualization")
    if args.usd_obstacle_points <= 0:
        raise ValueError("--usd-obstacle-points must be > 0")
    if args.table_points <= 0 or args.box_points <= 0 or args.robot_points <= 0:
        raise ValueError("--table-points, --box-points, and --robot-points must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = _resolve_device(args.device)
    fk_sampler = FrankaSampler(device, use_cache=True, default_prismatic_value=0.04)

    use_dataset_mode = bool(args.dataset_hdf5)
    if use_dataset_mode:
        dataset_hdf5 = Path(args.dataset_hdf5)
        saved_scenes_dir = Path(args.saved_scenes_dir)
        scene_prefix = args.scene_prefix.strip() or _infer_scene_prefix_from_dataset(dataset_hdf5)
        start_q, goal_q, obstacle_pcd, scene_path = _load_dataset_sample(
            dataset_hdf5=dataset_hdf5,
            saved_scenes_dir=saved_scenes_dir,
            scene_prefix=scene_prefix,
            demo_index=args.demo_index,
            pair_index=args.pair_index,
            device=device,
            obstacle_points=args.usd_obstacle_points,
            rng=rng,
        )
        source_desc = (
            f"dataset={dataset_hdf5.name}, demo={args.demo_index}, "
            f"pair={args.pair_index}, scene={scene_path.name}"
        )
    else:
        start_q = _parse_float_list(args.config1, expected_len=7, name="--config1")
        goal_q = _parse_float_list(args.config2, expected_len=7, name="--config2")
        box_center = _parse_float_list(args.box_center, expected_len=3, name="--box-center")
        box_half_extents = _parse_float_list(
            args.box_half_extents, expected_len=3, name="--box-half-extents"
        )
        obstacle_pcd = _build_scene_obstacle_pcd(
            device=device,
            table_points=args.table_points,
            box_points=args.box_points,
            box_center=box_center,
            box_half_extents=box_half_extents,
        )
        source_desc = "synthetic scene"

    start_q_t = torch.as_tensor(start_q, device=device, dtype=torch.float32).unsqueeze(0)
    goal_q_t = torch.as_tensor(goal_q, device=device, dtype=torch.float32).unsqueeze(0)
    start_robot_pcd = fk_sampler.sample(start_q_t, args.robot_points)[0]
    goal_robot_pcd = fk_sampler.sample(goal_q_t, args.robot_points)[0]
    start_eef = fk_sampler.end_effector_pose(start_q_t)[0, :3, 3]
    goal_eef = fk_sampler.end_effector_pose(goal_q_t)[0, :3, 3]

    obstacle_np = obstacle_pcd.detach().cpu().numpy()
    start_robot_np = start_robot_pcd.detach().cpu().numpy()
    goal_robot_np = goal_robot_pcd.detach().cpu().numpy()
    start_eef_np = start_eef.detach().cpu().numpy()
    goal_eef_np = goal_eef.detach().cpu().numpy()

    obstacle_min = obstacle_np.min(axis=0)
    obstacle_max = obstacle_np.max(axis=0)
    start_clearance = _pointcloud_clearance_m(start_robot_pcd, obstacle_pcd)
    goal_clearance = _pointcloud_clearance_m(goal_robot_pcd, obstacle_pcd)

    urdf_path = (
        Path(__file__).resolve().parent
        / "assets/urdf/franka_description/robots/franka_panda_gripper.urdf"
    )

    server = viser.ViserServer()
    server.gui.configure_theme(control_width="medium")
    server.scene.add_frame(
        "/WorldAxes",
        show_axes=True,
        axes_length=0.15,
        axes_radius=0.01,
        visible=True,
    )
    server.scene.add_grid(
        "/grid",
        width=10,
        height=10,
        position=(0.0, 0.0, 0.0),
        shadow_opacity=0.1,
    )

    start_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/start_robot",
        load_meshes=True,
        load_collision_meshes=False,
    )
    goal_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/goal_robot",
        load_meshes=True,
        load_collision_meshes=False,
    )
    start_urdf.update_cfg(_make_vis_cfg(start_urdf.get_actuated_joint_names(), start_q))
    goal_urdf.update_cfg(_make_vis_cfg(goal_urdf.get_actuated_joint_names(), goal_q))

    obstacle_handle = server.scene.add_point_cloud(
        name="/scene_pcd",
        points=obstacle_np.astype(np.float16, copy=False),
        colors=(60, 170, 80),
        point_size=0.006,
        precision="float16",
        visible=True,
    )
    start_handle = server.scene.add_point_cloud(
        name="/start_robot_pcd",
        points=start_robot_np.astype(np.float16, copy=False),
        colors=(0, 80, 255),
        point_size=0.006,
        precision="float16",
        visible=True,
    )
    goal_handle = server.scene.add_point_cloud(
        name="/goal_robot_pcd",
        points=goal_robot_np.astype(np.float16, copy=False),
        colors=(255, 120, 0),
        point_size=0.006,
        precision="float16",
        visible=True,
    )
    start_eef_handle = server.scene.add_point_cloud(
        name="/start_eef",
        points=start_eef_np.reshape(1, 3).astype(np.float16, copy=False),
        colors=(0, 80, 255),
        point_size=0.02,
        precision="float16",
        visible=True,
    )
    goal_eef_handle = server.scene.add_point_cloud(
        name="/goal_eef",
        points=goal_eef_np.reshape(1, 3).astype(np.float16, copy=False),
        colors=(255, 120, 0),
        point_size=0.02,
        precision="float16",
        visible=True,
    )

    server.gui.add_button("toggle obstacle").on_click(
        lambda _: setattr(obstacle_handle, "visible", not obstacle_handle.visible)
    )
    server.gui.add_button("toggle start robot").on_click(
        lambda _: (
            setattr(start_handle, "visible", not start_handle.visible),
            setattr(start_eef_handle, "visible", not start_eef_handle.visible),
        )
    )
    server.gui.add_button("toggle goal robot").on_click(
        lambda _: (
            setattr(goal_handle, "visible", not goal_handle.visible),
            setattr(goal_eef_handle, "visible", not goal_eef_handle.visible),
        )
    )

    print("=" * 80)
    print("DRP visualization inspector")
    print(f"Source                : {source_desc}")
    print(f"Device                : {device}")
    print(f"Obstacle points       : {obstacle_np.shape[0]}")
    print(f"Obstacle xyz min      : {np.round(obstacle_min, 4).tolist()}")
    print(f"Obstacle xyz max      : {np.round(obstacle_max, 4).tolist()}")
    print(f"Start q               : {np.round(start_q, 4).tolist()}")
    print(f"Goal q                : {np.round(goal_q, 4).tolist()}")
    print(f"Start eef xyz         : {np.round(start_eef_np, 4).tolist()}")
    print(f"Goal eef xyz          : {np.round(goal_eef_np, 4).tolist()}")
    print(f"Start clearance       : {start_clearance:.4f} m")
    print(f"Goal clearance        : {goal_clearance:.4f} m")
    print("Blue                  : start robot / start eef")
    print("Orange                : goal robot / goal eef")
    print("Green                 : obstacle point cloud")
    print("Open the Viser URL shown below to inspect the scene.")
    print("=" * 80)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
