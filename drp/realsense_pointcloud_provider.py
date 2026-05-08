import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


DEFAULT_D435_DEPTH_WIDTH = 848
DEFAULT_D435_DEPTH_HEIGHT = 480
DEFAULT_D435_FPS = 30


def _default_camera_extrinsics() -> List[str]:
    return [
        "cam/T_cam2base.npy",
        "cam/T_cam2base_cam2.npy",
        "cam/T_cam2base_cam3.npy",
        "cam/T_cam2base_cam4.npy",
    ]


def _load_transform(path: str) -> np.ndarray:
    matrix = np.load(Path(path))
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform in {path}, got {matrix.shape}")
    return matrix.astype(np.float32, copy=False)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.reshape(0, 3).astype(np.float32)
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32, copy=False)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _crop_workspace(points: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    return points[mask]


def _subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices].astype(np.float32, copy=False)


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        min_depth_m: float,
        max_depth_m: float,
        depth_stride: int,
        warmup_frames: int,
        cam2base: np.ndarray,
    ):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required. Please install librealsense and pyrealsense2."
            )

        self.serial = serial
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_stride = max(1, int(depth_stride))
        self.cam2base = cam2base

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
            raise RuntimeError(f"Failed to acquire aligned depth frame for camera {self.serial}")

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
        points_cam = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)
        return _transform_points(points_cam, self.cam2base)

    def close(self) -> None:
        self.pipeline.stop()


def _write_pointcloud(
    output_path: Path,
    points: np.ndarray,
    per_camera_counts: List[int],
    frame_idx: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=np.asarray(points, dtype=np.float32),
        per_camera_counts=np.asarray(per_camera_counts, dtype=np.int32),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        timestamp_s=np.asarray(time.time(), dtype=np.float64),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and merge pointclouds from four RealSense RGB-D cameras."
    )
    parser.add_argument("--serials", nargs=4, required=True, help="Four RealSense serial numbers in camera/extrinsics order.")
    parser.add_argument("--camera-extrinsics", nargs=4, default=_default_camera_extrinsics())
    parser.add_argument("--output", type=str, default="outputs/realsense_merged_pointcloud.npz")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--width", type=int, default=DEFAULT_D435_DEPTH_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_D435_DEPTH_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_D435_FPS)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--workspace-min", type=str, default="0.15,-0.80,-0.05")
    parser.add_argument("--workspace-max", type=str, default="1.10,0.80,1.20")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _parse_vec3(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.loop_hz <= 0:
        raise ValueError("--loop-hz must be > 0")
    if args.max_iterations < 0:
        raise ValueError("--max-iterations must be >= 0")
    rng = np.random.default_rng(args.seed)
    workspace_min = _parse_vec3(args.workspace_min)
    workspace_max = _parse_vec3(args.workspace_max)

    cameras = [
        RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            min_depth_m=args.min_depth,
            max_depth_m=args.max_depth,
            depth_stride=args.depth_stride,
            warmup_frames=args.warmup_frames,
            cam2base=_load_transform(extr_path),
        )
        for serial, extr_path in zip(args.serials, args.camera_extrinsics)
    ]

    output_path = Path(args.output)
    loop_period = 1.0 / float(args.loop_hz)

    print("=" * 80)
    print("RealSense pointcloud provider")
    print(f"Output                : {output_path}")
    print(f"Loop Hz               : {args.loop_hz}")
    print(f"Workspace min/max     : {workspace_min.tolist()} / {workspace_max.tolist()}")
    print("=" * 80)

    try:
        frame_idx = 0
        while args.max_iterations == 0 or frame_idx < args.max_iterations:
            tic = time.perf_counter()
            merged_parts: List[np.ndarray] = []
            per_camera_counts: List[int] = []

            for camera in cameras:
                points = camera.capture()
                points = _crop_workspace(points, workspace_min, workspace_max)
                per_camera_counts.append(int(points.shape[0]))
                if points.shape[0] > 0:
                    merged_parts.append(points)

            if merged_parts:
                merged = np.concatenate(merged_parts, axis=0)
                merged = _voxel_downsample(merged, args.voxel_size)
                merged = _subsample_points(merged, args.max_points, rng)
            else:
                merged = np.zeros((0, 3), dtype=np.float32)

            _write_pointcloud(output_path, merged, per_camera_counts, frame_idx)

            if args.print_every > 0 and frame_idx % args.print_every == 0:
                print(
                    f"[frame {frame_idx:>5d}] "
                    f"per_camera={per_camera_counts} merged={merged.shape[0]}"
                )

            frame_idx += 1
            sleep_time = loop_period - (time.perf_counter() - tic)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        for camera in cameras:
            camera.close()


if __name__ == "__main__":
    main()
