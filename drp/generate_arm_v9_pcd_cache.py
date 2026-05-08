from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise ImportError(
        "trimesh is required for generate_arm_v9_pcd_cache.py. "
        "Install it in the active environment before running this script."
    ) from exc


def _matrix_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
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


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def _parse_xyz_rpy(node: ET.Element) -> np.ndarray:
    xyz_str = node.attrib.get("xyz", "0 0 0")
    rpy_str = node.attrib.get("rpy", "0 0 0")
    xyz = np.fromstring(xyz_str, sep=" ", dtype=np.float32)
    rpy = np.fromstring(rpy_str, sep=" ", dtype=np.float32)
    return _matrix_from_xyz_rpy(xyz, rpy)


def _load_mesh_surface_points(mesh_path: Path, sample_count: int, seed: int) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def _resample_points(points: np.ndarray, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    replace = len(points) < sample_count
    indices = rng.choice(len(points), size=sample_count, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def _distribute_counts(lengths: Dict[str, int], total_count: int) -> Dict[str, int]:
    names = list(lengths.keys())
    weights = np.array([max(lengths[name], 0) for name in names], dtype=np.float64)
    if weights.sum() <= 0:
        return {name: 0 for name in names}

    raw = weights / weights.sum() * total_count
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total_count - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return {name: int(count) for name, count in zip(names, counts)}


@dataclass(frozen=True)
class AttachmentSpec:
    mesh_name: str
    semantic_link: str
    parent_joint: str
    attachment_joint: str
    description: str


ATTACHMENTS: Tuple[AttachmentSpec, ...] = (
    AttachmentSpec(
        mesh_name="Link5_lidar.STL",
        semantic_link="panda_link5",
        parent_joint="panda_link4",
        attachment_joint="Joint5_lidar",
        description="shared-parent transform from Link4",
    ),
    AttachmentSpec(
        mesh_name="Link6_lidar.STL",
        semantic_link="panda_link6",
        parent_joint="panda_link5",
        attachment_joint="Joint6_lidar",
        description="shared-parent transform from Link5",
    ),
    AttachmentSpec(
        mesh_name="Link7_lidar.STL",
        semantic_link="panda_link7",
        parent_joint="panda_link6",
        attachment_joint="Joint7_lidar",
        description="treated as Link7 attachment from shared parent Link6",
    ),
)

def _collect_joint_transforms(urdf_path: Path) -> Dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    transforms: Dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        origin = joint.find("origin")
        if origin is None:
            transforms[name] = np.eye(4, dtype=np.float32)
        else:
            transforms[name] = _parse_xyz_rpy(origin)
    return transforms


def _attachment_transform(
    joint_transforms: Dict[str, np.ndarray],
    parent_joint: str,
    attachment_joint: str,
) -> np.ndarray:
    parent_to_link = joint_transforms[parent_joint]
    parent_to_attachment = joint_transforms[attachment_joint]
    return np.linalg.inv(parent_to_link) @ parent_to_attachment


def _load_base_cache(cache_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    full_cache = np.load(cache_dir / "full_point_cloud.npy", allow_pickle=True).item()

    fixed_caches: Dict[int, Dict[str, np.ndarray]] = {}
    for path in sorted(cache_dir.glob("fixed_point_cloud_*.npy")):
        match = re.fullmatch(r"fixed_point_cloud_(\d+)\.npy", path.name)
        if match is None:
            continue
        fixed_caches[int(match.group(1))] = np.load(path, allow_pickle=True).item()
    return full_cache, fixed_caches


def _build_augmented_full_cache(
    full_cache: Dict[str, np.ndarray],
    meshes_dir: Path,
    joint_transforms: Dict[str, np.ndarray],
    attachment_full_points: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    merged = {name: np.asarray(points, dtype=np.float32).copy() for name, points in full_cache.items()}
    manifest: Dict[str, dict] = {}

    for index, spec in enumerate(ATTACHMENTS):
        mesh_path = meshes_dir / spec.mesh_name
        raw_points = _load_mesh_surface_points(
            mesh_path, sample_count=attachment_full_points, seed=seed + index
        )
        local_points = _transform_points(
            raw_points,
            _attachment_transform(
                joint_transforms,
                parent_joint=spec.parent_joint,
                attachment_joint=spec.attachment_joint,
            ).astype(np.float32),
        )
        merged[spec.semantic_link] = np.concatenate(
            [merged[spec.semantic_link], local_points], axis=0
        ).astype(np.float32, copy=False)

        attachment_transform = _attachment_transform(
            joint_transforms,
            parent_joint=spec.parent_joint,
            attachment_joint=spec.attachment_joint,
        ).astype(np.float32)
        manifest[spec.mesh_name] = {
            "semantic_link": spec.semantic_link,
            "parent_joint": spec.parent_joint,
            "attachment_joint": spec.attachment_joint,
            "description": spec.description,
            "source_mesh": str(mesh_path),
            "sampled_points": int(len(raw_points)),
            "attachment_in_arm_link_frame": attachment_transform.tolist(),
        }

    return merged, manifest


def _rebuild_fixed_cache(
    full_cache: Dict[str, np.ndarray],
    total_points: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = {name: len(points) for name, points in full_cache.items()}
    counts = _distribute_counts(lengths, total_points)
    rebuilt: Dict[str, np.ndarray] = {}
    for name, points in full_cache.items():
        rebuilt[name] = _resample_points(np.asarray(points, dtype=np.float32), counts[name], rng)
    return rebuilt


def _write_outputs(
    output_dir: Path,
    full_cache: Dict[str, np.ndarray],
    fixed_caches: Dict[int, Dict[str, np.ndarray]],
    manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "full_point_cloud.npy", full_cache, allow_pickle=True)
    for total_points, cache in fixed_caches.items():
        np.save(output_dir / f"fixed_point_cloud_{total_points}.npy", cache, allow_pickle=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Augment the standard DRP Franka point-cloud cache with arm_v9 attachment meshes. "
            "Only Link5/6/7 lidar meshes are added; the original Franka cache remains the base geometry."
        )
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo_root / "drp" / "arm_v9" / "urdf" / "arm_v9.urdf",
        help="arm_v9 URDF used to derive attachment-to-link transforms.",
    )
    parser.add_argument(
        "--meshes-dir",
        type=Path,
        default=repo_root / "drp" / "arm_v9" / "meshes",
        help="Directory containing Link5/6/7_lidar STL meshes.",
    )
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=repo_root / "drp" / "utils" / "pcd_cache" / "copy",
        help="Directory containing the original Franka full/fixed point-cloud caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "drp" / "utils" / "pcd_cache" / "arm_v9_generated",
        help="Output directory for the augmented cache files.",
    )
    parser.add_argument(
        "--attachment-full-points",
        type=int,
        default=4096,
        help="Number of surface points sampled per attachment for the full cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for mesh surface sampling and fixed-cache regeneration.",
    )
    args = parser.parse_args()

    joint_transforms = _collect_joint_transforms(args.urdf)
    full_cache, base_fixed_caches = _load_base_cache(args.base_cache_dir)

    augmented_full, attachment_manifest = _build_augmented_full_cache(
        full_cache=full_cache,
        meshes_dir=args.meshes_dir,
        joint_transforms=joint_transforms,
        attachment_full_points=args.attachment_full_points,
        seed=args.seed,
    )

    rebuilt_fixed_caches = {
        total_points: _rebuild_fixed_cache(
            full_cache=augmented_full,
            total_points=total_points,
            seed=args.seed + total_points,
        )
        for total_points in sorted(base_fixed_caches.keys())
    }

    manifest = {
        "urdf": str(args.urdf),
        "meshes_dir": str(args.meshes_dir),
        "base_cache_dir": str(args.base_cache_dir),
        "output_dir": str(args.output_dir),
        "attachment_full_points": args.attachment_full_points,
        "generated_fixed_caches": sorted(rebuilt_fixed_caches.keys()),
        "attachments": attachment_manifest,
        "semantic_link_counts": {
            name: {
                "base_points": int(len(full_cache[name])),
                "augmented_points": int(len(augmented_full[name])),
            }
            for name in sorted(augmented_full.keys())
        },
    }

    _write_outputs(
        output_dir=args.output_dir,
        full_cache=augmented_full,
        fixed_caches=rebuilt_fixed_caches,
        manifest=manifest,
    )

    print(f"Saved augmented cache to: {args.output_dir}")
    print("Attachment transforms derived from arm_v9.urdf:")
    for mesh_name, entry in attachment_manifest.items():
        print(f"  {mesh_name} -> {entry['semantic_link']}")


if __name__ == "__main__":
    main()
