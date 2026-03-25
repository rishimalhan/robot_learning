#!/usr/bin/env python3
"""
Greedily select a minimal pose set that covers the inspection part.

1. Sample many feasible camera poses within the ROI (shrink factor = 0.2).
2. For each pose, capture a single depth observation and measure voxel coverage.
3. Run a greedy set-cover algorithm to keep the smallest subset of poses whose
   union covers all observed voxels (or as much as we managed to sample).
"""

import rospy
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from neural_engine.rl_environment import InspectionEnv
from neural_engine.utils import sample_pose_within_roi, pose_within_bounds
from sensor_msgs.msg import PointCloud2
from train_a3c_inspection import ActorCritic, CHECKPOINT_PATH, DEVICE
from visualization_msgs.msg import Marker


POINT_RESOLUTION = 0.005  # meters
DROP_RATIO = 0.2


@dataclass
class PoseCoverage:
    position: np.ndarray
    orientation: np.ndarray
    points: np.ndarray
    point_keys: frozenset


def quantize_points(points: np.ndarray, resolution: float = POINT_RESOLUTION):
    if points.size == 0:
        return frozenset()
    quantized = np.round(points / resolution).astype(np.int32)
    return frozenset(map(tuple, quantized))


def capture_pointcloud(
    env: InspectionEnv, position: np.ndarray, orientation: np.ndarray
) -> Optional[np.ndarray]:
    env._apply_pose(position, orientation)
    cloud_msg = env._perception.trigger(
        publish=False,
        downsample=env._point_stride,
    )
    points_np = env._pointcloud_msg_to_numpy(cloud_msg)
    num_points = len(points_np) if points_np is not None and points_np.size > 0 else 0
    print(f"Received cloud message with {num_points} points for pose {position.tolist()}, {orientation.tolist()}")
    if points_np is None or points_np.size == 0 or not np.isfinite(points_np).all():
        return None
    return points_np


def prepare_policy_state(env: InspectionEnv, position: np.ndarray, orientation: np.ndarray):
    env._ctx.prev_action = np.zeros((env.action_dim,), dtype=np.float32)
    env._apply_pose(position, orientation)
    print(f"Applying pose: {position.tolist()}, {orientation.tolist()}")
    cloud_msg = env._perception.trigger(
        publish=False,
        downsample=env._point_stride,
    )
    env._voxel_grid.integrate_pointcloud(
        cloud_msg, pose_signature=env._pose_world_vec()
    )
    cloud_np = env._pointcloud_msg_to_numpy(cloud_msg)
    num_points = len(cloud_np) if cloud_np is not None and cloud_np.size > 0 else 0
    print(f"Received cloud message with {num_points} points for pose {position.tolist()}, {orientation.tolist()}")
    if cloud_np is None or cloud_np.size == 0 or not np.isfinite(cloud_np).all():
        return None, num_points
    points_cam = env._points_to_camera_frame(cloud_np)
    env._visualize_scene(cloud_msg=cloud_msg)
    state = env._build_state(points_camera=points_cam)
    return state, num_points


def greedy_select(candidates: List[PoseCoverage], min_distance: float = 0.1):
    """Greedy set-cover over quantized point keys with minimum position distance."""
    uncovered = set()
    for cand in candidates:
        uncovered.update(cand.point_keys)
    selected: List[PoseCoverage] = []
    remaining = candidates.copy()

    while uncovered:
        best = None
        best_gain = 0
        for cand in remaining:
            # Check minimum distance from already selected poses
            too_close = False
            for sel in selected:
                dist = float(np.linalg.norm(cand.position - sel.position))
                if dist < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
            
            gain = len(cand.point_keys & uncovered)
            if gain > best_gain:
                best = cand
                best_gain = gain
        if best is None or best_gain == 0:
            break
        selected.append(best)
        uncovered -= best.point_keys
        remaining = [cand for cand in remaining if cand is not best]
    return selected, uncovered


def prune_low_coverage(
    poses: List[PoseCoverage], drop_ratio: float = DROP_RATIO
) -> Tuple[List[PoseCoverage], List[PoseCoverage]]:
    if not poses or drop_ratio <= 0.0:
        return poses, []
    drop_count = int(np.floor(len(poses) * drop_ratio))
    if drop_count <= 0:
        return poses, []
    ordered = sorted(poses, key=lambda p: len(p.point_keys))
    dropped = ordered[:drop_count]
    kept = ordered[drop_count:]
    return kept, dropped


def nearest_pose(poses: List[PoseCoverage], reference: np.ndarray) -> int:
    ref = np.asarray(reference, dtype=np.float32)
    best_idx = 0
    best_dist = float("inf")
    for idx, pose in enumerate(poses):
        dist = float(np.linalg.norm(pose.position - ref))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def save_voxel_grid_state(voxel_grid) -> dict:
    """Save voxel grid state for restoration."""
    return {
        "occupancy": voxel_grid.occupancy.copy(),
        "_counts": voxel_grid._counts.copy(),
        "coverage": voxel_grid.coverage,
        "_surface_covered": voxel_grid._surface_covered.copy(),
        "_last_pose_signature": voxel_grid._last_pose_signature.copy() if voxel_grid._last_pose_signature is not None else None,
    }


def restore_voxel_grid_state(voxel_grid, state: dict):
    """Restore voxel grid state from saved state."""
    voxel_grid.occupancy = state["occupancy"]
    voxel_grid._counts = state["_counts"]
    voxel_grid.coverage = state["coverage"]
    voxel_grid._surface_covered = state["_surface_covered"]
    voxel_grid._last_pose_signature = state["_last_pose_signature"]


def load_policy_model(env: InspectionEnv) -> ActorCritic:
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Policy checkpoint not found at {CHECKPOINT_PATH}. Train the model first."
        )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def rollout_policy(
    env: InspectionEnv,
    model: ActorCritic,
    pose: PoseCoverage,
    pose_idx: int,
    steps: int = 10,
):
    state, num_points = prepare_policy_state(env, pose.position, pose.orientation)
    if state is None:
        if num_points == 0:
            print(f"[Pose {pose_idx}] Zero points detected, resetting environment to preserve coverage...")
            voxel_state = save_voxel_grid_state(env._voxel_grid)
            env.close()
            new_env = InspectionEnv(publish_pointcloud=False, visualize=False, point_stride=4)
            new_env.reset(reset_voxel_grid=False)
            restore_voxel_grid_state(new_env._voxel_grid, voxel_state)
            new_env._visualize = True
            new_env._marker_pub = rospy.Publisher(
                "/inspection_env/markers", Marker, queue_size=10
            )
            new_env._part_marker_pub = rospy.Publisher(
                "/inspection_env/part_marker", Marker, queue_size=1
            )
            new_env._debug_cloud_pub = rospy.Publisher(
                "/inspection_env/pointcloud", PointCloud2, queue_size=1
            )
            print(f"[Pose {pose_idx}] Environment reset, coverage preserved: {new_env._voxel_grid.coverage}")
            # Retry with new environment
            state, num_points = prepare_policy_state(new_env, pose.position, pose.orientation)
            if state is None:
                print(f"[Pose {pose_idx}] Still invalid after reset, skipping.")
                return False, new_env
            env = new_env
        else:
            print(f"[Pose {pose_idx}] Skipping due to invalid pointcloud.")
            return False, env
    
    print(f"\n[Pose {pose_idx}] Running policy for {steps} steps from selected pose.")
    prev_cov = env._voxel_grid.coverage
    for step_idx in range(1, steps + 1):
        state_tensor = torch.tensor(state[None, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            dist = model.dist(state_tensor)
            action = dist.mean.squeeze(0).cpu().numpy()
        next_state, reward, surface_done, terminated, info = env.step(action)
        done = surface_done
        breakdown = info.get("reward_breakdown", {})
        penalty = breakdown.get("penalty", 0.0) if isinstance(breakdown, dict) else 0.0
        current_cov = env._voxel_grid.coverage
        delta_cov = current_cov - prev_cov
        print(
            f"[Pose {pose_idx}] Step {step_idx} | reward={reward:.3f} | "
            f"coverage={current_cov:.3f} | delta_cov={delta_cov:.3f} | penalty={penalty:.3f}"
        )
        prev_cov = current_cov
        state = next_state
        if done or terminated:
            print(
                f"[Pose {pose_idx}] Episode ended early (done={done}, terminated={terminated})."
            )
            break
    return True, env


def main():
    rospy.init_node("run_inspection_policy", disable_signals=True)
    parser = argparse.ArgumentParser(
        description="Select inspection poses via greedy coverage."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of candidate poses to sample.",
    )
    parser.add_argument(
        "--shrink", type=float, default=0.2, help="Shrink factor for sampling ROI."
    )
    parser.add_argument(
        "--max-env-steps", type=int, default=1, help="Unused (compat placeholder)."
    )
    args = parser.parse_args()

    env = InspectionEnv(publish_pointcloud=False, visualize=False, point_stride=4)
    env.reset(reset_voxel_grid=True)
    candidates: List[PoseCoverage] = []

    try:
        attempts = 0
        while len(candidates) < args.samples and attempts < args.samples * 5:
            attempts += 1
            position, orientation = sample_pose_within_roi(
                env._roi_bounds, env._max_tilt, shrink_scale=args.shrink
            )
            if not pose_within_bounds(
                position=position,
                orientation=orientation,
                max_tilt=env._max_tilt,
                roi_bounds=env._roi_bounds,
            ):
                continue
            points = capture_pointcloud(
                env,
                np.asarray(position, dtype=np.float32),
                np.asarray(orientation, dtype=np.float32),
            )
            if points is None or points.size == 0:
                continue
            point_keys = quantize_points(points)
            if not point_keys:
                continue
            candidates.append(
                PoseCoverage(
                    position=np.asarray(position, dtype=np.float32),
                    orientation=np.asarray(orientation, dtype=np.float32),
                    points=points,
                    point_keys=point_keys,
                )
            )
        if not candidates:
            print("Failed to find any informative poses.")
            return

        selected, uncovered = greedy_select(candidates)
        selected, dropped = prune_low_coverage(selected, drop_ratio=DROP_RATIO)

        print(
            f"Sampled poses: {len(candidates)} | Greedy set size: {len(selected) + len(dropped)} | "
            f"Kept after pruning: {len(selected)} (dropped {len(dropped)})"
        )
        print(f"Uncovered points after selection: {len(uncovered)}")
        if dropped:
            print(
                f"Dropped coverage range: "
                f"{len(dropped[0].point_keys)}–{len(dropped[-1].point_keys)} voxels"
            )
        sortable = sorted(selected, key=lambda p: len(p.point_keys), reverse=True)
        for idx, pose in enumerate(sortable, 1):
            print(
                f"Pose {idx}: coverage={len(pose.point_keys)}, position={pose.position.tolist()}, "
                f"orientation={pose.orientation.tolist()}"
            )

        env.close()
        env = InspectionEnv(publish_pointcloud=False, visualize=True, point_stride=4)
        env.reset(reset_voxel_grid=False)
        try:
            policy_model = load_policy_model(env)
        except FileNotFoundError as exc:
            print(str(exc))
            return

        remaining = selected.copy()
        last_position = (
            env._ctx.position.copy()
            if env._ctx.position is not None
            else np.zeros(3, dtype=np.float32)
        )
        pose_counter = 1
        while remaining:
            idx = nearest_pose(remaining, last_position)
            pose = remaining.pop(idx)
            succeeded, env = rollout_policy(
                env, policy_model, pose, pose_idx=pose_counter, steps=10
            )
            if succeeded:
                last_position = (
                    env._ctx.position.copy()
                    if env._ctx.position is not None
                    else pose.position.copy()
                )
            else:
                print(f"[Pose {pose_counter}] skipped; retaining previous position target.")
            pose_counter += 1
            rospy.sleep(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
