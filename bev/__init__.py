"""
BEV (Bird's Eye View) Projection Pipeline.

Phase 2: Lift detections from 2D image to 3D ego-frame BEV space.

Modules:
  - bev_grid.py: Grid configuration (200×200, [0,50]m × [-25,25]m)
  - lift_to_3d.py: Project 2D detections to 3D ego frame (GT-depth mode)
  - lidar_project.py: Extract LiDAR depth per detection (robust estimation)
"""

from .bev_grid import (
    GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX,
    CELL_SIZE, GRID_WIDTH, GRID_HEIGHT,
    world_to_cell, cell_to_world, create_empty_grid, print_grid_config
)

from .lift_to_3d import (
    lift_to_3d,
    lift_detection_to_3d_gt_depth,
    lift_detections_batch_gt_depth,
    lift_detection_to_3d_with_depth,
    lift_detections_batch_lidar_depth,
    get_camera_intrinsics,
    get_camera_extrinsics,
    get_frame_calibration,
    project_gt_box_centers_to_image,
)

from .lidar_project import (
    map_pointcloud_to_image,
    map_sample_lidar_to_image,
    extract_depth_per_detection,
    extract_depth_per_detection_devkit,
    load_lidar_points_ego,
)

__all__ = [
    "GRID_X_MIN", "GRID_X_MAX", "GRID_Y_MIN", "GRID_Y_MAX",
    "CELL_SIZE", "GRID_WIDTH", "GRID_HEIGHT",
    "world_to_cell", "cell_to_world", "create_empty_grid", "print_grid_config",
    "lift_to_3d", "lift_detection_to_3d_gt_depth", "lift_detections_batch_gt_depth",
    "lift_detection_to_3d_with_depth", "lift_detections_batch_lidar_depth",
    "get_camera_intrinsics", "get_camera_extrinsics", "get_frame_calibration",
    "project_gt_box_centers_to_image",
    "map_pointcloud_to_image", "map_sample_lidar_to_image",
    "extract_depth_per_detection", "extract_depth_per_detection_devkit",
    "load_lidar_points_ego",
]