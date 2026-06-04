"""
BEV Grid Configuration and Utilities.

Defines the bird's-eye view grid extent, resolution, and coordinate transformations.

Grid Spec:
  - X (forward): [0, 50] m
  - Y (left): [-25, 25] m
  - Cell size: 0.25 m
  - Grid shape: 200 × 200
  - Origin: ego-frame (x forward, y left, z up)
"""

import numpy as np
from typing import Tuple


# Grid configuration (meters)
GRID_X_MIN = 0.0      # Behind ego
GRID_X_MAX = 50.0     # Forward
GRID_Y_MIN = -25.0    # Right
GRID_Y_MAX = 25.0     # Left
CELL_SIZE = 0.25      # meters per cell

# Derived. Rows index forward x; columns index lateral y.
GRID_HEIGHT = int((GRID_X_MAX - GRID_X_MIN) / CELL_SIZE)   # 200 rows
GRID_WIDTH = int((GRID_Y_MAX - GRID_Y_MIN) / CELL_SIZE)    # 200 cols


def world_to_cell(x_m: float, y_m: float) -> Tuple[int, int]:
    """
    Convert world coordinates (ego frame) to grid cell indices.
    
    Args:
        x_m: X position in meters (forward, [0, 50])
        y_m: Y position in meters (left, [-25, 25])
    
    Returns:
        (row, col) in grid coordinates, or None if out of bounds.
        
    Convention:
      - row 0 is x=0 (behind ego)
      - row 199 is x=50 (far forward)
      - col 0 is y=-25 (far right)
      - col 199 is y=25 (far left)
    """
    if x_m < GRID_X_MIN or x_m > GRID_X_MAX:
        return None
    if y_m < GRID_Y_MIN or y_m > GRID_Y_MAX:
        return None
    
    # Convert to cell indices
    row = int((x_m - GRID_X_MIN) / CELL_SIZE)
    col = int((y_m - GRID_Y_MIN) / CELL_SIZE)
    
    # Clamp to grid (in case of floating point errors)
    row = min(row, GRID_HEIGHT - 1)
    col = min(col, GRID_WIDTH - 1)
    
    return (row, col)


def cell_to_world(row: int, col: int) -> Tuple[float, float]:
    """
    Convert grid cell indices back to world coordinates.
    
    Args:
        row: Row index [0, 199]
        col: Column index [0, 199]
    
    Returns:
        (x_m, y_m) in ego frame coordinates
    """
    x_m = GRID_X_MIN + row * CELL_SIZE
    y_m = GRID_Y_MIN + col * CELL_SIZE
    return (x_m, y_m)


def create_empty_grid() -> np.ndarray:
    """
    Create an empty 200×200 BEV grid (float32).
    
    Returns:
        np.ndarray of shape (GRID_HEIGHT, GRID_WIDTH) initialized to 0.0
    """
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)


def print_grid_config():
    """Print grid configuration to console."""
    print("=" * 60)
    print("BEV GRID CONFIGURATION")
    print("=" * 60)
    print(f"X range (forward):  [{GRID_X_MIN}, {GRID_X_MAX}] m")
    print(f"Y range (left):     [{GRID_Y_MIN}, {GRID_Y_MAX}] m")
    print(f"Cell size:          {CELL_SIZE} m")
    print(f"Grid shape:         {GRID_HEIGHT} × {GRID_WIDTH} (row × col)")
    print(f"Total cells:        {GRID_HEIGHT * GRID_WIDTH}")
    print("=" * 60)
    print(f"Origin: ego-frame (x forward, y left, z up)")
    print(f"Row 0: x={GRID_X_MIN} m (behind ego)")
    print(f"Row {GRID_HEIGHT-1}: x={GRID_X_MAX} m (far forward)")
    print(f"Col 0: y={GRID_Y_MIN} m (far right)")
    print(f"Col {GRID_WIDTH-1}: y={GRID_Y_MAX} m (far left)")
    print("=" * 60)