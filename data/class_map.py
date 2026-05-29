# 9-class mapping from nuScenes 23-category taxonomy
#
# Vehicles: car, truck, bus, motorcycle, bicycle, emergency
# Pedestrians: all human.pedestrian subtypes → one class
# Static/infrastructure: barrier, traffic_cone
# Ignored (return None): animal, debris, pushable_pullable, bicycle_rack

CLASSES = [
    "car",           # 0
    "truck",         # 1
    "bus",           # 2
    "motorcycle",    # 3
    "bicycle",       # 4
    "emergency",     # 5
    "pedestrian",    # 6
    "barrier",       # 7
    "traffic_cone",  # 8
]

NUM_CLASSES = len(CLASSES)

# Maps each nuScenes category string to an integer class id, or None to ignore.
NUSCENES_TO_CLASS: dict[str, int | None] = {
    # ── Vehicles ─────────────────────────────────────────────────────────────
    "vehicle.car":                          0,
    "vehicle.truck":                        1,
    "vehicle.construction":                 1,  # collapse into truck
    "vehicle.trailer":                      1,  # collapse into truck
    "vehicle.bus.rigid":                    2,
    "vehicle.bus.bendy":                    2,
    "vehicle.motorcycle":                   3,
    "vehicle.bicycle":                      4,
    "vehicle.emergency.ambulance":          5,
    "vehicle.emergency.police":             5,

    # ── Pedestrians ──────────────────────────────────────────────────────────
    "human.pedestrian.adult":               6,
    "human.pedestrian.child":               6,
    "human.pedestrian.construction_worker": 6,
    "human.pedestrian.personal_mobility":   6,
    "human.pedestrian.police_officer":      6,
    "human.pedestrian.stroller":            6,
    "human.pedestrian.wheelchair":          6,

    # ── Static / infrastructure ───────────────────────────────────────────────
    "movable_object.barrier":               7,
    "movable_object.trafficcone":           8,

    # ── Ignored ───────────────────────────────────────────────────────────────
    "movable_object.debris":                None,
    "movable_object.pushable_pullable":     None,
    "static_object.bicycle_rack":           None,
    "animal":                               None,
}


def get_class_id(nuscenes_category: str) -> int | None:
    """Return the integer class id for a nuScenes category, or None to skip."""
    return NUSCENES_TO_CLASS.get(nuscenes_category, None)
