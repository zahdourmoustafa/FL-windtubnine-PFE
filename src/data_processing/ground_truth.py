"""
Ground truth data for sensor damage status across different datasets.
Used for calculating accuracy metrics in the evaluation process.

Format: 1 = damaged, 0 = healthy
"""

SENSOR_GROUND_TRUTH = {
    "unseenD": {"AN3": 1, "AN4": 1, "AN5": 1, "AN6": 1, "AN7": 1, "AN8": 1, "AN9": 1, "AN10": 1},
    "unseen": {"AN3": 1, "AN4": 0, "AN5": 1, "AN6": 0, "AN7": 0, "AN8": 0, "AN9": 0, "AN10": 0},
    "unseenH": {"AN3": 0, "AN4": 0, "AN5": 0, "AN6": 0, "AN7": 0, "AN8": 0, "AN9": 0, "AN10": 0}
} 