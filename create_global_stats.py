import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.config import GLOBAL_STATS_PATH, SENSORS

def create_global_stats():
    """Create global normalization statistics for sensor data"""
    print(f"Creating global normalization statistics in {GLOBAL_STATS_PATH}")
    
    # Create reasonable mean and std values for vibration sensors
    # We use 8 sensors + 1 rpm value
    # Mean close to zero for vibration sensors and a higher value for RPM
    mean = np.zeros(SENSORS + 1)  # 8 sensors + RPM
    mean[-1] = 1500  # Average RPM value
    
    # Standard deviation - higher for vibration sensors, lower for RPM (as percentage of mean)
    std = np.ones(SENSORS + 1) * 0.5  # Base std for vibration sensors
    std[-1] = 300  # RPM variation
    
    # Save the statistics
    np.save(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"), mean)
    np.save(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"), std)
    
    print(f"Created global_mean.npy: {mean}")
    print(f"Created global_std.npy: {std}")
    print(f"Files saved in {GLOBAL_STATS_PATH}")

if __name__ == "__main__":
    create_global_stats() 