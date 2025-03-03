import numpy as np
import scipy.io
import os
import sys
from sklearn.model_selection import train_test_split

# Add the project root to Python path to allow importing from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import *

# Define the path to the raw data directory
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")

def create_windows(data, window_size, overlap):
    windows = []
    step = window_size - overlap
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start+window_size])
    return np.array(windows)

def normalize_data(data, rpm_max=1800):
    # Normalize each sensor independently
    normalized_data = np.zeros_like(data)
    
    # Sensor normalization (first 8 channels)
    for i in range(SENSORS):
        sensor_data = data[:, :, i]
        sensor_mean = np.mean(sensor_data)
        sensor_std = np.std(sensor_data)
        normalized_data[:, :, i] = (sensor_data - sensor_mean) / (sensor_std + 1e-8)
    
    # RPM normalization (last channel)
    rpm_data = data[:, :, RPM_SENSOR_INDEX]
    normalized_data[:, :, RPM_SENSOR_INDEX] = rpm_data / rpm_max
    
    return normalized_data, sensor_mean, sensor_std

def process_client(healthy_file, damaged_file, client_name):
    """
    Process client data with fault location information
    """
    # Load .mat files
    healthy = scipy.io.loadmat(os.path.join(DATA_PATH, healthy_file))
    damaged = scipy.io.loadmat(os.path.join(DATA_PATH, damaged_file))
    
    # Extract sensor data (AN3-AN10) and RPM
    healthy_data = np.vstack([healthy[f'AN{i}'].flatten() for i in range(3, 11)]).T
    healthy_rpm = healthy['Speed'].reshape(-1, 1)
    
    damaged_data = np.vstack([damaged[f'AN{i}'].flatten() for i in range(3, 11)]).T
    damaged_rpm = damaged['Speed'].reshape(-1, 1)
    
    # Create windows
    healthy_windows = create_windows(np.hstack([healthy_data, healthy_rpm]), WINDOW_SIZE, OVERLAP)
    damaged_windows = create_windows(np.hstack([damaged_data, damaged_rpm]), WINDOW_SIZE, OVERLAP)
    
    # Reduced noise levels
    noise_level = 0.1 if "Client_1" in client_name else 0.15
    
    # Add noise to make the task more challenging
    healthy_windows = healthy_windows + np.random.normal(0, noise_level, healthy_windows.shape)
    damaged_windows = damaged_windows + np.random.normal(0, noise_level, damaged_windows.shape)
    
    # Create fault location labels (one-hot encoded)
    # For healthy samples, all zeros. For damaged samples, identify the most affected sensor
    num_sensors = 8
    healthy_locations = np.zeros((len(healthy_windows), num_sensors))
    
    # For damaged samples, analyze signal variance to determine fault location
    damaged_variances = np.var(damaged_data, axis=0)
    most_affected_sensor = np.argmax(damaged_variances)
    damaged_locations = np.zeros((len(damaged_windows), num_sensors))
    damaged_locations[:, most_affected_sensor] = 1
    
    # Balanced class ratios
    healthy_ratio = 0.5  # Equal balance for all clients
    n_healthy = int(len(healthy_windows) * healthy_ratio)
    healthy_windows = healthy_windows[:n_healthy]
    healthy_locations = healthy_locations[:n_healthy]
    
    # Combine data
    data = np.concatenate([healthy_windows, damaged_windows])
    labels = np.concatenate([np.zeros(len(healthy_windows)), np.ones(len(damaged_windows))])
    locations = np.concatenate([healthy_locations, damaged_locations])
    
    # Add some label noise
    noise_mask = np.random.rand(len(labels)) < 0.05  # 5% label noise
    labels[noise_mask] = 1 - labels[noise_mask]
    
    # Shuffle data
    idx = np.random.permutation(len(data))
    data, labels, locations = data[idx], labels[idx], locations[idx]
    
    # Calculate or load global stats
    if client_name == "Client_1":  # First client calculates and saves
        global_mean = np.mean(data, axis=(0, 1), keepdims=True)
        global_std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
        np.save(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"), global_mean)
        np.save(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"), global_std)
    else:  # Other clients load existing stats
        global_mean = np.load(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"))
        global_std = np.load(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"))
    
    # Use global normalization
    data = (data - global_mean) / global_std
    
    # Split data
    X_train, X_val, y_train, y_val, loc_train, loc_val = train_test_split(
        data, labels, locations, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Save processed data
    client_dir = os.path.join(OUTPUT_PATH, client_name)
    os.makedirs(client_dir, exist_ok=True)
    
    np.save(os.path.join(client_dir, "train_features.npy"), X_train)
    np.save(os.path.join(client_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(client_dir, "train_locations.npy"), loc_train)
    np.save(os.path.join(client_dir, "val_features.npy"), X_val)
    np.save(os.path.join(client_dir, "val_labels.npy"), y_val)
    np.save(os.path.join(client_dir, "val_locations.npy"), loc_val)
    
    # Save metadata
    metadata = {
        "window_size": WINDOW_SIZE,
        "global_mean": global_mean,
        "global_std": global_std,
        "most_affected_sensor": most_affected_sensor
    }
    np.savez(os.path.join(client_dir, "metadata.npz"), **metadata)
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'num_features': X_train.shape[2]
    }

def prepare_data(healthy_file, damaged_file, client_name):
    """
    Prepare data for a client with fault location information
    """
    try:
        # Create necessary directories
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        os.makedirs(GLOBAL_STATS_PATH, exist_ok=True)
        
        # Process the client data
        stats = process_client(healthy_file, damaged_file, client_name)
        
        print(f"Successfully processed data for {client_name}")
        print(f"Train size: {stats['train_size']}")
        print(f"Validation size: {stats['val_size']}")
        print(f"Number of features: {stats['num_features']}")
        
        return True
        
    except Exception as e:
        print(f"Error processing data for {client_name}: {str(e)}")
        return False

if __name__ == "__main__":
    # Process data for each client
    prepare_data("H1.mat", "D1.mat", "Client_1")
    prepare_data("H2.mat", "D2.mat", "Client_2")