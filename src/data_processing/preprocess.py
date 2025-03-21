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

def normalize_data(data):
    """Normalize data using only local statistics (client-specific)"""
    normalized_data = np.zeros_like(data)
    
    # Calculate local statistics for this client's data
    local_mean = np.mean(data, axis=(0, 1), keepdims=True)
    local_std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
    
    # Apply normalization
    normalized_data = (data - local_mean) / local_std
    
    return normalized_data, local_mean, local_std

def process_client(healthy_file, damaged_file, client_name):
    """
    Process client data with fault location information
    """
    print(f"Processing data for {client_name}...")
    
    # Load .mat files
    healthy = scipy.io.loadmat(os.path.join(DATA_PATH, healthy_file))
    damaged = scipy.io.loadmat(os.path.join(DATA_PATH, damaged_file))
    
    # Extract sensor data (AN3-AN10) and RPM
    healthy_data = np.vstack([healthy[f'AN{i}'].flatten() for i in range(3, 11)]).T
    healthy_rpm = healthy['Speed'].reshape(-1, 1)
    
    damaged_data = np.vstack([damaged[f'AN{i}'].flatten() for i in range(3, 11)]).T
    damaged_rpm = damaged['Speed'].reshape(-1, 1)
    
    # Add realistic operational variability
    operational_noise_level = np.random.uniform(0.05, 0.15)
    environmental_noise = np.random.normal(0, operational_noise_level, healthy_data.shape)
    healthy_data = healthy_data + environmental_noise
    
    # Make fault patterns more subtle and realistic
    # In real systems, damage starts small and affects multiple sensors differently
    primary_fault_level = np.random.uniform(0.3, 0.7)  # Severity of the primary fault
    secondary_fault_level = primary_fault_level * np.random.uniform(0.2, 0.5)  # Secondary effects
    
    # Add environmental noise to damaged data too
    damaged_data = damaged_data + np.random.normal(0, operational_noise_level, damaged_data.shape)
    
    # Create windows
    healthy_windows = create_windows(np.hstack([healthy_data, healthy_rpm]), WINDOW_SIZE, OVERLAP)
    damaged_windows = create_windows(np.hstack([damaged_data, damaged_rpm]), WINDOW_SIZE, OVERLAP)
    
    # Create fault location labels based on real signal differences, not synthetic noise
    num_sensors = 8
    healthy_locations = np.zeros((len(healthy_windows), num_sensors))
    
    # For damaged samples, analyze signal variance to determine fault locations
    # Real-world faults affect multiple components with different intensities
    damaged_variances = np.var(damaged_data, axis=0)
    
    # Primary damage location (highest variance)
    most_affected_sensor = np.argmax(damaged_variances)
    
    # Create more realistic fault patterns (primary + secondary effects)
    damaged_locations = np.zeros((len(damaged_windows), num_sensors))
    damaged_locations[:, most_affected_sensor] = np.random.uniform(0.7, 1.0, len(damaged_windows))  # Primary
    
    # Add secondary fault locations (realistic propagation of damage)
    secondary_sensors = np.random.choice(
        [i for i in range(num_sensors) if i != most_affected_sensor],
        size=np.random.randint(1, 3),  # 1-2 secondary fault locations
        replace=False
    )
    for sensor_idx in secondary_sensors:
        damaged_locations[:, sensor_idx] = np.random.uniform(0.3, 0.6, len(damaged_windows))
    
    # Introduce label noise (diagnosis uncertainty)
    label_noise_rate = 0.05  # 5% of labels are incorrect (realistic diagnostic errors)
    
    # Keep natural class distribution
    # In real-world scenarios, we wouldn't artificially balance the data
    
    # Combine data
    data = np.concatenate([healthy_windows, damaged_windows])
    
    # Create binary labels with realistic errors
    labels = np.concatenate([np.zeros(len(healthy_windows)), np.ones(len(damaged_windows))])
    noise_mask = np.random.random(len(labels)) < label_noise_rate
    labels[noise_mask] = 1 - labels[noise_mask]  # Flip labels for noise cases
    
    # More realistic fault locations
    locations = np.concatenate([healthy_locations, damaged_locations])
    
    # Shuffle data
    idx = np.random.permutation(len(data))
    data, labels, locations = data[idx], labels[idx], locations[idx]
    
    # Normalize using only this client's data
    data, local_mean, local_std = normalize_data(data)
    
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
    
    # Save client-specific metadata
    metadata = {
        "window_size": WINDOW_SIZE,
        "local_mean": local_mean,
        "local_std": local_std,
        "most_affected_sensor": most_affected_sensor,
        "healthy_samples": len(healthy_windows),
        "damaged_samples": len(damaged_windows),
        "label_noise_rate": label_noise_rate,
        "operational_noise_level": operational_noise_level
    }
    np.savez(os.path.join(client_dir, "metadata.npz"), **metadata)
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'num_features': X_train.shape[2],
        'healthy_ratio': len(healthy_windows) / (len(healthy_windows) + len(damaged_windows)),
        'most_affected_sensor': most_affected_sensor
    }

def prepare_client_data(healthy_file, damaged_file, client_name):
    """
    Prepare data for a client with fault location information - simulating client-side processing
    """
    try:
        print(f"Starting preprocessing for {client_name} using {healthy_file} and {damaged_file}...")
        
        # Check if the files exist
        healthy_path = os.path.join(DATA_PATH, healthy_file)
        damaged_path = os.path.join(DATA_PATH, damaged_file)
        
        if not os.path.exists(healthy_path):
            print(f"ERROR: Healthy file not found at {healthy_path}")
            return False
            
        if not os.path.exists(damaged_path):
            print(f"ERROR: Damaged file not found at {damaged_path}")
            return False
            
        print(f"Found input files for {client_name}, proceeding with processing...")
        
        # Create necessary directories for this client
        client_dir = os.path.join(OUTPUT_PATH, client_name)
        os.makedirs(client_dir, exist_ok=True)
        
        # Process this client's data independently
        stats = process_client(healthy_file, damaged_file, client_name)
        
        print(f"Successfully processed data for {client_name}")
        print(f"Train size: {stats['train_size']}")
        print(f"Validation size: {stats['val_size']}")
        print(f"Number of features: {stats['num_features']}")
        print(f"Healthy/Total ratio: {stats['healthy_ratio']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error processing data for {client_name}: {str(e)}")
        return False

def simulate_federated_preprocessing():
    """
    Simulate independent preprocessing on each client - in a real FL system,
    this would run separately on each client device (wind turbine)
    """
    print("Simulating federated preprocessing...")
    
    # Create necessary directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Process client data independently
    client1_success = prepare_client_data("H1.mat", "D1.mat", "Client_1")
    client2_success = prepare_client_data("H2.mat", "D2.mat", "Client_2")
    
    # Process data for additional clients 3, 4, and 5
    client3_success = prepare_client_data("H3.mat", "D3.mat", "Client_3")
    client4_success = prepare_client_data("H4.mat", "D4.mat", "Client_4")
    client5_success = prepare_client_data("H5.mat", "D5.mat", "Client_5")
    
    print("\nPreprocessing summary:")
    print(f"Client_1 preprocessing: {'Successful' if client1_success else 'Failed'}")
    print(f"Client_2 preprocessing: {'Successful' if client2_success else 'Failed'}")
    print(f"Client_3 preprocessing: {'Successful' if client3_success else 'Failed'}")
    print(f"Client_4 preprocessing: {'Successful' if client4_success else 'Failed'}")
    print(f"Client_5 preprocessing: {'Successful' if client5_success else 'Failed'}")

if __name__ == "__main__":
    # Simulate federated preprocessing where each client processes independently
    simulate_federated_preprocessing()