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

# --- Default Configuration Placeholders ---
# These values should ideally be defined in and imported from config.config.py
# They are provided here as fallbacks. Some are not directly used by preprocess.py anymore for primary binary labeling.

NUM_FAULT_CLASSES = 8 # Kept for potential future use, not for primary binary labels
FAULT_TYPE_MAPPING_CONFIG = { # Kept for potential future use, not for primary binary labels
    "Client_1": 1, "Client_2": 2, "Client_3": 3, "Client_4": 4, "Client_5": 5,
}
TRAIN_RATIO = 0.6
VALIDATION_RATIO_OF_REMAINING = 0.25
APPLY_FFT_FEATURES = False
APPLY_STATISTICAL_FEATURES = False
# --- End Default Configuration Placeholders ---

def create_windows(data, window_size, overlap):
    windows = []
    step = window_size - overlap
    # Ensure data has at least one window
    if data.shape[0] < window_size:
        # Handle cases with insufficient data, e.g., by returning an empty array or raising an error
        print(f"Warning: Data length {data.shape[0]} is less than window size {window_size}. Returning empty windows.")
        return np.array([])
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start+window_size])
    return np.array(windows)

def normalize_data(data):
    """Normalize data using only local statistics (client-specific)"""
    # Expects data of shape (num_windows, window_length, num_channels/features)
    if data.ndim == 2: # If it's (num_windows, window_length) for single channel/feature
        data = data[:, :, np.newaxis]
    
    local_mean = np.mean(data, axis=(0, 1), keepdims=True) # Mean over samples and time steps
    local_std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8 # Std over samples and time steps
    
    normalized_data = (data - local_mean) / local_std
    return normalized_data, local_mean, local_std

# Placeholder for FFT feature extraction
def extract_fft_features(windows_data, sampling_rate=40000):
    # Input: (num_windows, window_length, num_channels)
    # Output: (num_windows, num_fft_features, num_channels) or flattened
    print("Applying FFT feature extraction (placeholder)...")
    # Example: Compute magnitude spectrum for each channel in each window
    fft_features = np.abs(np.fft.fft(windows_data, axis=1))
    # Select relevant part of spectrum (e.g., positive frequencies)
    fft_features = fft_features[:, :windows_data.shape[1] // 2, :]
    # Further processing: binning, selecting peaks, etc.
    # For simplicity, returning a reshaped version or specific components
    # This needs to be properly implemented based on feature requirements.
    # For now, let's just return the mean of the FFT magnitudes as a dummy feature.
    return np.mean(fft_features, axis=1)

# Placeholder for statistical feature extraction
def extract_statistical_features(windows_data):
    # Input: (num_windows, window_length, num_channels)
    # Output: (num_windows, num_stat_features, num_channels) or flattened
    print("Applying statistical feature extraction (placeholder)...")
    # Example features: RMS, Kurtosis, Skewness, Peak-to-Peak for each channel
    rms = np.sqrt(np.mean(windows_data**2, axis=1))
    # kurtosis = scipy.stats.kurtosis(windows_data, axis=1) # Requires scipy.stats
    # For simplicity, returning RMS. More features can be added.
    # This needs to be properly implemented.
    return rms

def process_client(healthy_file, damaged_file, client_name):
    """
    Process client data:
    - Loads 8 sensor channels (AN3-AN10).
    - Generates BINARY fault labels (0 for healthy, 1 for damaged).
    - Generates per-sensor fault location/intensity for damaged data.
    - Applies windowing.
    - (Optional) Placeholder for feature extraction.
    - Normalizes data locally.
    - Splits data into train (60%), validation (10%), and test (30%).
    - Saves processed data and metadata.
    """
    print(f"Processing data for {client_name}...")
    
    healthy_mat = scipy.io.loadmat(os.path.join(DATA_PATH, healthy_file))
    damaged_mat = scipy.io.loadmat(os.path.join(DATA_PATH, damaged_file))
    
    healthy_sensor_data = np.vstack([healthy_mat[f'AN{i}'].flatten() for i in range(3, 11)]).T
    damaged_sensor_data = np.vstack([damaged_mat[f'AN{i}'].flatten() for i in range(3, 11)]).T

    operational_noise_level = np.random.uniform(0.05, 0.15)
    healthy_sensor_data += np.random.normal(0, operational_noise_level, healthy_sensor_data.shape)
    damaged_sensor_data += np.random.normal(0, operational_noise_level, damaged_sensor_data.shape)
    
    healthy_windows = create_windows(healthy_sensor_data, WINDOW_SIZE, OVERLAP)
    damaged_windows = create_windows(damaged_sensor_data, WINDOW_SIZE, OVERLAP)
    
    if healthy_windows.size == 0 or damaged_windows.size == 0:
        print(f"ERROR: No windows created for {client_name}. Skipping.")
        return None

    # --- BINARY Label Generation (0 for healthy, 1 for damaged) ---
    binary_healthy_labels = np.zeros(len(healthy_windows), dtype=int)
    binary_damaged_labels = np.ones(len(damaged_windows), dtype=int)

    # --- Per-sensor fault location/intensity (for damaged data) ---
    healthy_locations = np.zeros((len(healthy_windows), healthy_sensor_data.shape[1]))
    damaged_locations = np.full((len(damaged_windows), damaged_sensor_data.shape[1]), 0.0) # Initialize all to 0 or a default healthy score

    # Check for specific damage profile for the current damaged_file
    # Ensure SPECIFIC_SENSOR_DAMAGE_PROFILES is imported or accessed correctly, might need 'cfg.' prefix
    # Assuming cfg is imported as 'from config.config import *' or 'import config.config as cfg'
    # For this edit, I'll assume 'SPECIFIC_SENSOR_DAMAGE_PROFILES' is directly available if 'from config.config import *' was used
    # Or, if 'import config.config as cfg', then it would be 'cfg.SPECIFIC_SENSOR_DAMAGE_PROFILES'

    # Use a direct reference, assuming 'from config.config import *'
    profile_key = os.path.basename(damaged_file) # Get filename e.g. "seiko.mat"
    if SPECIFIC_SENSOR_DAMAGE_PROFILES and profile_key in SPECIFIC_SENSOR_DAMAGE_PROFILES:
        print(f"  Applying specific damage profile for {profile_key}...")
        profile = SPECIFIC_SENSOR_DAMAGE_PROFILES[profile_key]
        num_sensors = damaged_sensor_data.shape[1]
        
        target_healthy_score = profile.get("target_healthy_score", 0.05)
        target_damaged_score = profile.get("target_damaged_score", 0.95)
        
        # Initialize all sensor locations to the healthy score for this profile
        damaged_locations[:] = target_healthy_score
        
        # Set specified damaged sensors to the damaged score
        for sensor_idx in profile["damaged_indices"]:
            if 0 <= sensor_idx < num_sensors:
                damaged_locations[:, sensor_idx] = target_damaged_score
            else:
                print(f"    Warning: Sensor index {sensor_idx} in profile for {profile_key} is out of bounds (0-{num_sensors-1}).")
        
        # For this specific profile method, we won't use variance based most_affected_sensor_idx
        # but we still need to define it for metadata, perhaps set to -1 or first damaged sensor.
        most_affected_sensor_idx = profile["damaged_indices"][0] if profile["damaged_indices"] else -1

    else: # Fallback to original variance-based method
        print(f"  Using variance-based method for sensor locations for {profile_key}.")
        damaged_variances = np.var(damaged_sensor_data, axis=0)
        most_affected_sensor_idx = np.argmax(damaged_variances)
        
        damaged_locations[:, most_affected_sensor_idx] = np.random.uniform(0.7, 1.0, len(damaged_windows))
        secondary_sensors = np.random.choice(
            [i for i in range(damaged_sensor_data.shape[1]) if i != most_affected_sensor_idx],
            size=np.random.randint(1, 3), replace=False
        )
        for sensor_idx in secondary_sensors:
            damaged_locations[:, sensor_idx] = np.random.uniform(0.3, 0.6, len(damaged_windows))

    current_data_healthy = healthy_windows
    current_data_damaged = damaged_windows

    if APPLY_FFT_FEATURES:
        current_data_healthy = extract_fft_features(current_data_healthy)
        current_data_damaged = extract_fft_features(current_data_damaged)

    if APPLY_STATISTICAL_FEATURES:
        current_data_healthy = extract_statistical_features(current_data_healthy)
        current_data_damaged = extract_statistical_features(current_data_damaged)

    all_data = np.concatenate([current_data_healthy, current_data_damaged])
    all_binary_labels = np.concatenate([binary_healthy_labels, binary_damaged_labels])
    all_locations = np.concatenate([healthy_locations, damaged_locations])

    # Introduce label noise (applied to binary labels)
    label_noise_rate = 0.05
    noise_mask = np.random.random(len(all_binary_labels)) < label_noise_rate
    # Flip binary labels for noisy samples
    all_binary_labels[noise_mask] = 1 - all_binary_labels[noise_mask]
            
    idx = np.random.permutation(len(all_data))
    all_data, all_binary_labels, all_locations = all_data[idx], all_binary_labels[idx], all_locations[idx]
    
    all_data_normalized, local_mean, local_std = normalize_data(all_data)

    X_train, X_temp, y_train_binary, y_temp_binary, loc_train, loc_temp = train_test_split(
        all_data_normalized, all_binary_labels, all_locations,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_STATE,
        stratify=all_binary_labels # Stratify by binary labels
    )

    relative_val_size = VALIDATION_RATIO_OF_REMAINING 
    if len(np.unique(y_temp_binary)) < 2: # Cannot stratify with less than 2 unique labels (binary case)
         X_val, X_test, y_val_binary, y_test_binary, loc_val, loc_test = train_test_split(
            X_temp, y_temp_binary, loc_temp,
            test_size=(1.0 - relative_val_size),
            random_state=RANDOM_STATE
    )
    else:
        X_val, X_test, y_val_binary, y_test_binary, loc_val, loc_test = train_test_split(
            X_temp, y_temp_binary, loc_temp,
            test_size=(1.0 - relative_val_size),
            random_state=RANDOM_STATE,
            stratify=y_temp_binary # Stratify by binary labels
        )

    client_dir = os.path.join(OUTPUT_PATH, client_name)
    os.makedirs(client_dir, exist_ok=True)
    
    np.save(os.path.join(client_dir, "train_features.npy"), X_train)
    np.save(os.path.join(client_dir, "train_labels.npy"), y_train_binary) # Binary labels
    np.save(os.path.join(client_dir, "train_locations.npy"), loc_train)

    np.save(os.path.join(client_dir, "val_features.npy"), X_val)
    np.save(os.path.join(client_dir, "val_labels.npy"), y_val_binary) # Binary labels
    np.save(os.path.join(client_dir, "val_locations.npy"), loc_val)

    np.save(os.path.join(client_dir, "test_features.npy"), X_test)
    np.save(os.path.join(client_dir, "test_labels.npy"), y_test_binary) # Binary labels
    np.save(os.path.join(client_dir, "test_locations.npy"), loc_test)
    
    metadata = {
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "local_mean": local_mean,
        "local_std": local_std,
        "label_type": "binary_healthy_damaged", # Clarify label type
        "most_affected_sensor_original_idx": most_affected_sensor_idx,
        "num_healthy_windows_original": len(healthy_windows),
        "num_damaged_windows_original": len(damaged_windows),
        "label_noise_rate": label_noise_rate,
        "operational_noise_level": operational_noise_level,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "num_channels_or_features": X_train.shape[-1] if X_train.ndim == 3 else (X_train.shape[1] if X_train.ndim == 2 else None),
        "data_shape_example": X_train.shape
    }
    np.savez(os.path.join(client_dir, "metadata.npz"), **metadata)
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'num_input_dimensions': X_train.shape[1:],
        'healthy_ratio_original': len(healthy_windows) / (len(healthy_windows) + len(damaged_windows)) if (len(healthy_windows) + len(damaged_windows)) > 0 else 0,
        'most_affected_sensor_idx': most_affected_sensor_idx
    }

def prepare_client_data(healthy_file, damaged_file, client_name):
    """
    Prepare data for a client with BINARY fault labels (0=healthy, 1=damaged)
    and train/val/test splits.
    """
    try:
        print(f"Starting preprocessing for {client_name} using {healthy_file} and {damaged_file}...")
        
        healthy_path = os.path.join(DATA_PATH, healthy_file)
        damaged_path = os.path.join(DATA_PATH, damaged_file)
        
        if not os.path.exists(healthy_path):
            print(f"ERROR: Healthy file not found at {healthy_path}")
            return None # Return None instead of False for easier checking
            
        if not os.path.exists(damaged_path):
            print(f"ERROR: Damaged file not found at {damaged_path}")
            return None
            
        print(f"Found input files for {client_name}, proceeding with processing...")
        
        client_dir = os.path.join(OUTPUT_PATH, client_name)
        os.makedirs(client_dir, exist_ok=True)
        
        stats = process_client(healthy_file, damaged_file, client_name)
        
        if stats is None: # Check if process_client failed (e.g. no windows)
             print(f"Failed to process data for {client_name}. Stats object is None.")
             return None
        
        print(f"Successfully processed data for {client_name}")
        print(f"  Train size: {stats['train_size']}, Val size: {stats['val_size']}, Test size: {stats['test_size']}")
        print(f"  Input dimensions: {stats['num_input_dimensions']}")
        print(f"  Healthy/Total ratio (original): {stats['healthy_ratio_original']:.2f}")
        
        return stats # Return stats for summary
        
    except Exception as e:
        print(f"Error processing data for {client_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def simulate_federated_preprocessing():
    """
    Simulate independent preprocessing on each client.
    """
    print("Simulating federated preprocessing with BINARY labels (0=healthy, 1=damaged) and train/val/test splits...")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    clients_files = {
        "Client_1": ("H1.mat", "D1.mat"),
        "Client_2": ("H2.mat", "D2.mat"),
        "Client_3": ("H3.mat", "D3.mat"),
        "Client_4": ("H4.mat", "D4.mat"),
        "Client_5": ("H5.mat", "D5.mat"),
        "Client_Seiko": ("H1.mat", "seiko.mat") # Added Client_Seiko with seiko.mat
    }

    all_client_stats = {}

    for client_name, (h_file, d_file) in clients_files.items():
        client_stats = prepare_client_data(h_file, d_file, client_name)
        all_client_stats[client_name] = client_stats is not None # True if successful, False if failed
    
    print("\nPreprocessing summary:")
    for client_name, success_status in all_client_stats.items():
        print(f"  {client_name} preprocessing: {'Successful' if success_status else 'Failed'}")

if __name__ == "__main__":
    # Check if the placeholder default configurations are still in use.
    # This indicates they might not have been overridden by values from config.config.py.
    
    default_configs_in_use = []
    # Check a characteristic value of the default FAULT_TYPE_MAPPING_CONFIG
    if FAULT_TYPE_MAPPING_CONFIG.get("Client_1") == 1 and len(FAULT_TYPE_MAPPING_CONFIG) == 5 and \
       NUM_FAULT_CLASSES == 8: # Checking more specific default conditions for these unused configs
        default_configs_in_use.append("FAULT_TYPE_MAPPING_CONFIG and NUM_FAULT_CLASSES (Note: Not used for primary binary labeling)")
    if TRAIN_RATIO == 0.6: # Default value
        default_configs_in_use.append("TRAIN_RATIO")
    if VALIDATION_RATIO_OF_REMAINING == 0.25: # Default value
        default_configs_in_use.append("VALIDATION_RATIO_OF_REMAINING")
    if APPLY_FFT_FEATURES is False: # Default value
        default_configs_in_use.append("APPLY_FFT_FEATURES")
    if APPLY_STATISTICAL_FEATURES is False: # Default value
        default_configs_in_use.append("APPLY_STATISTICAL_FEATURES")

    # Check for essential variables expected from config.config import *
    essential_vars_from_config = {
        "WINDOW_SIZE": "e.g., 40000",
        "OVERLAP": "e.g., 0",
        "RANDOM_STATE": "e.g., 42",
        "OUTPUT_PATH": "e.g., 'data/processed'"
    }
    missing_essential_vars = []
    for var_name in essential_vars_from_config:
        if var_name not in globals():
            missing_essential_vars.append(var_name)

    if missing_essential_vars:
        print("\n--- CONFIGURATION WARNING ---")
        print("The following essential configuration variables are MISSING.")
        print("Please define them in your 'config/config.py' file:")
        for var_name in missing_essential_vars:
            print(f"  - {var_name} ({essential_vars_from_config[var_name]})")
        print("Cannot proceed without these essential configurations.")
        sys.exit(1) # Exit if essential configs are missing

    if default_configs_in_use:
        print("\n--- CONFIGURATION INFO ---")
        print("The following configuration variables are using their SCRIPT-LEVEL DEFAULTS:")
        for var_name in default_configs_in_use:
            print(f"  - {var_name}")
        print("It is recommended to define these explicitly in your 'config/config.py' file "
              "to ensure intended behavior and central management of configurations.")
        print("Continuing with script-level defaults for now...")
        print("--- END CONFIGURATION INFO ---\n")
    
    # Ensure other essential variables like WINDOW_SIZE, OVERLAP, RANDOM_STATE, OUTPUT_PATH
    # are correctly defined in your config.config module and thus imported by 'from config.config import *'
    
    simulate_federated_preprocessing()