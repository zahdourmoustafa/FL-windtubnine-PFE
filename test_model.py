import os
import sys
import numpy as np
import scipy.io
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Attempt to import from config and model
try:
    from config.config import (
        WINDOW_SIZE, OVERLAP, SENSORS,  # Data processing parameters
        BASE_DIR, GLOBAL_STATS_PATH as GLOBAL_STATS_DIR # Paths
    )
    # Assuming the model class is named 'SensorFaultDetector' in 'src.models.model'
    # This might need adjustment if the class name or file name is different.
    from src.models.model import GearboxCNNLSTM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure config/config.py and src/models/model.py are correctly set up.")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pth")
UNSEEN_DATA_DIR = os.path.join(BASE_DIR, "data", "unseen_data")
GLOBAL_STATS_FILE = os.path.join(GLOBAL_STATS_DIR, "global_stats.npz")

# Sensor names corresponding to AN3-AN10 for an 8-sensor setup
SENSOR_NAMES = [f"AN{i+3}" for i in range(SENSORS)] 

# --- Helper Functions ---

def load_global_stats(stats_file):
    """Loads global mean and std from the .npz file."""
    if not os.path.exists(stats_file):
        print(f"Error: Global stats file not found at {stats_file}")
        print("Please run the calculate_global_stats.py script first.")
        sys.exit(1)
    try:
        stats = np.load(stats_file)
        global_mean = stats['global_mean']
        global_std = stats['global_std']
        print(f"Loaded global mean (shape: {global_mean.shape}) and std (shape: {global_std.shape}) from {stats_file}")
        return global_mean, global_std
    except Exception as e:
        print(f"Error loading global stats: {e}")
        sys.exit(1)

def create_windows_from_data(data, window_size, overlap):
    """Creates overlapping windows from sensor data."""
    # data shape: (num_samples, num_channels)
    windows = []
    step = window_size - overlap
    if data.shape[0] < window_size:
        print(f"Warning: Data length {data.shape[0]} is less than window size {window_size}. No windows created.")
        return np.array([])
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start + window_size, :])
    return np.array(windows) # Shape: (num_windows, window_size, num_channels)

def preprocess_single_mat_file(mat_filepath, global_mean, global_std, window_size, overlap, num_sensors=8):
    """Loads a single .mat file, extracts sensor data, windows, and normalizes it."""
    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        # Extract sensor data AN3 to AN10 (assuming they are keys)
        # Data is expected to be (samples, 1) per sensor, so stack them to (samples, num_sensors)
        sensor_data_list = []
        for i in range(num_sensors):
            sensor_key = f'AN{i+3}'
            if sensor_key in mat_data:
                channel_data = mat_data[sensor_key].astype(np.float32).flatten()
                sensor_data_list.append(channel_data)
            else:
                print(f"Error: Sensor key {sensor_key} not found in {mat_filepath}.")
                return None
        
        # Ensure all channels have the same length
        min_len = min(len(ch) for ch in sensor_data_list)
        sensor_data_array = np.array([ch[:min_len] for ch in sensor_data_list]).T # Shape: (min_len, num_sensors)

        # Create windows
        windows = create_windows_from_data(sensor_data_array, window_size, overlap)
        if windows.size == 0:
            print(f"No windows created for {mat_filepath}. Skipping.")
            return None

        # Normalize data: (windows - mean) / std
        # global_mean and global_std are likely (1, 1, num_sensors) or similar for broadcasting
        normalized_windows = (windows - global_mean) / (global_std + 1e-8) # Add epsilon to std for stability
        return normalized_windows
    except Exception as e:
        print(f"Error processing {mat_filepath}: {e}")
        return None

def get_ground_truth_for_file(filename_simple, num_windows, num_sensors=SENSORS, healthy_score=0.05, damaged_score=0.95):
    """Generates ground truth labels based on filename and user specification."""
    gt_overall_label_scalar = 0
    gt_sensor_locations_pattern = np.full(num_sensors, healthy_score, dtype=np.float32)

    if filename_simple == "test_1.mat":
        gt_overall_label_scalar = 1  # Damaged
        gt_sensor_locations_pattern[:] = damaged_score # All sensors damaged
    elif filename_simple == "test_2.mat":
        gt_overall_label_scalar = 0  # Healthy
        gt_sensor_locations_pattern[:] = healthy_score # All sensors healthy
    elif filename_simple == "mixed.mat":
        gt_overall_label_scalar = 1  # Damaged
        # AN3 (idx 0), AN7 (idx 4), AN9 (idx 6) are damaged
        damaged_indices = [0, 4, 6] 
        for idx in damaged_indices:
            if 0 <= idx < num_sensors:
                gt_sensor_locations_pattern[idx] = damaged_score
    else:
        print(f"Warning: No ground truth defined for {filename_simple}. Assuming healthy.")
        # Default to healthy if undefined for safety, though this case shouldn't be hit with defined test files.

    # Repeat for each window
    gt_overall_labels_array = np.full(num_windows, gt_overall_label_scalar, dtype=int)
    gt_sensor_locations_array = np.tile(gt_sensor_locations_pattern, (num_windows, 1))
    
    return gt_overall_labels_array, gt_sensor_locations_array

# --- Main Testing Logic ---
def main():
    print("Starting model testing script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load global statistics
    global_mean, global_std = load_global_stats(GLOBAL_STATS_FILE)

    # 2. Load Model
    # Determine model input channels from global_mean (should be num_sensors)
    # Assuming global_mean shape is (1,1,num_features) or (num_features)
    if global_mean.ndim == 3: # (1,1,C)
        model_input_channels = global_mean.shape[2]
    elif global_mean.ndim == 1: # (C)
        model_input_channels = global_mean.shape[0]
    else:
        print(f"Error: Unexpected global_mean shape: {global_mean.shape}. Cannot determine model input channels.")
        sys.exit(1)
        
    if model_input_channels != SENSORS:
        print(f"Warning: SENSORS in config ({SENSORS}) doesn't match features in global_stats ({model_input_channels}). Using {model_input_channels}.")
    
    # Assuming SensorFaultDetector takes input_channels and num_classes (for overall classification)
    # The num_classes for overall is 1 (damage vs no damage, outputting logits)
    # The per-sensor output is a vector of SENSORS scores.
    try:
        # We need to know the exact signature of SensorFaultDetector.
        # Common parameters: input_channels (or num_features), num_classes_overall, num_sensors_output.
        # Based on common practice for such a model described:
        model = GearboxCNNLSTM(
            window_size=WINDOW_SIZE,       # From config.py
            lstm_hidden_size=32,           # Default from GearboxCNNLSTM
            num_lstm_layers=1,             # Default from GearboxCNNLSTM
            num_sensors=SENSORS,           # From config.py, should match model_input_channels
            dropout_rate=0.0               # Set to 0.0 for evaluation
        )
        # THE ABOVE LINE IS A GUESS. The actual parameters for SensorFaultDetector MUST be verified
        # from src/models/model.py. This is a common point of failure.
        print(f"Instantiated GearboxCNNLSTM model.")
    except TypeError as e:
        print(f"TypeError when instantiating GearboxCNNLSTM: {e}")
        print("Please verify the constructor arguments for GearboxCNNLSTM in src/models/model.py.")
        print("Common args: window_size, lstm_hidden_size, num_lstm_layers, num_sensors, dropout_rate etc.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred when instantiating GearboxCNNLSTM: {e}")
        sys.exit(1)


    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    try:
        # Adjust map_location if loading a GPU-trained model on CPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {MODEL_PATH} and set to evaluation mode on {device}.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    # 3. Define Unseen Files to Test
    unseen_files = {
        "test_1.mat": "All sensors damaged",
        "test_2.mat": "All sensors healthy",
        "mixed.mat": "AN3, AN7, AN9 damaged"
    }

    all_results = {}

    for filename, description in unseen_files.items():
        print(f"\n--- Testing file: {filename} ({description}) ---")
        mat_filepath = os.path.join(UNSEEN_DATA_DIR, filename)

        if not os.path.exists(mat_filepath):
            print(f"Warning: File {mat_filepath} not found. Skipping.")
            continue

        # Preprocess the .mat file
        # Reshape global_mean, global_std to be (1, 1, num_sensors) for normalization if they are not already
        # The create_windows function returns (num_windows, window_size, num_channels)
        # The normalization logic expects mean/std to broadcast over window_size.
        # Mean/std from stats script are (1,1,C), so they should broadcast correctly over (N, W, C).
        processed_windows = preprocess_single_mat_file(mat_filepath, global_mean, global_std, WINDOW_SIZE, OVERLAP, SENSORS)

        if processed_windows is None or processed_windows.size == 0:
            print(f"Could not process {filename}. Skipping.")
            continue
        
        num_windows_processed = processed_windows.shape[0]
        print(f"Processed {filename} into {num_windows_processed} windows of shape {processed_windows.shape[1:]}")

        # Get Ground Truth
        gt_overall_labels, gt_sensor_locations = get_ground_truth_for_file(filename, num_windows_processed, SENSORS)

        # Convert to PyTorch tensors
        windows_tensor = torch.from_numpy(processed_windows).float().to(device)
        # Model might expect (batch, channels, sequence_length) if using Conv1D,
        # or (batch, sequence_length, channels) if using LSTM directly on features.
        # Our current window shape is (N, W, C). Let's assume LSTM-style: (N, window_size, num_channels)
        # If model expects (N, C, W), then: windows_tensor = windows_tensor.permute(0, 2, 1)
        
        # Make Predictions
        pred_overall_logits_list = []
        pred_sensor_scores_list = []
        
        # Process in batches if necessary, though for typical test file sizes, one batch might be fine
        # For simplicity, predict all windows at once. If memory becomes an issue, batching is needed.
        with torch.no_grad():
            # Assuming model returns: overall_logits, sensor_scores
            # This aligns with the "senior ML engineer" persona's assumed model output.
            model_outputs = model(windows_tensor)
            pred_overall_raw_output_batch = model_outputs['fault_detection'] # This is already a probability if model.eval() and no mc_dropout
            pred_sensor_logits_batch = model_outputs['sensor_anomalies']   # These are logits
            pred_sensor_attention_weights = model_outputs.get('sensor_attention') # Use .get() for safety
        
        # Overall damage: output from model is already probability due to internal sigmoid in eval mode
        pred_overall_probs = pred_overall_raw_output_batch.cpu().numpy().squeeze()
        pred_overall_binary = (pred_overall_probs > 0.5).astype(int)
        
        # Per-sensor: apply sigmoid to logits
        pred_sensor_scores = torch.sigmoid(pred_sensor_logits_batch).cpu().numpy() # Shape: (num_windows, num_sensors)

        # DEBUG: Print some raw predicted sensor scores
        if filename == "test_1.mat" or filename == "mixed.mat":
            print(f"Sample predicted sensor scores for {filename} (first 3 windows):\n{pred_sensor_scores[:3, :]}")
            # For mixed.mat, also print ground truth for comparison
            if filename == "mixed.mat":
                print(f"Corresponding ground truth sensor locations for {filename} (first 3 windows):\n{gt_sensor_locations[:3, :]}")
            if pred_sensor_attention_weights is not None:
                print(f"Sample sensor attention weights for {filename} (first 3 windows):\n{pred_sensor_attention_weights[:3, :]}")

        # --- Evaluate Overall Damage ---
        print("\n--- Overall Damage Evaluation ---")
        overall_accuracy = accuracy_score(gt_overall_labels, pred_overall_binary)
        overall_precision = precision_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_recall = recall_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_f1 = f1_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_cm = confusion_matrix(gt_overall_labels, pred_overall_binary)

        print(f"  Accuracy:  {overall_accuracy:.4f}")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall:    {overall_recall:.4f}")
        print(f"  F1-Score:  {overall_f1:.4f}")
        print(f"  Confusion Matrix:\n{overall_cm}")

        # --- Evaluate Per-Sensor Damage ---
        print("\n--- Per-Sensor Damage/Location Evaluation ---")
        # Binarize ground truth scores (e.g., > 0.5 is damaged)
        # Using damaged_score directly from get_ground_truth_for_file for GT binarization
        # (healthy_score=0.05, damaged_score=0.95)
        gt_sensor_binary = (gt_sensor_locations >= 0.9).astype(int) # If score is close to damaged_score
        
        # Binarize predicted sensor scores
        pred_sensor_binary = (pred_sensor_scores > 0.5).astype(int)

        sensor_metrics = {}
        for i in range(SENSORS):
            sensor_name = SENSOR_NAMES[i]
            gt_sensor_i = gt_sensor_binary[:, i]
            pred_sensor_i = pred_sensor_binary[:, i]
            
            acc = accuracy_score(gt_sensor_i, pred_sensor_i)
            prec = precision_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            rec = recall_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            f1 = f1_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            
            sensor_metrics[sensor_name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
            print(f"  Sensor {sensor_name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # Average metrics across sensors (Macro Average)
        avg_sensor_acc = np.mean([m["acc"] for m in sensor_metrics.values()])
        avg_sensor_prec = np.mean([m["prec"] for m in sensor_metrics.values()])
        avg_sensor_rec = np.mean([m["rec"] for m in sensor_metrics.values()])
        avg_sensor_f1 = np.mean([m["f1"] for m in sensor_metrics.values()])
        print(f"  Average (Macro) Sensor Metrics: Acc={avg_sensor_acc:.4f}, Prec={avg_sensor_prec:.4f}, Rec={avg_sensor_rec:.4f}, F1={avg_sensor_f1:.4f}")

        # MSE/MAE on raw scores
        sensor_mse = mean_squared_error(gt_sensor_locations, pred_sensor_scores)
        sensor_mae = mean_absolute_error(gt_sensor_locations, pred_sensor_scores)
        print(f"  Sensor Scores MSE: {sensor_mse:.4f}")
        print(f"  Sensor Scores MAE: {sensor_mae:.4f}")

        all_results[filename] = {
            "overall_accuracy": overall_accuracy,
            "overall_f1": overall_f1,
            "avg_sensor_f1": avg_sensor_f1,
            "sensor_mse": sensor_mse
        }

    print("\n\n--- Overall Test Summary ---")
    for filename, metrics in all_results.items():
        print(f"File: {filename}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}, Overall F1: {metrics['overall_f1']:.4f}")
        print(f"  Avg Sensor F1:    {metrics['avg_sensor_f1']:.4f}, Sensor MSE: {metrics['sensor_mse']:.4f}")

if __name__ == "__main__":
    main()
