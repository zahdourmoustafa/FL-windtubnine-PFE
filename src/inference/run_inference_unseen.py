import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Added for metrics

# Add project root to sys.path to allow importing from src and config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.model import GearboxCNNLSTM
    from src.data_processing.preprocess import create_windows # Assuming create_windows is in preprocess.py
    from config import config as cfg # Import the main config
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print("Ensure your PYTHONPATH is set correctly or the script is run from the project root.")
    sys.exit(1)

def preprocess_unseen_data(mat_file_path, metadata):
    """Loads and preprocesses an unseen MATLAB data file."""
    print(f"Loading unseen data from: {mat_file_path}")
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"ERROR: MATLAB file not found at {mat_file_path}")
        return None

    # Extract sensor data (AN3-AN10)
    try:
        sensor_data_list = [mat_data[f'AN{i}'].flatten() for i in range(3, 11)]
        unseen_sensor_data = np.vstack(sensor_data_list).T # Shape: (num_samples, num_sensors)
    except KeyError as e:
        print(f"ERROR: Sensor key {e} not found in {mat_file_path}. Ensure it has AN3-AN10.")
        return None

    print(f"  Raw unseen data shape: {unseen_sensor_data.shape}")

    # Apply operational noise (using the level from Client_1's training for consistency)
    operational_noise_level = metadata.get('operational_noise_level')
    if operational_noise_level is None:
        print("Warning: 'operational_noise_level' not found in metadata. Generating a new random one.")
        operational_noise_level = np.random.uniform(0.05, 0.15)
    else:
        print(f"  Applying operational noise with level: {operational_noise_level:.4f}")
    
    unseen_sensor_data = unseen_sensor_data.astype(np.float32) # Ensure float32 for noise addition
    unseen_sensor_data += np.random.normal(0, operational_noise_level, unseen_sensor_data.shape)

    # Create windows
    window_size = metadata['window_size']
    overlap = metadata['overlap']
    unseen_windows = create_windows(unseen_sensor_data, window_size, overlap) # (num_windows, window_length, num_sensors)
    
    if unseen_windows.size == 0:
        print("ERROR: No windows created from the unseen data. Check data length and window parameters.")
        return None
    print(f"  Created {unseen_windows.shape[0]} windows with shape {unseen_windows.shape[1:]}")

    # Normalize data using Client_1's statistics
    local_mean = metadata['local_mean']
    local_std = metadata['local_std']
    
    # Ensure mean and std are broadcastable. Original shape from preprocess.py for normalize_data is (1,1,num_channels)
    # If it was saved differently, adjust here. For (num_windows, window_length, num_channels)
    # we need mean/std to be (1,1,num_channels) or (1, window_length, num_channels) or (num_windows, window_length, num_channels)
    # Given normalize_data in preprocess.py, it's (1,1,num_sensors)
    
    # Reshape mean and std if they are (num_sensors,) or similar, to ensure broadcasting
    if local_mean.ndim == 1 and local_mean.shape[0] == unseen_windows.shape[-1]: # (num_sensors,)
        local_mean = local_mean.reshape(1, 1, -1)
    if local_std.ndim == 1 and local_std.shape[0] == unseen_windows.shape[-1]: # (num_sensors,)
        local_std = local_std.reshape(1, 1, -1)

    # Safety check for dimensions after trying to reshape.
    expected_stat_shape_suffix = (1, 1, unseen_windows.shape[-1])
    if not (local_mean.shape == expected_stat_shape_suffix and local_std.shape == expected_stat_shape_suffix):
         # If still not matching, check if it's (1, D_features_model_expects, D_channels)
         # The saved mean/std from preprocess.py should be (1,1,num_channels)
         # based on `np.mean(data, axis=(0, 1), keepdims=True)` where data is (N, Win, Chan)
         print(f"Warning: local_mean shape {local_mean.shape} or local_std shape {local_std.shape} might not be directly broadcastable to data shape {unseen_windows.shape}. Expected mean/std to be like (1,1,num_channels). Will attempt to proceed.")


    normalized_unseen_windows = (unseen_windows - local_mean) / (local_std + 1e-8)
    print(f"  Normalized data shape: {normalized_unseen_windows.shape}")

    return torch.from_numpy(normalized_unseen_windows).float()


def main():
    # --- Configuration ---
    # !!! SET THE GROUND TRUTH STATE OF THE UNSEEN FILE !!!
    # Options: "healthy", "damaged", or "unknown" (if metrics are not needed/applicable for window-level)
    # For "mixed.mat" where only some windows might be damaged, set to "unknown" unless you have per-window labels.
    GROUND_TRUTH_STATE_OF_UNSEEN_FILE = "unknown" # MODIFIED_BY_USER 

    CLIENT_ID_FOR_MODEL = "Client_Seiko" # Model trained on this client
    MODEL_FILENAME = f"best_model_{CLIENT_ID_FOR_MODEL}.pth"
    SAVED_MODEL_PATH = os.path.join(project_root, MODEL_FILENAME)
    
    UNSEEN_DATA_FILENAME = "mixed.mat" # MODIFIED - Testing with mixed.mat
    UNSEEN_DATA_PATH = os.path.join(project_root, "data", "unseen_data", UNSEEN_DATA_FILENAME)
    
    # Ground truth for SENSOR-LEVEL analysis for mixed.mat (which sensors are truly faulty when damage occurs)
    # Stored as a set of sensor names like "AN3", "AN4", etc.
    KNOWN_FAULTY_SENSORS_GROUND_TRUTH = {"AN3", "AN7", "AN9"} # For mixed.mat

    METADATA_PATH = os.path.join(cfg.OUTPUT_PATH, CLIENT_ID_FOR_MODEL, "metadata.npz")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INFERENCE_BATCH_SIZE = getattr(cfg, 'BATCH_SIZE', 32) 
    
    DETECTION_THRESHOLD = 0.691 

    SENSOR_FAILURE_MODE_MAPPING = {
        "AN3": "Scuffing, polishing, fretting corrosion",
        "AN4": "Scuffing, polishing, fretting corrosion",
        "AN5": "Not directly mapped (Low-speed stage not listed)",
        "AN6": "Fretting corrosion, scuffing, polishing wear, assembly damage, dents",
        "AN7": "Scuffing",
        "AN8": "No direct match (downwind is mentioned)",
        "AN9": "Overheating",
        "AN10": "Fretting corrosion"
    }

    print(f"Using device: {DEVICE}")
    print(f"Loading model trained on: {CLIENT_ID_FOR_MODEL} from {SAVED_MODEL_PATH}")
    print(f"Analyzing unseen data: {UNSEEN_DATA_PATH}")
    if UNSEEN_DATA_FILENAME == "mixed.mat":
        print(f"  Expecting faults related to sensors: {KNOWN_FAULTY_SENSORS_GROUND_TRUTH}")
    print(f"Using metadata from: {METADATA_PATH}")

    # --- Load Metadata for Preprocessing ---
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Metadata file not found at {METADATA_PATH}")
        return
    try:
        client_metadata = dict(np.load(METADATA_PATH, allow_pickle=True))
        required_metadata_keys = ['window_size', 'overlap', 'local_mean', 'local_std']
        for key in required_metadata_keys:
            if key == 'overlap' and key not in client_metadata: 
                print(f"Warning: Key '{key}' not found in metadata. Falling back to value from config.py (cfg.OVERLAP).")
                client_metadata[key] = cfg.OVERLAP 
            elif key not in client_metadata:
                print(f"ERROR: Key '{key}' not found in metadata file {METADATA_PATH} and no fallback available for it.")
                return
        if client_metadata['window_size'] != cfg.WINDOW_SIZE:
             print(f"Warning: WINDOW_SIZE in metadata ({client_metadata['window_size']}) differs from config ({cfg.WINDOW_SIZE}). Using metadata's.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
        
    print(f"  Metadata loaded. Window Size: {client_metadata['window_size']}, Overlap: {client_metadata['overlap']}")

    # --- Load and Preprocess Unseen Data ---
    unseen_data_tensor = preprocess_unseen_data(UNSEEN_DATA_PATH, client_metadata)
    if unseen_data_tensor is None:
        return
    
    unseen_dataset = torch.utils.data.TensorDataset(unseen_data_tensor)
    unseen_loader = torch.utils.data.DataLoader(unseen_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    # --- Load Trained Model ---
    model_params = {
        'window_size': client_metadata['window_size'], 
        'lstm_hidden_size': getattr(cfg, 'LSTM_HIDDEN_SIZE', 32),
        'num_lstm_layers': getattr(cfg, 'NUM_LSTM_LAYERS', 1),
        'num_sensors': getattr(cfg, 'SENSORS', 8), # Should match the number of channels (AN3-AN10 is 8)
        'dropout_rate': getattr(cfg, 'DROPOUT_RATE', 0.3) 
    }
    model = GearboxCNNLSTM(**model_params).to(DEVICE)

    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"ERROR: Saved model file not found at {SAVED_MODEL_PATH}")
        return
    try:
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
        
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # --- Perform Inference ---
    all_fault_probs = []
    all_sensor_anomalies = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_data_tuple in unseen_loader:
            batch_data = batch_data_tuple[0].to(DEVICE) 
            outputs = model(batch_data)
            
            fault_logits = outputs['fault_detection'].squeeze()
            fault_probs = torch.sigmoid(fault_logits)
            
            sensor_anomaly_logits = outputs['sensor_anomalies']
            sensor_anomaly_probs = torch.sigmoid(sensor_anomaly_logits)

            all_fault_probs.append(fault_probs.cpu().numpy())
            all_sensor_anomalies.append(sensor_anomaly_probs.cpu().numpy())
    
    if not all_fault_probs:
        print("No predictions were made. Check data loading and preprocessing.")
        return

    all_fault_probs = np.concatenate(all_fault_probs)
    all_sensor_anomalies = np.concatenate(all_sensor_anomalies) 
    
    print(f"Inference completed. Processed {all_fault_probs.shape[0]} windows.")

    # --- Analyze Predictions ---
    # Window-level Fault Detection
    predicted_labels_binary = (all_fault_probs >= DETECTION_THRESHOLD).astype(int)
    num_damaged_windows = np.sum(predicted_labels_binary)
    total_windows = len(predicted_labels_binary)
    percentage_damaged_windows = (num_damaged_windows / total_windows) * 100 if total_windows > 0 else 0

    print(f"\\n--- Window-Level Inference Results ({UNSEEN_DATA_FILENAME}) ---")
    print(f"Using Detection Threshold: {DETECTION_THRESHOLD:.3f}")
    print(f"Total windows processed: {total_windows}")
    print(f"Number of windows predicted as DAMAGED by model: {num_damaged_windows}")
    print(f"Percentage of windows predicted as DAMAGED by model: {percentage_damaged_windows:.2f}%")

    # --- Calculate and Print Window-Level Classification Metrics ---
    if GROUND_TRUTH_STATE_OF_UNSEEN_FILE != "unknown" and total_windows > 0:
        if GROUND_TRUTH_STATE_OF_UNSEEN_FILE == "damaged":
            true_labels = np.ones_like(predicted_labels_binary)
            positive_label_str = "DAMAGED (1)"
        elif GROUND_TRUTH_STATE_OF_UNSEEN_FILE == "healthy":
            true_labels = np.zeros_like(predicted_labels_binary)
            positive_label_str = "DAMAGED (1)" 
        else:
            print(f"\\nWarning: Invalid value for GROUND_TRUTH_STATE_OF_UNSEEN_FILE ({GROUND_TRUTH_STATE_OF_UNSEEN_FILE}). Window-level metrics not calculated.")
            true_labels = None

        if true_labels is not None:
            accuracy = accuracy_score(true_labels, predicted_labels_binary)
            precision = precision_score(true_labels, predicted_labels_binary, zero_division=0)
            recall = recall_score(true_labels, predicted_labels_binary, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels_binary, zero_division=0)

            print(f"\\n--- Window-Level Classification Metrics (File Ground Truth: {GROUND_TRUTH_STATE_OF_UNSEEN_FILE}) ---")
            print(f"Positive class for Precision/Recall/F1: {positive_label_str}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
    elif total_windows == 0:
        print("\\nNo windows processed. Window-level metrics not calculated.")
    else: # GROUND_TRUTH_STATE_OF_UNSEEN_FILE is "unknown"
        print(f"\\nWindow-level ground truth for '{UNSEEN_DATA_FILENAME}' is '{GROUND_TRUTH_STATE_OF_UNSEEN_FILE}'. Window-level metrics not calculated.")


    # --- Sensor-Level Anomaly Analysis (for windows predicted as DAMAGED by model) ---
    if num_damaged_windows > 0:
        damaged_window_indices = np.where(predicted_labels_binary == 1)[0]
        sensor_anomalies_in_model_damaged_windows = all_sensor_anomalies[damaged_window_indices] 

        avg_anomaly_scores_in_model_damaged = np.mean(sensor_anomalies_in_model_damaged_windows, axis=0)
        
        most_anomalous_sensor_indices = np.argmax(sensor_anomalies_in_model_damaged_windows, axis=1)
        sensor_implication_counts = np.bincount(most_anomalous_sensor_indices, minlength=model_params['num_sensors'])

        # Identify sensors flagged by model
        DAMAGED_SENSOR_THRESHOLD_MODEL = 0.5 # Threshold for model to call a sensor "anomalous"
        model_flagged_sensor_indices = np.where(avg_anomaly_scores_in_model_damaged > DAMAGED_SENSOR_THRESHOLD_MODEL)[0]
        model_flagged_sensors_set = {f"AN{i+3}" for i in model_flagged_sensor_indices}


        print(f"\\n--- Sensor-Level Analysis (for {num_damaged_windows} windows PREDICTED AS DAMAGED by model) ---")
        print(f"Model's Damaged Sensor Threshold for this summary: > {DAMAGED_SENSOR_THRESHOLD_MODEL} avg. anomaly score")
        print(f"Model flagged sensors (based on avg score in its predicted damaged windows): {model_flagged_sensors_set if model_flagged_sensors_set else 'None'}")

        if UNSEEN_DATA_FILENAME == "mixed.mat" and KNOWN_FAULTY_SENSORS_GROUND_TRUTH:
            print(f"  Ground Truth Faulty Sensors for '{UNSEEN_DATA_FILENAME}': {KNOWN_FAULTY_SENSORS_GROUND_TRUTH}")
            
            correctly_identified_by_model = KNOWN_FAULTY_SENSORS_GROUND_TRUTH.intersection(model_flagged_sensors_set)
            missed_by_model = KNOWN_FAULTY_SENSORS_GROUND_TRUTH.difference(model_flagged_sensors_set)
            model_false_alarms_sensors = model_flagged_sensors_set.difference(KNOWN_FAULTY_SENSORS_GROUND_TRUTH)

            print(f"    Model Correctly Identified: {correctly_identified_by_model if correctly_identified_by_model else 'None'}")
            print(f"    Ground Truth Faulty Sensors Missed by Model: {missed_by_model if missed_by_model else 'None'}")
            print(f"    Sensors Flagged by Model but NOT in Ground Truth: {model_false_alarms_sensors if model_false_alarms_sensors else 'None'}")


        print("\\n  Detailed Sensor Anomaly Scores (Average in model's predicted DAMAGED windows):")
        for i, score in enumerate(avg_anomaly_scores_in_model_damaged):
            sensor_name = f"AN{i+3}"
            status = "FLAGGED BY MODEL" if sensor_name in model_flagged_sensors_set else "below model's threshold"
            failure_modes = SENSOR_FAILURE_MODE_MAPPING.get(sensor_name, "Unknown failure mode")
            print(f"    {sensor_name}: {score:.4f} ({status}) -> Potential Failure Mode(s): {failure_modes}")
        
        print("\\n  Frequency of Sensor being MOST Anomalous (in model's predicted DAMAGED windows):")
        for i, count in enumerate(sensor_implication_counts):
            sensor_name = f"AN{i+3}"
            percentage_implication = (count / num_damaged_windows) * 100 if num_damaged_windows > 0 else 0
            status = "FLAGGED BY MODEL" if sensor_name in model_flagged_sensors_set else "" # Might not be relevant here if only looking at 'most'
            print(f"    {sensor_name}: {count} times ({percentage_implication:.2f}%) {status.lower()}")
        
        anomaly_threshold_for_counting = 0.5
        sensors_frequently_high_in_damaged = np.sum(sensor_anomalies_in_model_damaged_windows > anomaly_threshold_for_counting, axis=0)
        print(f"\\n  Number of model's DAMAGED windows where sensor anomaly > {anomaly_threshold_for_counting}:")
        for i, count in enumerate(sensors_frequently_high_in_damaged):
             sensor_name = f"AN{i+3}"
             percentage_freq = (count / num_damaged_windows) * 100 if num_damaged_windows > 0 else 0
             status = "FLAGGED BY MODEL" if sensor_name in model_flagged_sensors_set else ""
             print(f"    {sensor_name}: {count} windows ({percentage_freq:.2f}%) {status.lower()}")

    else:
        print("\\nModel predicted NO windows as DAMAGED. Sensor-level anomaly analysis skipped.")
    
    print("\\n--- End of Inference ---")

if __name__ == '__main__':
    main() 