import torch
import numpy as np
import scipy.io
import os
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the model and configurations
from src.models.model import GearboxCNNLSTM
from config.config import *

def load_and_preprocess_data(dataset_name):
    """Load and preprocess unseen data"""
    # Construct the file path - data should be in data/unseen_data directory
    file_path = os.path.join(BASE_DIR, "data", "unseen_data", f"{dataset_name}.mat")
    
    try:
        print(f"Loading dataset {dataset_name} from {file_path}")
        data = scipy.io.loadmat(file_path)
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")
    
    # Extract sensor data and RPM (similar to preprocessing)
    sensor_data = np.vstack([data[f'AN{i}'].flatten() for i in range(3, 11)]).T
    rpm_data = data['Speed'].reshape(-1, 1)
    combined_data = np.hstack([sensor_data, rpm_data])
    
    # Create windows
    windows = []
    step = WINDOW_SIZE - OVERLAP
    for start in range(0, combined_data.shape[0] - WINDOW_SIZE + 1, step):
        windows.append(combined_data[start:start+WINDOW_SIZE])
    windows = np.array(windows)
    
    # Load global normalization stats
    try:
        mean = np.load(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"))
        std = np.load(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"))
        
        # Normalize data
        normalized_windows = (windows - mean) / std
        print(f"Data normalized using global stats. Shape: {normalized_windows.shape}")
    except FileNotFoundError:
        print("Warning: Global normalization stats not found. Using raw data.")
        normalized_windows = windows
    
    return normalized_windows

def predict_gearbox_health(model, data):
    """Make predictions on the windowed data"""
    device = next(model.parameters()).device
    model.eval()
    
    # Process in batches to avoid memory issues
    batch_size = 64
    fault_probabilities = []
    sensor_anomalies = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i:i+batch_size]).to(device)
            outputs = model(batch)
            
            # Extract fault detection probabilities
            fault_probs = outputs['fault_detection'].cpu().numpy()
            fault_probabilities.extend(fault_probs)
            
            # Extract sensor-specific anomaly scores
            anomalies = outputs['sensor_anomalies'].cpu().numpy()
            sensor_anomalies.extend(anomalies)
    
    return np.array(fault_probabilities), np.array(sensor_anomalies)

def load_threshold():
    """Load the optimal threshold or use default"""
    threshold_path = os.path.join(BASE_DIR, "output", "plots", "optimal_threshold.npy")
    try:
        threshold = float(np.load(threshold_path))
        print(f"Using calibrated threshold: {threshold:.3f}")
    except FileNotFoundError:
        threshold = 0.5
        print(f"Calibrated threshold not found, using default: {threshold:.3f}")
    
    return threshold

def visualize_results(fault_probs, predictions, dataset_name, sensor_anomalies):
    """Visualize the prediction results"""
    # Create output directory
    output_dir = os.path.join(BASE_DIR, "output", "unseen_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot fault probability distribution
    plt.figure(figsize=(12, 6))
    plt.hist(fault_probs, bins=30, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel('Fault Probability')
    plt.ylabel('Count')
    plt.title(f'Fault Probability Distribution - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_probability_dist.png"))
    
    # Plot sensor anomaly scores
    plt.figure(figsize=(14, 8))
    sensor_names = [f'AN{i}' for i in range(3, 11)]
    mean_anomalies = np.mean(sensor_anomalies, axis=0)
    plt.bar(sensor_names, mean_anomalies)
    plt.xlabel('Sensor')
    plt.ylabel('Mean Anomaly Score')
    plt.title(f'Sensor Anomaly Scores - {dataset_name}')
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_sensor_anomalies.png"))
    
    # Save results to a text file
    healthy_percent = (1 - np.mean(predictions)) * 100
    damaged_percent = np.mean(predictions) * 100
    
    with open(os.path.join(output_dir, f"{dataset_name}_results.txt"), 'w') as f:
        f.write(f"Results for {dataset_name}:\n")
        f.write(f"Healthy windows: {np.sum(~predictions)} ({healthy_percent:.2f}%)\n")
        f.write(f"Damaged windows: {np.sum(predictions)} ({damaged_percent:.2f}%)\n")
        f.write("\nSensor Anomaly Scores:\n")
        for i, sensor in enumerate(sensor_names):
            f.write(f"{sensor}: {mean_anomalies[i]:.4f}\n")
    
    print(f"Results saved to {output_dir}/{dataset_name}_results.txt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the model on unseen data")
    parser.add_argument("--dataset", required=True, help="Name of the dataset to test (without .mat extension)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold for fault detection")
    args = parser.parse_args()
    
    # Load and preprocess the data
    try:
        data = load_and_preprocess_data(args.dataset)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GearboxCNNLSTM().to(device)
    
    try:
        model_path = os.path.join(BASE_DIR, "final_model.pth")
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: final_model.pth not found at {os.path.join(BASE_DIR, 'final_model.pth')}!")
        return
    
    # Make predictions
    print(f"Making predictions on {len(data)} windows...")
    start_time = time.time()
    fault_probs, sensor_anomalies = predict_gearbox_health(model, data)
    
    # Use the provided threshold or load calibrated one
    threshold = args.threshold if args.threshold is not None else load_threshold()
    
    # Apply threshold to get binary predictions
    predictions = (fault_probs >= threshold).flatten()
    
    # Calculate overall health status
    percent_damaged = np.mean(predictions) * 100
    
    # Determine final health status based on percentage of damaged windows
    if percent_damaged > 40:  # If more than 40% of windows are classified as damaged
        health_status = "DAMAGED"
    elif percent_damaged > 15:  # If between 15-40% of windows are damaged
        health_status = "POTENTIALLY DAMAGED (Maintenance Recommended)"
    else:
        health_status = "HEALTHY"
    
    # Print results
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
    print(f"\nResults for dataset: {args.dataset}")
    print(f"------------------------------")
    print(f"Number of windows analyzed: {len(data)}")
    print(f"Healthy windows: {np.sum(~predictions)} ({100-percent_damaged:.2f}%)")
    print(f"Damaged windows: {np.sum(predictions)} ({percent_damaged:.2f}%)")
    print(f"\nOverall gearbox health status: {health_status}")
    
    # Print sensor-specific anomalies
    mean_anomalies = np.mean(sensor_anomalies, axis=0)
    print("\nSensor anomaly scores:")
    for i, score in enumerate(mean_anomalies):
        sensor_name = f'AN{i+3}'
        status = "HIGH" if score > 0.5 else "NORMAL"
        print(f"{sensor_name}: {score:.4f} - {status}")
    
    # If any sensor has high anomaly, give additional information
    high_anomaly_sensors = [f'AN{i+3}' for i, score in enumerate(mean_anomalies) if score > 0.5]
    if high_anomaly_sensors:
        print(f"\nHigh anomaly detected in sensors: {', '.join(high_anomaly_sensors)}")
        print("Recommendation: Inspect these specific sensors for potential damage.")
    
    # Visualize the results
    visualize_results(fault_probs, predictions, args.dataset, sensor_anomalies)

if __name__ == "__main__":
    main()
