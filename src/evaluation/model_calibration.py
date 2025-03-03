import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.model import GearboxCNNLSTM as GearboxCNN
from config.config import *
import os
import scipy.io
import argparse

def load_data(dataset_name):
    """Load and prepare data from a specific dataset"""
    file_path = os.path.join(DATA_PATH, f"{dataset_name}.mat")
    
    try:
        data = scipy.io.loadmat(file_path)
        print(f"Loading {dataset_name} dataset")
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")
    
    # Extract sensor data and RPM
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
    mean = np.load(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"))
    std = np.load(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"))
    
    # Normalize data
    normalized_windows = (windows - mean) / std
    
    return normalized_windows

def get_model_outputs(model, data_samples):
    """Get raw model outputs for data samples"""
    device = next(model.parameters()).device
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(data_samples), batch_size):
            batch = torch.FloatTensor(data_samples[i:i+batch_size]).to(device)
            outputs = model(batch).squeeze().cpu().numpy()
            all_outputs.extend(outputs)
    
    return np.array(all_outputs)

def find_optimal_threshold(healthy_outputs, damaged_outputs):
    """Find the optimal threshold to separate healthy and damaged samples"""
    min_healthy = np.min(healthy_outputs)
    max_healthy = np.max(healthy_outputs)
    min_damaged = np.min(damaged_outputs)
    max_damaged = np.max(damaged_outputs)
    
    print(f"Healthy output range: {min_healthy:.3f} - {max_healthy:.3f}")
    print(f"Damaged output range: {min_damaged:.3f} - {max_damaged:.3f}")
    
    # Try different threshold values
    thresholds = np.linspace(0.5, 0.95, 10)
    results = []
    
    for threshold in thresholds:
        # Calculate metrics
        healthy_accuracy = np.mean(healthy_outputs < threshold) * 100
        damaged_accuracy = np.mean(damaged_outputs >= threshold) * 100
        balanced_accuracy = (healthy_accuracy + damaged_accuracy) / 2
        
        results.append({
            'threshold': threshold,
            'healthy_accuracy': healthy_accuracy,
            'damaged_accuracy': damaged_accuracy,
            'balanced_accuracy': balanced_accuracy
        })
        
        print(f"Threshold {threshold:.2f}: Healthy Acc={healthy_accuracy:.1f}%, "
              f"Damaged Acc={damaged_accuracy:.1f}%, Balanced={balanced_accuracy:.1f}%")
    
    # Find best threshold
    best_threshold = max(results, key=lambda x: x['balanced_accuracy'])
    print(f"\nBest threshold: {best_threshold['threshold']:.3f} with "
          f"balanced accuracy: {best_threshold['balanced_accuracy']:.1f}%")
    
    return best_threshold['threshold']

def visualize_distributions(healthy_outputs, damaged_outputs, threshold=None):
    """Visualize the output distributions of healthy and damaged samples"""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(healthy_outputs, bins=30, alpha=0.5, label='Healthy', color='green')
    plt.hist(damaged_outputs, bins=30, alpha=0.5, label='Damaged', color='red')
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})')
    
    plt.xlabel('Model Output')
    plt.ylabel('Count')
    plt.title('Distribution of Model Outputs')
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/model_output_distributions.png')
    print("Saved distribution visualization to output/plots/model_output_distributions.png")

def create_calibration_curve(model, healthy_datasets, damaged_datasets):
    """Create and save model calibration curves"""
    # Load data
    healthy_samples = []
    for dataset in healthy_datasets:
        try:
            samples = load_data(dataset)
            healthy_samples.append(samples)
            print(f"Loaded {len(samples)} samples from {dataset}")
        except Exception as e:
            print(f"Could not load {dataset}: {e}")
    
    damaged_samples = []
    for dataset in damaged_datasets:
        try:
            samples = load_data(dataset)
            damaged_samples.append(samples)
            print(f"Loaded {len(samples)} samples from {dataset}")
        except Exception as e:
            print(f"Could not load {dataset}: {e}")
    
    if not healthy_samples or not damaged_samples:
        print("Error: Need at least one healthy and one damaged dataset")
        return
    
    healthy_data = np.concatenate(healthy_samples)
    damaged_data = np.concatenate(damaged_samples)
    
    # Get model outputs
    healthy_outputs = get_model_outputs(model, healthy_data)
    damaged_outputs = get_model_outputs(model, damaged_data)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(healthy_outputs, damaged_outputs)
    
    # Visualize distributions
    visualize_distributions(healthy_outputs, damaged_outputs, optimal_threshold)
    
    # Create and save threshold-to-accuracy mapping
    plt.figure(figsize=(10, 6))
    thresholds = np.linspace(0.1, 0.9, 50)
    healthy_accs = []
    damaged_accs = []
    balanced_accs = []
    
    for threshold in thresholds:
        healthy_acc = np.mean(healthy_outputs < threshold) * 100
        damaged_acc = np.mean(damaged_outputs >= threshold) * 100
        balanced_acc = (healthy_acc + damaged_acc) / 2
        
        healthy_accs.append(healthy_acc)
        damaged_accs.append(damaged_acc)
        balanced_accs.append(balanced_acc)
    
    plt.plot(thresholds, healthy_accs, label='Healthy Accuracy', color='green')
    plt.plot(thresholds, damaged_accs, label='Damaged Accuracy', color='red')
    plt.plot(thresholds, balanced_accs, label='Balanced Accuracy', color='blue')
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
               label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Classification Accuracy vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/threshold_calibration.png')
    print("Saved threshold calibration curve to output/plots/threshold_calibration.png")
    
    # Save the optimal threshold
    np.save("output/plots/optimal_threshold.npy", optimal_threshold)
    print(f"Saved optimal threshold ({optimal_threshold:.3f}) to output/plots/optimal_threshold.npy")
    
    return optimal_threshold

def main():
    parser = argparse.ArgumentParser(description='Calibrate model thresholds')
    parser.add_argument('--healthy', nargs='+', default=['H1', 'H6'], 
                        help='List of healthy datasets')
    parser.add_argument('--damaged', nargs='+', default=['D1', 'D2'], 
                        help='List of damaged datasets')
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GearboxCNN().to(device)
    try:
        model.load_state_dict(torch.load("best_global_model.pth", map_location=device))
        print("Successfully loaded best_global_model.pth")
    except FileNotFoundError:
        print("Error: best_global_model.pth not found!")
        return
    
    # Perform calibration
    optimal_threshold = create_calibration_curve(model, args.healthy, args.damaged)
    
    print(f"\nRecommended command to use the calibrated threshold:")
    print(f"python test_unseen_data.py --dataset H6 --threshold {optimal_threshold}")

if __name__ == "__main__":
    main()
