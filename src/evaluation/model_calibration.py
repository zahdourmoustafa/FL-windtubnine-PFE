import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score, precision_score, recall_score
import argparse

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the model and other utilities
from src.models.model import GearboxCNNLSTM
from config.config import *

def load_calibration_data():
    """Load validation data for threshold calibration"""
    # Load processed validation data from the last client
    val_data_path = os.path.join(OUTPUT_PATH, "client_2_val_data.npy")
    val_labels_path = os.path.join(OUTPUT_PATH, "client_2_val_labels.npy")
    
    try:
        X_val = np.load(val_data_path)
        y_val = np.load(val_labels_path)
        print(f"Loaded validation data: {X_val.shape}, labels: {y_val.shape}")
        return X_val, y_val
    except FileNotFoundError:
        print("Error: Validation data not found! Run training first.")
        return None, None

def calibrate_threshold(model, X_val, y_val, output_dir=None):
    """Find the optimal threshold for fault detection balancing precision and recall"""
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "output", "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Running calibration on {len(X_val)} validation samples...")
    
    with torch.no_grad():
        # Convert data to PyTorch tensors
        val_data = torch.FloatTensor(X_val).to(device)
        val_labels = y_val
        
        # Get predictions
        batch_size = 64
        all_probs = []
        
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            outputs = model(batch)
            probs = torch.sigmoid(outputs['fault_detection']).cpu().numpy().flatten()
            all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(val_labels, all_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = []
    for i in range(len(precision)):
        if i < len(thresholds):
            # Use the threshold to make predictions
            preds = (all_probs >= thresholds[i]).astype(int)
            f1 = f1_score(val_labels, preds)
            f1_scores.append(f1)
        else:
            # For the last point, use a very high threshold
            f1_scores.append(0)
    
    # Find thresholds meeting minimum recall requirement
    valid_idx = []
    for i, r in enumerate(recall):
        if r >= MIN_RECALL and i < len(thresholds):
            valid_idx.append(i)
    
    if len(valid_idx) > 0:
        # Find threshold with best F1 score among those with sufficient recall
        valid_f1_scores = [f1_scores[i] for i in valid_idx]
        best_idx = valid_idx[np.argmax(valid_f1_scores)]
        optimal_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        best_f1 = max(valid_f1_scores)
    else:
        # Use a low threshold to prioritize recall if none meet the minimum
        optimal_threshold = DEFAULT_THRESHOLD
        # Calculate metrics at this threshold
        preds = (all_probs >= optimal_threshold).astype(int)
        best_precision = precision_score(val_labels, preds)
        best_recall = recall_score(val_labels, preds)
        best_f1 = f1_score(val_labels, preds)
        
        print(f"Warning: No threshold found with recall >= {MIN_RECALL}")
        print(f"Using default threshold of {optimal_threshold} with recall {best_recall:.4f}")
    
    # Save optimal threshold
    np.save(os.path.join(output_dir, "optimal_threshold.npy"), optimal_threshold)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {auc(recall, precision):.3f})')
    plt.scatter([best_recall], [best_precision], color='red', s=100, zorder=10, 
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    # Add the minimum recall line
    plt.axvline(x=MIN_RECALL, color='orange', linestyle='--', 
               label=f'Minimum recall = {MIN_RECALL}')
    
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(val_labels, all_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    
    # Plot threshold vs F1 curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1])  # Exclude the last point which doesn't have a threshold
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
               label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "f1_vs_threshold.png"))
    
    # Plot threshold distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_probs, bins=50, alpha=0.7)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
               label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.title('Prediction Score Distribution')
    plt.xlabel('Fault Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    
    print(f"\nCalibration Results:")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"\nCalibration plots saved to {output_dir}")
    
    return optimal_threshold

def main():
    parser = argparse.ArgumentParser(description="Calibrate model threshold for optimal performance")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save calibration results")
    args = parser.parse_args()
    
    # Load validation data
    X_val, y_val = load_calibration_data()
    if X_val is None or y_val is None:
        return
    
    # Load the model
    model = GearboxCNNLSTM()
    try:
        model_path = os.path.join(BASE_DIR, "final_model.pth")
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: final_model.pth not found! Run training first.")
        return
    
    # Calibrate threshold
    optimal_threshold = calibrate_threshold(model, X_val, y_val, args.output_dir)
    
    print(f"\nCalibration complete. When testing, use:\n")
    print(f"python -m src.evaluation.test_unseen_data --dataset YOUR_DATASET --threshold {optimal_threshold:.4f}")

if __name__ == "__main__":
    main() 