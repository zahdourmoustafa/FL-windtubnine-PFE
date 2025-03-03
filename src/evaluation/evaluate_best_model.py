import torch
import numpy as np
from model import GearboxCNNLSTM as GearboxCNN
from federated_train import calculate_metrics
from config import *

def evaluate_best_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the best model
    model = GearboxCNN().to(device)
    try:
        model.load_state_dict(torch.load("best_global_model.pth"))
        print("Successfully loaded best_global_model.pth")
    except FileNotFoundError:
        print("Error: best_global_model.pth not found!")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test data from both clients
    all_metrics = []
    for client in ["Client_1", "Client_2"]:
        client_path = os.path.join(OUTPUT_PATH, client)
        
        # Load validation data (as test set)
        X_val = np.load(os.path.join(client_path, "val_features.npy"))
        y_val = np.load(os.path.join(client_path, "val_labels.npy"))
        
        # Load normalization stats
        try:
            mean = np.load(os.path.join(GLOBAL_STATS_PATH, "global_mean.npy"))
            std = np.load(os.path.join(GLOBAL_STATS_PATH, "global_std.npy"))
            
            # Normalize data
            X_val = (X_val - mean) / std
            
        except FileNotFoundError:
            print(f"Warning: Using non-normalized data for {client}")
        
        # Calculate metrics
        metrics = calculate_metrics(model, X_val, y_val)
        all_metrics.append(metrics)
        
        print(f"\nMetrics for {client}:")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"F1 Score: {metrics['f1_score']:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}%")
        print(f"Recall: {metrics['recall']:.2f}%")
    
    # Calculate average metrics across clients
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\nAverage metrics across all clients:")
    print(f"Accuracy: {avg_metrics['accuracy']:.2f}%")
    print(f"F1 Score: {avg_metrics['f1_score']:.2f}%")
    print(f"Precision: {avg_metrics['precision']:.2f}%")
    print(f"Recall: {avg_metrics['recall']:.2f}%")

if __name__ == "__main__":
    evaluate_best_model() 