import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(model, X_test, y_test, threshold=0.5):
    """Calculate comprehensive metrics for the model"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        if isinstance(X_test, np.ndarray):
            X_test = torch.FloatTensor(X_test).to(device)
        if isinstance(y_test, np.ndarray):
            y_test = torch.FloatTensor(y_test).to(device)
            
        outputs = model(X_test).squeeze()
        predicted = (outputs > threshold).float()
        
        # Move tensors to CPU for sklearn metrics
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred) * 100  # Convert to percentage
        precision = precision_score(y_true, y_pred, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
    }

def load_unseen_dataset(file_path, window_size=128, overlap=64):
    """Load and prepare an unseen dataset for testing"""
    import scipy.io
    import os
    
    try:
        data = scipy.io.loadmat(file_path)
        filename = os.path.basename(file_path)
        print(f"Successfully loaded {filename}")
        
        # Extract sensor data and RPM
        sensor_data = np.vstack([data[f'AN{i}'].flatten() for i in range(3, 11)]).T
        rpm_data = data['Speed'].reshape(-1, 1)
        combined_data = np.hstack([sensor_data, rpm_data])
        
        # Create windows
        windows = []
        step = window_size - overlap
        for start in range(0, combined_data.shape[0] - window_size + 1, step):
            windows.append(combined_data[start:start+window_size])
        windows = np.array(windows)
        
        return windows, filename
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create and save a confusion matrix plot"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Healthy', 'Damaged'],
               yticklabels=['Healthy', 'Damaged'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    filename = f"confusion_matrix_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    return filename