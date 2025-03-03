import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class FaultLocalizationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
    
    def forward(self, outputs, targets):
        fault_detection = outputs['fault_detection'].squeeze()
        sensor_anomalies = outputs['sensor_anomalies']
        fault_label = targets['fault_label']
        
        # Detection loss
        detection_loss = self.bce(fault_detection, fault_label)
        
        # Sensor anomaly loss (only for faulty samples)
        faulty_mask = fault_label == 1
        if torch.any(faulty_mask):
            # For faulty samples, at least one sensor should show high anomaly
            max_anomaly_scores = torch.max(sensor_anomalies[faulty_mask], dim=1)[0]
            anomaly_loss = F.binary_cross_entropy(max_anomaly_scores, 
                                                torch.ones_like(max_anomaly_scores))
            
            # For healthy samples, all sensors should show low anomaly
            if torch.any(~faulty_mask):
                healthy_loss = F.binary_cross_entropy(
                    sensor_anomalies[~faulty_mask], 
                    torch.zeros_like(sensor_anomalies[~faulty_mask])
                )
                anomaly_loss = (anomaly_loss + healthy_loss) / 2
        else:
            anomaly_loss = torch.tensor(0.0).to(fault_detection.device)
        
        # Combined loss
        total_loss = detection_loss + self.alpha * anomaly_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'anomaly_loss': anomaly_loss
        }

def train_epoch(model, train_loader, optimizer, criterion=None, device=None):
    if criterion is None:
        criterion = FaultLocalizationLoss()
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss = 0
    detection_losses = 0
    anomaly_losses = 0
    
    for batch_idx, (data, labels, _) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        loss_dict = criterion(outputs, {
            'fault_label': labels
        })
        
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        total_loss += loss_dict['total_loss'].item()
        detection_losses += loss_dict['detection_loss'].item()
        anomaly_losses += loss_dict['anomaly_loss'].item()
    
    num_batches = len(train_loader)
    return {
        'total_loss': total_loss / num_batches,
        'detection_loss': detection_losses / num_batches,
        'anomaly_loss': anomaly_losses / num_batches
    }

def evaluate(model, val_loader, criterion=None, device=None):
    if criterion is None:
        criterion = FaultLocalizationLoss()
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0
    detection_losses = 0
    anomaly_losses = 0
    
    all_fault_preds = []
    all_fault_labels = []
    all_sensor_anomalies = []
    
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            
            loss_dict = criterion(outputs, {
                'fault_label': labels
            })
            
            total_loss += loss_dict['total_loss'].item()
            detection_losses += loss_dict['detection_loss'].item()
            anomaly_losses += loss_dict['anomaly_loss'].item()
            
            # Store predictions for metrics
            all_fault_preds.append(outputs['fault_detection'].cpu().numpy())
            all_fault_labels.append(labels.cpu().numpy())
            all_sensor_anomalies.append(outputs['sensor_anomalies'].cpu().numpy())
    
    # Concatenate all predictions
    fault_preds = np.concatenate(all_fault_preds)
    fault_labels = np.concatenate(all_fault_labels)
    sensor_anomalies = np.concatenate(all_sensor_anomalies)
    
    # Calculate detection metrics
    detection_auc = roc_auc_score(fault_labels, fault_preds)
    
    # Calculate precision, recall, and F1 score
    binary_preds = (fault_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(fault_labels, binary_preds, average='binary')
    
    # Calculate sensor-wise metrics
    sensor_metrics = {}
    for i in range(sensor_anomalies.shape[1]):
        sensor_name = f'AN{i+3}'  # AN3 through AN10
        # Calculate mean anomaly score for this sensor
        mean_anomaly = sensor_anomalies[:, i].mean()
        # Calculate anomaly AUC (how well sensor anomalies align with faults)
        sensor_auc = roc_auc_score(fault_labels, sensor_anomalies[:, i])
        sensor_metrics[sensor_name] = {
            'mean_anomaly': mean_anomaly,
            'anomaly_auc': sensor_auc
        }
    
    num_batches = len(val_loader)
    metrics = {
        'total_loss': total_loss / num_batches,
        'detection_loss': detection_losses / num_batches,
        'anomaly_loss': anomaly_losses / num_batches,
        'detection_auc': detection_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensor_metrics': sensor_metrics,
        'val_loss': total_loss / num_batches  # Add this for compatibility
    }
    
    # Print sensor-wise metrics
    print("\nSensor Anomaly Details:")
    for sensor, sensor_data in sensor_metrics.items():
        print(f"{sensor}: Mean Anomaly = {sensor_data['mean_anomaly']:.4f}, AUC = {sensor_data['anomaly_auc']:.4f}")
    
    # Print classification metrics
    print(f"\nClassification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return metrics

def train(model, train_loader, val_loader, config):
    device = torch.device(config.device)
    model = model.to(device)
    
    criterion = FaultLocalizationLoss(alpha=config.location_loss_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"Detection AUC: {val_metrics['detection_auc']:.4f}")
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model 