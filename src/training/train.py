import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from config.config import DEFAULT_THRESHOLD, MIN_RECALL

class FaultLocalizationLoss(nn.Module):
    def __init__(self, alpha=0.6, label_smoothing=0.05, num_sensors=8, min_faulty_sensors=2, 
                 focal_gamma=2.0, pos_weight=3.0):
        super().__init__()
        self.alpha = alpha
        # Use BCEWithLogitsLoss with positive weight for handling class imbalance
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.num_sensors = num_sensors
        self.min_faulty_sensors = min_faulty_sensors
        self.focal_gamma = focal_gamma  # Focal loss gamma parameter
        self.pos_weight = pos_weight    # Weight for positive examples
    
    def forward(self, outputs, targets):
        fault_detection = outputs['fault_detection'].squeeze()
        sensor_anomalies = outputs['sensor_anomalies']
        fault_label = targets['fault_label']
        
        # Apply label smoothing for more realistic training signals
        smoothed_targets = fault_label.clone()
        smoothed_targets = smoothed_targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Calculate BCE loss
        bce_loss = self.bce(fault_detection, smoothed_targets)
        
        # Apply focal loss modifier to focus more on hard examples
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Apply class weight to address imbalance (focus more on fault detection)
        class_weight = torch.ones_like(bce_loss)
        pos_mask = fault_label > 0.5
        class_weight[pos_mask] = self.pos_weight
        
        # Combine focal weight and class weight
        weighted_loss = focal_weight * class_weight * bce_loss
        detection_loss = weighted_loss.mean()
        
        # NEW MULTI-SENSOR ANOMALY LOSS APPROACH
        faulty_mask = fault_label > 0.5
        
        # Initialize anomaly loss
        anomaly_loss = torch.tensor(0.0).to(fault_detection.device)
        
        # For healthy samples, all sensors should show low anomaly scores
        if torch.any(~faulty_mask):
            healthy_samples = sensor_anomalies[~faulty_mask]
            # Allow small anomalies even in healthy state but keep them low
            healthy_target = torch.zeros_like(healthy_samples) + 0.1
            healthy_loss = F.binary_cross_entropy(healthy_samples, healthy_target)
            anomaly_loss += healthy_loss
        
        # For faulty samples, encourage multiple sensors to show high anomaly scores
        if torch.any(faulty_mask):
            faulty_samples = sensor_anomalies[faulty_mask]
            
            # MULTI-SENSOR APPROACH: Encourage top-k sensors to have high anomaly scores
            # Sort anomaly scores in descending order
            sorted_anomalies, _ = torch.sort(faulty_samples, dim=1, descending=True)
            
            # Target for top k sensors (use at least min_faulty_sensors or 30% of sensors)
            k = max(self.min_faulty_sensors, int(self.num_sensors * 0.3))
            
            # Targets for top-k sensors: high anomaly scores (0.8-0.9)
            top_k_targets = torch.ones_like(sorted_anomalies[:, :k]) * 0.9
            # Use higher weight for top-k sensor loss to emphasize multi-sensor detection
            top_k_loss = F.binary_cross_entropy(sorted_anomalies[:, :k], top_k_targets) * 2.0
            
            # Targets for remaining sensors: varied anomaly scores (0.2-0.4)
            # This allows some sensors to show moderate anomalies, more realistic
            remaining_targets = torch.ones_like(sorted_anomalies[:, k:]) * 0.3
            remaining_loss = F.binary_cross_entropy(sorted_anomalies[:, k:], remaining_targets)
            
            # Penalize uniform anomaly scores to avoid all sensors showing same value
            variance_penalty = -torch.var(faulty_samples, dim=1).mean() * 0.2
            
            # Combined faulty loss - increase weight of top_k_loss for better multi-sensor detection
            faulty_loss = top_k_loss + 0.5 * remaining_loss + variance_penalty
            anomaly_loss += faulty_loss
        
        # Combined loss with increased weight on anomaly_loss
        total_loss = detection_loss + self.alpha * anomaly_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'anomaly_loss': anomaly_loss
        }

def create_multi_sensor_damage_patterns(batch_data, batch_labels, num_sensors=8):
    """
    Create synthetic examples with multiple damaged sensors
    
    Args:
        batch_data: Tensor of shape [batch_size, time_steps, channels]
        batch_labels: Tensor of shape [batch_size]
        num_sensors: Number of sensors in the data
        
    Returns:
        Augmented data and labels with multi-sensor damage patterns
    """
    augmented_data = []
    augmented_labels = []
    
    # Define realistic damage patterns from domain knowledge
    # Each list contains indices of sensors that would be damaged together
    damage_patterns = [
        [0, 1],          # AN3, AN4 (Ring gear)
        [3, 6],          # AN6, AN9 (IMS and downwind bearing)
        [5, 6],          # AN8, AN9 (Both HS-SH bearings)
        [2, 7],          # AN5, AN10 (LS and carrier)
        [0, 1, 3],       # AN3, AN4, AN6 (Ring gear + IMS)
        [3, 4, 5, 6]     # AN6, AN7, AN8, AN9 (Complete high-speed section)
    ]
    
    batch_size = batch_data.size(0)
    time_steps = batch_data.size(1)
    
    for i in range(batch_size):
        # Add the original sample
        augmented_data.append(batch_data[i])
        augmented_labels.append(batch_labels[i])
        
        # Only augment damaged samples (skip healthy ones)
        if batch_labels[i] > 0.5:
            # Add 2 random damage patterns
            for _ in range(2):
                # Select a random damage pattern
                pattern = damage_patterns[torch.randint(0, len(damage_patterns), (1,)).item()]
                
                # Create a copy of the original sample
                new_sample = batch_data[i].clone()
                
                # Generate correlated noise pattern (more realistic than random noise)
                # First create a base pattern
                base_noise = torch.randn(time_steps, 1) * 0.3  # Temporal correlation
                
                # Apply correlated noise to all sensors in the pattern with small variations
                for sensor_idx in pattern:
                    # Add small per-sensor variation to the base noise
                    sensor_noise = base_noise + torch.randn(time_steps, 1) * 0.1
                    # Apply the noise to the sensor data
                    new_sample[:, sensor_idx] += sensor_noise.squeeze()
                
                augmented_data.append(new_sample)
                augmented_labels.append(batch_labels[i])  # Same label as original
    
    return torch.stack(augmented_data), torch.stack(augmented_labels)

def train_epoch(model, train_loader, optimizer, criterion=None, device=None, use_augmentation=True):
    """Train for one epoch with improved multi-sensor detection"""
    if criterion is None:
        criterion = FaultLocalizationLoss()
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss = 0
    detection_losses = 0
    anomaly_losses = 0
    
    # Track predictions and targets for metrics
    all_outputs = []
    all_targets = []
    
    batch_count = 0
    for data, labels, locations in train_loader:
        batch_count += 1
        
        # Move to device
        data = data.to(device)
        labels = labels.to(device)
        locations = locations.to(device)
        
        # Create targets dict for updated loss function
        targets = {'fault_label': labels, 'fault_location': locations}
        
        # Apply data augmentation for multi-sensor patterns (only to faulty samples)
        if use_augmentation:
            # Identify faulty samples
            faulty_idx = torch.where(labels > 0.5)[0]
            if len(faulty_idx) > 0:
                augmented_data = data.clone()
                
                # For each faulty sample, apply random variations to multiple sensors
                for idx in faulty_idx:
                    # Randomly select 2-3 sensors to make more anomalous
                    num_sensors_to_modify = min(np.random.randint(2, 4), data.shape[2]-1)  # -1 for RPM
                    sensors_to_modify = np.random.choice(range(data.shape[2]-1), num_sensors_to_modify, replace=False)
                    
                    # Apply random modifications to those sensors
                    for sensor in sensors_to_modify:
                        # Choose a random modification technique:
                        mod_type = np.random.choice(['noise', 'amplitude', 'phase'])
                        
                        if mod_type == 'noise':
                            # Add noise
                            noise_level = np.random.uniform(0.05, 0.15)
                            augmented_data[idx, :, sensor] += torch.randn_like(data[idx, :, sensor]) * noise_level
                        
                        elif mod_type == 'amplitude':
                            # Increase amplitude
                            amp_factor = np.random.uniform(1.1, 1.3)
                            augmented_data[idx, :, sensor] *= amp_factor
                        
                        elif mod_type == 'phase':
                            # Add phase shift by rolling the signal
                            shift = np.random.randint(1, 5)
                            augmented_data[idx, :, sensor] = torch.roll(data[idx, :, sensor], shifts=shift)
                
                # Mix original and augmented batches with a weight
                mix_ratio = 0.7  # 70% original, 30% augmented
                data = data * mix_ratio + augmented_data * (1.0 - mix_ratio)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Use improved loss function
        losses = criterion(outputs, targets)
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        detection_losses += losses['detection_loss'].item()
        anomaly_losses += losses['anomaly_loss'].item()
        
        # Store outputs and targets for metrics calculation
        all_outputs.append(outputs['fault_detection'].detach().cpu())
        all_targets.append(labels.detach().cpu())
    
    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        detection_auc = roc_auc_score(all_targets, all_outputs)
    except:
        detection_auc = 0.5
    
    # Use dynamic thresholding for classification metrics
    # Find the threshold that gives at least MIN_RECALL recall
    try:
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(all_targets, all_outputs)
        
        # Find threshold that gives at least MIN_RECALL
        valid_idx = np.where(recall >= MIN_RECALL)[0]
        if len(valid_idx) > 0:
            # Choose the threshold that gives the highest precision among those with sufficient recall
            best_idx = valid_idx[np.argmax(precision[valid_idx])]
            threshold = thresholds[best_idx]
        else:
            # If no threshold gives sufficient recall, choose the one with highest recall
            threshold = thresholds[np.argmax(recall)]
    except:
        threshold = DEFAULT_THRESHOLD
    
    # Calculate metrics with dynamic threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    binary_preds = (all_outputs >= threshold).astype(float)
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    
    metrics = {
        'total_loss': total_loss / batch_count,
        'detection_loss': detection_losses / batch_count,
        'anomaly_loss': anomaly_losses / batch_count,
        'detection_auc': detection_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold
    }
    
    return metrics

def evaluate_multi_sensor_detection(model, val_loader, device=None):
    """
    Evaluate the model's ability to detect multiple damaged sensors
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with multi-sensor detection metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Track sensor-specific metrics
    all_sensor_anomalies = []
    all_fault_labels = []
    
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data = data.to(device)
            outputs = model(data)
            
            # Collect anomaly scores
            all_sensor_anomalies.append(outputs['sensor_anomalies'].cpu().numpy())
            all_fault_labels.append(labels.cpu().numpy())
    
    # Concatenate results
    sensor_anomalies = np.concatenate(all_sensor_anomalies)
    fault_labels = np.concatenate(all_fault_labels)
    
    # Calculate sensor-wise detection AUC
    sensor_metrics = {}
    for i in range(sensor_anomalies.shape[1]):
        sensor_name = f'AN{i+3}'
        sensor_anomaly = sensor_anomalies[:, i]
        
        # Calculate AUC for this sensor
        try:
            sensor_auc = roc_auc_score(fault_labels, sensor_anomaly)
        except:
            sensor_auc = 0.5  # Default if calculation fails
        
        # Calculate average anomaly score for faulty vs healthy
        if np.any(fault_labels > 0.5) and np.any(fault_labels < 0.5):
            faulty_mean = sensor_anomaly[fault_labels > 0.5].mean()
            healthy_mean = sensor_anomaly[fault_labels < 0.5].mean()
            separation = faulty_mean - healthy_mean
        else:
            separation = 0
        
        sensor_metrics[sensor_name] = {
            'auc': sensor_auc,
            'separation': separation,
            'faulty_anomaly_mean': sensor_anomaly[fault_labels > 0.5].mean() if np.any(fault_labels > 0.5) else 0,
            'healthy_anomaly_mean': sensor_anomaly[fault_labels < 0.5].mean() if np.any(fault_labels < 0.5) else 0
        }
    
    # Analyze multi-sensor patterns
    # Count how many sensors show high anomalies in faulty samples
    faulty_samples = sensor_anomalies[fault_labels > 0.5]
    if len(faulty_samples) > 0:
        high_anomaly_count = (faulty_samples > 0.5).sum(axis=1).mean()
    else:
        high_anomaly_count = 0
    
    multi_sensor_metrics = {
        'avg_high_anomaly_sensors': high_anomaly_count,
        'sensor_metrics': sensor_metrics
    }
    
    return multi_sensor_metrics

def evaluate(model, val_loader, criterion=None, device=None):
    if criterion is None:
        criterion = FaultLocalizationLoss()
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0
    detection_losses = 0
    anomaly_losses = 0
    
    all_fault_probs = []
    all_fault_labels = []
    all_sensor_anomalies = []
    
    with torch.no_grad():
        for data, labels, locations in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            locations = locations.to(device)
            
            targets = {'fault_label': labels, 'fault_location': locations}
            
            outputs = model(data)
            losses = criterion(outputs, targets)
            
            # Store losses
            total_loss += losses['total_loss'].item()
            detection_losses += losses['detection_loss'].item()
            anomaly_losses += losses['anomaly_loss'].item()
            
            # Apply sigmoid to get probabilities
            fault_probs = torch.sigmoid(outputs['fault_detection']).squeeze().cpu().numpy()
            
            # Collect predictions and labels
            all_fault_probs.extend(fault_probs)
            all_fault_labels.extend(labels.cpu().numpy())
            all_sensor_anomalies.append(outputs['sensor_anomalies'].cpu().numpy())
    
    # Convert to numpy arrays
    all_fault_probs = np.array(all_fault_probs)
    all_fault_labels = np.array(all_fault_labels)
    all_sensor_anomalies = np.vstack(all_sensor_anomalies)
    
    # Calculate AUC
    try:
        detection_auc = roc_auc_score(all_fault_labels, all_fault_probs)
    except:
        detection_auc = 0.5
    
    # Use dynamic thresholding to find best F1 score
    # But ensure minimum recall is maintained (from config)
    try:
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_fault_labels, all_fault_probs)
        
        # Find thresholds that meet minimum recall requirement
        valid_idx = np.where(recall >= MIN_RECALL)[0]
        
        if len(valid_idx) > 0:
            # From valid thresholds, find one that maximizes F1
            f1_scores = 2 * precision[valid_idx] * recall[valid_idx] / (precision[valid_idx] + recall[valid_idx] + 1e-7)
            best_idx = valid_idx[np.argmax(f1_scores)]
            best_threshold = thresholds[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
            best_f1 = f1_scores[np.argmax(f1_scores)]
        else:
            # If no threshold meets recall requirement, use DEFAULT_THRESHOLD
            predictions = (all_fault_probs >= DEFAULT_THRESHOLD).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            best_threshold = DEFAULT_THRESHOLD
            best_precision = precision_score(all_fault_labels, predictions, zero_division=0)
            best_recall = recall_score(all_fault_labels, predictions, zero_division=0)
            best_f1 = f1_score(all_fault_labels, predictions, zero_division=0)
            
            print(f"Warning: No threshold found meeting minimum recall of {MIN_RECALL}.")
            print(f"Using default threshold {DEFAULT_THRESHOLD:.2f} with recall {best_recall:.4f}")
    except Exception as e:
        print(f"Error in threshold calculation: {e}")
        best_threshold = DEFAULT_THRESHOLD
        predictions = (all_fault_probs >= best_threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        best_precision = precision_score(all_fault_labels, predictions, zero_division=0)
        best_recall = recall_score(all_fault_labels, predictions, zero_division=0)
        best_f1 = f1_score(all_fault_labels, predictions, zero_division=0)
    
    # Calculate sensor-wise metrics
    sensor_names = [f'AN{i}' for i in range(3, 11)]
    sensor_metrics = {}
    
    for i, sensor_name in enumerate(sensor_names):
        sensor_anomaly = all_sensor_anomalies[:, i]
        
        # Calculate AUC for this sensor
        try:
            sensor_auc = roc_auc_score(all_fault_labels, sensor_anomaly)
        except:
            sensor_auc = 0.5  # Default if calculation fails
        
        # Calculate average anomaly score for faulty vs healthy
        if np.any(all_fault_labels > 0.5) and np.any(all_fault_labels < 0.5):
            faulty_mean = sensor_anomaly[all_fault_labels > 0.5].mean()
            healthy_mean = sensor_anomaly[all_fault_labels < 0.5].mean()
            separation = faulty_mean - healthy_mean
        else:
            separation = 0
        
        sensor_metrics[sensor_name] = {
            'anomaly_auc': sensor_auc,
            'mean_anomaly': sensor_anomaly.mean(),
            'separation': separation
        }
    
    # Get multi-sensor detection metrics
    multi_sensor_metrics = evaluate_multi_sensor_detection(model, val_loader, device)
    
    num_batches = len(val_loader)
    metrics = {
        'total_loss': total_loss / num_batches,
        'detection_loss': detection_losses / num_batches,
        'anomaly_loss': anomaly_losses / num_batches,
        'detection_auc': detection_auc,
        'precision': best_precision,
        'recall': best_recall,
        'f1_score': best_f1,
        'threshold': best_threshold,
        'sensor_metrics': sensor_metrics,
        'multi_sensor_metrics': multi_sensor_metrics,
        'val_loss': total_loss / num_batches  # Add this for compatibility
    }
    
    # Print sensor-wise metrics
    print("\nSensor Anomaly Details:")
    for sensor, sensor_data in sensor_metrics.items():
        print(f"{sensor}: Mean Anomaly = {sensor_data['mean_anomaly']:.4f}, AUC = {sensor_data['anomaly_auc']:.4f}, Separation = {sensor_data['separation']:.4f}")
    
    # Print multi-sensor metrics
    print(f"\nMulti-sensor Detection: Avg high-anomaly sensors in faulty samples = {multi_sensor_metrics['avg_high_anomaly_sensors']:.2f}")
    
    # Print classification metrics
    print(f"\nClassification Metrics (threshold={best_threshold:.2f}):")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    
    return metrics

def train(model, train_loader, val_loader, config):
    device = torch.device(config.device)
    model = model.to(device)
    
    # Use updated loss function with multi-sensor focus
    criterion = FaultLocalizationLoss(
        alpha=config.location_loss_weight,
        num_sensors=8,  # Number of sensors in the data
        min_faulty_sensors=2,  # Minimum number of sensors expected to show anomalies
        focal_gamma=config.focal_gamma,
        pos_weight=config.pos_weight
    )
    
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
    best_multi_sensor_score = 0
    
    for epoch in range(config.num_epochs):
        # Train with multi-sensor damage augmentation
        train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            use_augmentation=True  # Enable augmentation
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model - now consider both validation loss and multi-sensor metrics
        multi_sensor_score = val_metrics['multi_sensor_metrics']['avg_high_anomaly_sensors']
        
        # Save model if it has better validation loss OR if it maintains good loss but has better multi-sensor detection
        if val_metrics['total_loss'] < best_val_loss or (
            val_metrics['total_loss'] < best_val_loss * 1.05 and multi_sensor_score > best_multi_sensor_score
        ):
            best_val_loss = min(best_val_loss, val_metrics['total_loss'])
            best_multi_sensor_score = max(best_multi_sensor_score, multi_sensor_score)
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"Detection AUC: {val_metrics['detection_auc']:.4f}")
        print(f"Multi-sensor detection score: {multi_sensor_score:.2f}")
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model 