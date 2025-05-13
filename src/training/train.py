import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from config.config import DEFAULT_THRESHOLD, MIN_RECALL, LOCATION_LOSS_WEIGHT, FOCAL_GAMMA, POS_WEIGHT

class FaultLocalizationLoss(nn.Module):
    def __init__(self, alpha=LOCATION_LOSS_WEIGHT, label_smoothing=0.05, 
                 focal_gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.bce_detection = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_anomaly = nn.BCEWithLogitsLoss(reduction='mean')
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
    
    def forward(self, outputs, targets):
        fault_detection_logits = outputs['fault_detection'].squeeze()
        sensor_anomalies_pred = outputs['sensor_anomalies']
        
        fault_label_binary = targets['fault_label']
        sensor_locations_gt = targets['fault_location']
        
        smoothed_targets_binary = fault_label_binary.clone().float()
        smoothed_targets_binary = smoothed_targets_binary * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce_loss_detection = self.bce_detection(fault_detection_logits, smoothed_targets_binary)
        
        pt_detection = torch.exp(-bce_loss_detection)
        focal_weight_detection = (1 - pt_detection) ** self.focal_gamma
        
        class_weight_detection = torch.ones_like(bce_loss_detection)
        pos_mask_detection = fault_label_binary > 0.5
        class_weight_detection[pos_mask_detection] = self.pos_weight
        
        weighted_detection_loss = focal_weight_detection * class_weight_detection * bce_loss_detection
        detection_loss = weighted_detection_loss.mean()
        
        anomaly_loss = self.bce_anomaly(sensor_anomalies_pred, sensor_locations_gt)
        
        total_loss = detection_loss + self.alpha * anomaly_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'anomaly_loss': anomaly_loss
        }

def create_multi_sensor_damage_patterns(batch_data, batch_labels, batch_locations, num_sensors=8):
    augmented_data_list = []
    augmented_labels_list = []
    augmented_locations_list = []
    
    damage_patterns = [
        [0, 1],          # AN3, AN4 (Ring gear)
        [3, 6],          # AN6, AN9 (IMS and downwind bearing)
        [5, 6],          # AN8, AN9
        [2, 7],          # AN5, AN10
        [0, 1, 3],       # AN3, AN4, AN6
        [3, 4, 5, 6]     # AN6, AN7, AN8, AN9
    ]
    
    batch_size = batch_data.size(0)
    time_steps = batch_data.size(1)
    
    for i in range(batch_size):
        augmented_data_list.append(batch_data[i])
        augmented_labels_list.append(batch_labels[i])
        augmented_locations_list.append(batch_locations[i])
        
        if batch_labels[i] > 0.5:
            num_augmentations = np.random.randint(1, 3)
            for _ in range(num_augmentations):
                pattern_idx = torch.randint(0, len(damage_patterns), (1,)).item()
                pattern = damage_patterns[pattern_idx]
                
                new_sample_data = batch_data[i].clone()
                new_sample_locations = batch_locations[i].clone()
                
                base_noise_amplitude = np.random.uniform(0.1, 0.4)
                base_noise = torch.randn(time_steps, 1) * base_noise_amplitude 
                
                for sensor_idx_in_pattern in pattern:
                    if 0 <= sensor_idx_in_pattern < num_sensors:
                        sensor_noise = base_noise + torch.randn(time_steps, 1) * (base_noise_amplitude / 2)
                        new_sample_data[:, sensor_idx_in_pattern] += sensor_noise.squeeze().to(new_sample_data.device)
                        
                        current_anomaly = new_sample_locations[sensor_idx_in_pattern].item()
                        augmentation_strength = np.random.uniform(0.3, 0.6)
                        new_anomaly_score = min(current_anomaly + augmentation_strength, 1.0)
                        new_sample_locations[sensor_idx_in_pattern] = new_anomaly_score
                    else:
                        print(f"Warning: Sensor index {sensor_idx_in_pattern} in pattern is out of bounds for {num_sensors} sensors.")

                augmented_data_list.append(new_sample_data)
                augmented_labels_list.append(batch_labels[i])
                augmented_locations_list.append(new_sample_locations)
    
    return torch.stack(augmented_data_list), torch.stack(augmented_labels_list), torch.stack(augmented_locations_list)

def train_epoch(model, train_loader, optimizer, criterion=None, device=None, use_augmentation=True, num_sensors=8, gradient_clip_val=1.0):
    if criterion is None:
        criterion = FaultLocalizationLoss(alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT)
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss_epoch = 0
    detection_loss_epoch = 0
    anomaly_loss_epoch = 0
    
    all_fault_predictions_logits = []
    all_fault_true_labels = []
    
    batch_count = 0
    for data, labels_binary, locations_gt in train_loader:
        batch_count += 1
        
        data = data.to(device)
        labels_binary = labels_binary.to(device)
        locations_gt = locations_gt.to(device)
        
        if use_augmentation:
            data, labels_binary, locations_gt = create_multi_sensor_damage_patterns(
                data, labels_binary, locations_gt, num_sensors=num_sensors
            )
        
        targets = {'fault_label': labels_binary, 'fault_location': locations_gt}
        
        optimizer.zero_grad()
        outputs = model(data)
        
        losses = criterion(outputs, targets)
        loss = losses['total_loss']
        
        loss.backward()
        
        if gradient_clip_val is not None and gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
            
        optimizer.step()
        
        total_loss_epoch += loss.item()
        detection_loss_epoch += losses['detection_loss'].item()
        anomaly_loss_epoch += losses['anomaly_loss'].item()
        
        all_fault_predictions_logits.append(outputs['fault_detection'].detach().cpu())
        all_fault_true_labels.append(labels_binary.detach().cpu())
    
    all_fault_predictions_logits = torch.cat(all_fault_predictions_logits).numpy()
    all_fault_true_labels = torch.cat(all_fault_true_labels).numpy()
    
    try:
        detection_auc = roc_auc_score(all_fault_true_labels, all_fault_predictions_logits)
    except ValueError:
        detection_auc = 0.5
    
    all_fault_probs = 1 / (1 + np.exp(-all_fault_predictions_logits))
    try:
        precision_vals, recall_vals, threshold_options = precision_recall_curve(all_fault_true_labels, all_fault_probs)
        
        if len(threshold_options) == len(precision_vals):
            thresholds = threshold_options
        else:
            thresholds = threshold_options
            precision_vals = precision_vals[:-1]
            recall_vals = recall_vals[:-1]

        valid_indices = np.where(recall_vals >= MIN_RECALL)[0]
        
        if len(valid_indices) > 0:
            f1_scores = 2 * (precision_vals[valid_indices] * recall_vals[valid_indices]) / \
                        (precision_vals[valid_indices] + recall_vals[valid_indices] + 1e-8)
            
            best_f1_idx_within_valid = np.argmax(f1_scores)
            best_threshold_idx = valid_indices[best_f1_idx_within_valid]
            
            if best_threshold_idx < len(thresholds):
                chosen_threshold = thresholds[best_threshold_idx]
            else:
                chosen_threshold = DEFAULT_THRESHOLD
                print("Warning: Could not determine best threshold based on F1, using default.")

        else:
            max_recall_idx = np.argmax(recall_vals)
            if max_recall_idx < len(thresholds):
                 chosen_threshold = thresholds[max_recall_idx]
            else:
                 chosen_threshold = DEFAULT_THRESHOLD
                 print("Warning: Could not determine threshold for max recall, using default.")

    except Exception as e:
        print(f"Warning: Error during threshold calculation in train_epoch: {e}. Using default threshold.")
        chosen_threshold = DEFAULT_THRESHOLD
    
    binary_preds_for_metric = (all_fault_probs >= chosen_threshold).astype(float)
    precision_metric, recall_metric, f1_metric, _ = precision_recall_fscore_support(
        all_fault_true_labels, binary_preds_for_metric, average='binary', zero_division=0
    )
    
    metrics = {
        'total_loss': total_loss_epoch / batch_count if batch_count > 0 else 0,
        'detection_loss': detection_loss_epoch / batch_count if batch_count > 0 else 0,
        'anomaly_loss': anomaly_loss_epoch / batch_count if batch_count > 0 else 0,
        'detection_auc': detection_auc,
        'precision': precision_metric,
        'recall': recall_metric,
        'f1_score': f1_metric,
        'threshold': chosen_threshold
    }
    
    return metrics

def evaluate_multi_sensor_detection(model, val_loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    all_sensor_anomalies = []
    all_fault_labels = []
    
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data = data.to(device)
            outputs = model(data)
            
            all_sensor_anomalies.append(outputs['sensor_anomalies'].cpu().numpy())
            all_fault_labels.append(labels.cpu().numpy())
    
    sensor_anomalies = np.concatenate(all_sensor_anomalies)
    fault_labels = np.concatenate(all_fault_labels)
    
    sensor_metrics = {}
    for i in range(sensor_anomalies.shape[1]):
        sensor_name = f'AN{i+3}'
        sensor_anomaly = sensor_anomalies[:, i]
        
        try:
            sensor_auc = roc_auc_score(fault_labels, sensor_anomaly)
        except:
            sensor_auc = 0.5
        
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

def evaluate(model, val_loader, criterion=None, device=None, num_sensors=8):
    if criterion is None:
        criterion = FaultLocalizationLoss(alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT)
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss_epoch = 0
    detection_loss_epoch = 0
    anomaly_loss_epoch = 0
    
    all_fault_logits_list = []
    all_fault_true_labels_list = []
    all_sensor_anomalies_pred_list = []
    all_sensor_locations_gt_list = []
    
    with torch.no_grad():
        for data, labels_binary, locations_gt in val_loader:
            data = data.to(device)
            labels_binary = labels_binary.to(device)
            locations_gt = locations_gt.to(device)
            
            targets = {'fault_label': labels_binary, 'fault_location': locations_gt}
            
            outputs = model(data)
            losses = criterion(outputs, targets)
            
            total_loss_epoch += losses['total_loss'].item()
            detection_loss_epoch += losses['detection_loss'].item()
            anomaly_loss_epoch += losses['anomaly_loss'].item()
            
            all_fault_logits_list.append(outputs['fault_detection'].squeeze().cpu())
            all_fault_true_labels_list.append(labels_binary.cpu())
            all_sensor_anomalies_pred_list.append(outputs['sensor_anomalies'].cpu())
            all_sensor_locations_gt_list.append(locations_gt.cpu())
    
    all_fault_logits = torch.cat(all_fault_logits_list).numpy()
    all_fault_true_labels = torch.cat(all_fault_true_labels_list).numpy()
    all_sensor_anomalies_pred = torch.cat(all_sensor_anomalies_pred_list).numpy()
    all_sensor_locations_gt = torch.cat(all_sensor_locations_gt_list).numpy()

    try:
        detection_auc = roc_auc_score(all_fault_true_labels, all_fault_logits)
    except ValueError:
        detection_auc = 0.5
    
    all_fault_probs = 1 / (1 + np.exp(-all_fault_logits))
    try:
        precision_vals, recall_vals, threshold_options = precision_recall_curve(all_fault_true_labels, all_fault_probs)
        valid_indices = np.where(recall_vals >= MIN_RECALL)[0]
        if len(valid_indices) > 0:
            best_threshold_idx = valid_indices[np.argmax(precision_vals[valid_indices])]
            chosen_threshold = threshold_options[best_threshold_idx]
        else:
            chosen_threshold = threshold_options[np.argmax(recall_vals)]
    except Exception:
        chosen_threshold = DEFAULT_THRESHOLD

    binary_preds_for_metric = (all_fault_probs >= chosen_threshold).astype(int)
    precision_metric, recall_metric, f1_metric, _ = precision_recall_fscore_support(
        all_fault_true_labels, binary_preds_for_metric, average='binary', zero_division=0
    )

    sensor_anomaly_mse = np.mean((all_sensor_anomalies_pred - all_sensor_locations_gt)**2)
    per_sensor_mse = np.mean((all_sensor_anomalies_pred - all_sensor_locations_gt)**2, axis=0)
    sensor_metrics_detailed = {f'AN{i+3}_mse': mse_val for i, mse_val in enumerate(per_sensor_mse)}

    multi_sensor_eval_metrics = evaluate_multi_sensor_detection(model, val_loader, device)
    
    num_batches = len(val_loader)
    metrics = {
        'total_loss': total_loss_epoch / num_batches,
        'detection_loss': detection_loss_epoch / num_batches,
        'anomaly_loss': anomaly_loss_epoch / num_batches,
        'detection_auc': detection_auc,
        'precision': precision_metric,
        'recall': recall_metric,
        'f1_score': f1_metric,
        'threshold': chosen_threshold,
        'sensor_anomaly_mse': sensor_anomaly_mse,
        'per_sensor_mse': sensor_metrics_detailed,
        'multi_sensor_metrics_original_eval': multi_sensor_eval_metrics,
        'val_loss': total_loss_epoch / num_batches 
    }
    
    print_evaluation_summary(metrics)
    return metrics

def print_evaluation_summary(metrics):
    print(f"\n--- Evaluation Summary ---")
    print(f"Overall Loss: {metrics['total_loss']:.4f} (Detection: {metrics['detection_loss']:.4f}, Anomaly: {metrics['anomaly_loss']:.4f})")
    print(f"Detection AUC: {metrics['detection_auc']:.4f}")
    print(f"Detection Metrics (Threshold: {metrics['threshold']:.3f}):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1_score']:.4f}")
    print(f"Sensor Anomaly MSE (vs locations_gt): {metrics['sensor_anomaly_mse']:.4f}")
    if 'multi_sensor_metrics_original_eval' in metrics and metrics['multi_sensor_metrics_original_eval']:
        print(f"Multi-sensor (Original Eval): Avg high-anomaly sensors in faulty samples = {metrics['multi_sensor_metrics_original_eval']['avg_high_anomaly_sensors']:.2f}")
    print("----------------------------")

def train(model, train_loader, val_loader, config):
    device = torch.device(config.device if hasattr(config, 'device') else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    
    num_sensors = 8
    gradient_clip_val = getattr(config, 'gradient_clip_val', 1.0)
    early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
    metric_to_monitor = getattr(config, 'metric_for_best_model', 'f1_score')
    monitor_mode = 'max' if metric_to_monitor != 'val_loss' else 'min'

    criterion = FaultLocalizationLoss(
        alpha=getattr(config, 'LOCATION_LOSS_WEIGHT', 1.0),
        focal_gamma=getattr(config, 'FOCAL_GAMMA', FOCAL_GAMMA),
        pos_weight=getattr(config, 'POS_WEIGHT', POS_WEIGHT),
        label_smoothing=getattr(config, 'label_smoothing', 0.05)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=getattr(config, 'learning_rate', getattr(config, 'LR', 0.001)),
        weight_decay=getattr(config, 'weight_decay', getattr(config, 'WEIGHT_DECAY', 1e-4))
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=getattr(config, 'scheduler_patience', 5)
    )
    
    best_metric_val = float('-inf') if monitor_mode == 'max' else float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    num_epochs = getattr(config, 'num_epochs', getattr(config, 'EPOCHS', 20))

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Monitoring validation '{metric_to_monitor}' ('{monitor_mode}' mode) for best model.")
    print(f"Early stopping patience: {early_stopping_patience} epochs.")
    print(f"Gradient clipping max_norm: {gradient_clip_val}")
    print(f"Loss alpha (LOCATION_LOSS_WEIGHT fallback to 1.0 if not in config): {criterion.alpha}")

    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            use_augmentation=config.use_augmentation if hasattr(config, 'use_augmentation') else True,
            num_sensors=num_sensors,
            gradient_clip_val=gradient_clip_val
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device, num_sensors=num_sensors)
        
        scheduler.step(val_metrics['val_loss'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        current_val_metric = val_metrics.get(metric_to_monitor)
        if current_val_metric is None:
            print(f"Warning: Metric '{metric_to_monitor}' not found in validation metrics. Using 'val_loss' for early stopping.")
            metric_to_monitor = 'val_loss'
            monitor_mode = 'min'
            current_val_metric = val_metrics['val_loss']
            best_metric_val = float('inf')
        
        print(f"--- Epoch {epoch+1}/{num_epochs} Completed --- LR: {current_lr:.6f} ---")
        print(f"Train Metrics: Loss={train_metrics['total_loss']:.4f}, AUC={train_metrics['detection_auc']:.4f}, F1={train_metrics['f1_score']:.4f}")
        print_evaluation_summary(val_metrics)
        
        improved = False
        if monitor_mode == 'max':
            if current_val_metric > best_metric_val:
                best_metric_val = current_val_metric
                improved = True
        else:
            if current_val_metric < best_metric_val:
                best_metric_val = current_val_metric
                improved = True
                
        if improved:
            print(f"Epoch {epoch+1}: Validation {metric_to_monitor} improved to {best_metric_val:.4f}. Saving model...")
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: Validation {metric_to_monitor} did not improve ({current_val_metric:.4f}). ({epochs_no_improve}/{early_stopping_patience})")

        print("----------------------------------------------------")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in validation {metric_to_monitor} for {early_stopping_patience} epochs.")
            break
            
    if best_model_state:
        print(f"Loading best model state with validation {metric_to_monitor}: {best_metric_val:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state was saved during training. Returning the model from the last epoch.")
        
    return model 
