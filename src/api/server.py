import torch
from src.models.model import GearboxCNNLSTM as GearboxCNN
from config.config import *
import numpy as np
import os
from src.utils.utils import calculate_metrics
from sklearn.metrics import f1_score, accuracy_score
from src.training.train import train_epoch, evaluate  # Updated import
from src.api.client import FederatedClient
import torch.nn.functional as F

class FederatedServer:
    def __init__(self, num_clients=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_clients = num_clients
        self.clients = []
        
        # Initialize or load global model
        save_dir = os.path.join(BASE_DIR, 'models')
        model_path = os.path.join(save_dir, 'global_model.pt')
        
        self.global_model = GearboxCNN().to(self.device)
        if os.path.exists(model_path):
            print("Loading existing global model...")
            self.global_model.load_state_dict(torch.load(model_path))
        else:
            print("Initializing new global model...")
            os.makedirs(save_dir, exist_ok=True)
            self.save_model()  # Save initial model
        
        # Initialize clients
        for i in range(num_clients):
            client = FederatedClient(f"Client_{i+1}")
            self.clients.append(client)
        
        print("Server initialized")
        print(f"Initialized {num_clients} clients\n")
    
    def aggregate_models(self, client_models, client_metrics):
        """Aggregate client models using weighted FedAvg based on performance metrics with enhanced parameter-specific aggregation"""
        print(f"\n[Server aggregating {len(client_models)} client models]")
        
        # Compute weights based on client metrics with enhanced specialization awareness
        weights = []
        for i, metrics in enumerate(client_metrics):
            # Extract key metrics: detection AUC, data quality, validation loss, and sensor metrics
            auc = metrics.get('detection_auc', 0.5)  # Default to 0.5 if not available
            data_quality = metrics.get('data_quality', 0.5)  # Default to 0.5 if not available
            val_loss = metrics.get('val_loss', 1.0)  # Default to 1.0 if not available
            
            # Consider sensor metrics if available (reward models that detect multiple sensors)
            sensor_reward = 0.0
            if 'multi_sensor_metrics' in metrics and 'avg_high_anomaly_sensors' in metrics['multi_sensor_metrics']:
                avg_high_sensors = metrics['multi_sensor_metrics']['avg_high_anomaly_sensors']
                # Normalize: reward models that detect more sensors on average when fault occurs
                sensor_reward = min(1.0, avg_high_sensors / 4.0)  # Cap at 1.0
            
            # Calculate weight based on metrics (normalize loss to be higher for lower loss)
            loss_factor = 1.0 / (val_loss + 0.1)  # Add small epsilon to avoid division by zero
            
            # Get client specializations if available
            specialization_bonus = 0.0
            if 'specialized_damage_types' in metrics:
                # Add bonus for specialized clients
                specialization_bonus = 0.1 * len(metrics['specialized_damage_types'])
            
            # Compute combined weight with enhanced components
            weight = (
                (auc * 0.4) + 
                (data_quality * 0.2) + 
                (loss_factor * 0.15) + 
                (sensor_reward * 0.15) +
                specialization_bonus
            )
            weights.append(weight)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            # Equal weights if all weights are zero
            normalized_weights = [1.0 / len(weights) for _ in weights]
        
        print(f"Client weights: {[f'{w:.3f}' for w in normalized_weights]}")
        
        # Memory retention of previous global model (prevents catastrophic forgetting)
        # Use adaptive memory retention based on client diversity
        client_diversity = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
        memory_retention = min(0.3, max(0.1, 0.2 - client_diversity))  # Adaptive between 0.1-0.3
        print(f"Using memory retention of {memory_retention:.3f} (based on client diversity)")
        
        # Layer-specific aggregation parameters - different types of layers need different strategies
        layer_configs = {
            # Feature extraction layers - more global, less personalized
            'sensor_cnns': {'memory': memory_retention, 'avg_factor': 1.0},
            'freq_extractors': {'memory': memory_retention, 'avg_factor': 1.0},
            'lstm': {'memory': memory_retention, 'avg_factor': 1.0},
            'rpm_cnn': {'memory': memory_retention, 'avg_factor': 1.0},
            
            # Relationship modeling - blend of global and personalized
            'sensor_relationship_matrix': {'memory': memory_retention * 0.7, 'avg_factor': 0.9},
            'cross_sensor_interaction': {'memory': memory_retention * 0.8, 'avg_factor': 0.95},
            
            # Anomaly detection - more personalized
            'joint_anomaly_head': {'memory': memory_retention * 1.2, 'avg_factor': 0.8},
            'anomaly_calibration': {'memory': memory_retention * 1.3, 'avg_factor': 0.7},
            'sensor_anomaly_heads': {'memory': memory_retention * 1.3, 'avg_factor': 0.7},
            
            # Fault detection - blend of global and personalized
            'fault_detector': {'memory': memory_retention * 1.1, 'avg_factor': 0.85}
        }
        
        # Create a knowledge distillation ensemble for improved aggregation
        sensor_ensemble_outputs = None
        fault_ensemble_outputs = None
        
        # Generate ensemble predictions from all clients on a validation set
        if len(self.clients) > 0 and hasattr(self.clients[0], 'X_val'):
            # Use first client's validation data as reference
            val_data = torch.FloatTensor(self.clients[0].X_val).to(self.device)
            
            # Collect predictions from all models
            all_sensor_preds = []
            all_fault_preds = []
            
            with torch.no_grad():
                for client_model in client_models:
                    client_model.eval()
                    outputs = client_model(val_data)
                    all_sensor_preds.append(outputs['sensor_anomalies'].cpu().numpy())
                    all_fault_preds.append(torch.sigmoid(outputs['fault_detection']).cpu().numpy())
            
            # Create weighted ensemble predictions
            if all_sensor_preds and all_fault_preds:
                sensor_ensemble_outputs = np.zeros_like(all_sensor_preds[0])
                fault_ensemble_outputs = np.zeros_like(all_fault_preds[0])
                
                for i in range(len(all_sensor_preds)):
                    sensor_ensemble_outputs += normalized_weights[i] * all_sensor_preds[i]
                    fault_ensemble_outputs += normalized_weights[i] * all_fault_preds[i]
        
        # Perform weighted FedAvg with parameter-specific aggregation
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            # Determine layer type and get its config
            layer_type = None
            for layer_name, config in layer_configs.items():
                if layer_name in key:
                    layer_type = layer_name
                    break
            
            # Use default config if layer type not found
            config = layer_configs.get(layer_type, {'memory': memory_retention, 'avg_factor': 1.0})
            
            # Apply layer-specific memory retention
            layer_memory = config['memory']
            avg_factor = config['avg_factor']
            
            # Initialize with memory retention of previous global model
            global_state[key] = global_state[key] * layer_memory
            
            # Add weighted contributions from client models
            client_contribution = torch.zeros_like(global_state[key], dtype=torch.float32)
            
            for i, client_model in enumerate(client_models):
                # Apply parameter-specific weighting
                client_param = client_model.state_dict()[key].float()
                client_contribution += normalized_weights[i] * client_param
            
            # Apply global averaging factor (controls global vs personalized balance)
            global_state[key] += (1.0 - layer_memory) * avg_factor * client_contribution
            
            # Convert back to original dtype
            global_state[key] = global_state[key].to(global_state[key].dtype)
        
        # Load aggregated state
        self.global_model.load_state_dict(global_state)
        
        # Apply knowledge distillation if ensemble predictions are available
        if sensor_ensemble_outputs is not None and fault_ensemble_outputs is not None:
            # Finetune global model to match ensemble predictions
            print("[Server applying knowledge distillation from ensemble]")
            self._apply_knowledge_distillation(
                val_data, 
                torch.FloatTensor(sensor_ensemble_outputs).to(self.device),
                torch.FloatTensor(fault_ensemble_outputs).to(self.device)
            )
        
        print("[Server completed enhanced model aggregation]")
        return self.global_model
        
    def _apply_knowledge_distillation(self, val_data, ensemble_sensor_preds, ensemble_fault_preds):
        """Apply knowledge distillation to align global model with ensemble predictions"""
        # Set model to training mode
        self.global_model.train()
        
        # Use Adam optimizer with small learning rate for gentle distillation
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.0005)
        
        # Perform distillation for a few iterations
        for iteration in range(10):
            optimizer.zero_grad()
            
            # Get predictions from global model
            outputs = self.global_model(val_data)
            model_sensor_preds = outputs['sensor_anomalies']
            model_fault_preds = torch.sigmoid(outputs['fault_detection'])
            
            # Calculate distillation loss (MSE for alignment)
            sensor_loss = F.mse_loss(model_sensor_preds, ensemble_sensor_preds)
            fault_loss = F.mse_loss(model_fault_preds, ensemble_fault_preds)
            
            # Combined loss - emphasize sensor anomaly matching
            loss = fault_loss + 2.0 * sensor_loss
            
            # Backpropagate and update
            loss.backward()
            optimizer.step()
            
            if (iteration + 1) % 5 == 0:
                print(f"Distillation iteration {iteration+1}/10, loss: {loss.item():.6f}")
        
        # Set model back to evaluation mode
        self.global_model.eval()
    
    def evaluate_global_model(self):
        """Evaluate the global model on all clients"""
        self.global_model.eval()
        total_metrics = []
        clients_with_locations = 0
        
        with torch.no_grad():
            for client in self.clients:
                # Prepare data
                val_data = torch.FloatTensor(client.X_val).to(self.device)
                val_labels = torch.FloatTensor(client.y_val).to(self.device)
                
                # Create location labels if they exist
                if client.has_location_labels:
                    val_locations = torch.FloatTensor(client.loc_val).to(self.device)
                    clients_with_locations += 1
                else:
                    val_locations = torch.zeros((len(val_labels), 8)).to(self.device)
                
                # Create data loader
                val_dataset = torch.utils.data.TensorDataset(val_data, val_labels, val_locations)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=32,
                    shuffle=False
                )
                
                # Evaluate using the evaluate function from train.py
                client_metrics = evaluate(
                    self.global_model, 
                    val_loader,
                    criterion=self.global_model.loss_fn if hasattr(self.global_model, 'loss_fn') else None,
                    device=self.device
                )
                if client_metrics is not None:
                    total_metrics.append(client_metrics)
        
        if not total_metrics:
            return {
                'test_loss': 0.0,
                'detection_auc': 0.0,
                'accuracy': 0.0,
                'location_accuracy': 0.0
            }
        
        # Calculate average metrics
        avg_metrics = {}
        metric_keys = ['total_loss', 'detection_auc', 'accuracy', 'location_accuracy']
        
        for key in metric_keys:
            values = [m.get(key, 0.0) for m in total_metrics]
            avg_metrics[key] = float(np.mean([v for v in values if v is not None]))
        
        # Rename total_loss to test_loss for consistency
        avg_metrics['test_loss'] = avg_metrics.pop('total_loss')
        
        return avg_metrics
    
    def save_model(self):
        """Save/Update the global model"""
        save_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'global_model.pt')
        torch.save(self.global_model.state_dict(), save_path)
        print("[Updated global model saved]")
    
    def train(self, num_rounds=5):
        """Train using Federated Learning"""
        print("\n[Server starting federated training]")
        best_auc = 0.0
        
        # Initialize all clients with the current global model
        for client in self.clients:
            client.receive_model(self.global_model.state_dict())
        
        for round in range(num_rounds):
            print(f"\n[Round {round + 1}/{num_rounds}]")
            
            # Train on each client using current global model
            client_models = []
            metrics_list = []
            for client in self.clients:
                model, metrics = client.train()
                client_models.append(model)
                metrics_list.append(metrics)
                print(f"Client {client.client_id} metrics: Loss={metrics['val_loss']:.4f}, Detection AUC={metrics['detection_auc']:.4f}")
                
                # Print sensor anomaly details
                print("\nSensor Anomaly Details:")
                for sensor, sensor_metrics in metrics['sensor_metrics'].items():
                    print(f"{sensor}: Mean Anomaly = {sensor_metrics['faulty_anomaly_mean']:.4f}, AUC = {sensor_metrics['auc']:.4f}")
            
            # Aggregate models and update global model
            self.global_model = self.aggregate_models(client_models, metrics_list)
            
            # Evaluate global model
            metrics = self.evaluate_global_model()
            print(f"\nGlobal model metrics: Loss={metrics['test_loss']:.4f}, Detection AUC={metrics['detection_auc']:.4f}")
            
            # Save if better performance
            if metrics['detection_auc'] > best_auc:
                best_auc = metrics['detection_auc']
                self.save_model()
                print(f"[New best model saved with Detection AUC: {best_auc:.4f}]")
            
            # Send updated global model to clients for next round
            if round < num_rounds - 1:  # Only send if not the last round
                for client in self.clients:
                    client.receive_model(self.global_model.state_dict())

# Or alternatively, define the function in server.py:
def calculate_metrics(model, X_test, y_test):
    """Calculate F1 score and accuracy for the model"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        
        # Calculate metrics
        f1 = f1_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        
    return {
        'f1_score': f1 * 100,  # Convert to percentage
        'accuracy': accuracy * 100  # Convert to percentage
    } 