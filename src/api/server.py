import torch
from src.models.model import GearboxCNNLSTM as GearboxCNN
from config.config import *
import numpy as np
import os
from src.utils.utils import calculate_metrics
from sklearn.metrics import f1_score, accuracy_score
from src.training.train import train_epoch, evaluate  # Updated import
from src.api.client import FederatedClient

class FederatedServer:
    def __init__(self, num_clients=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_clients = num_clients
        self.clients = []
        
        # Initialize or load global model
        save_dir = os.path.join(OUTPUT_PATH, 'models')
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
        """Aggregate client models using weighted FedAvg based on performance metrics"""
        print(f"\n[Server aggregating {len(client_models)} client models]")
        
        # Compute weights based on client metrics
        # Higher AUC, better data quality, and lower loss get higher weights
        weights = []
        for i, metrics in enumerate(client_metrics):
            # Extract key metrics: detection AUC, data quality, and validation loss
            auc = metrics.get('detection_auc', 0.5)  # Default to 0.5 if not available
            data_quality = metrics.get('data_quality', 0.5)  # Default to 0.5 if not available
            val_loss = metrics.get('val_loss', 1.0)  # Default to 1.0 if not available
            
            # Calculate weight based on metrics (normalize loss to be higher for lower loss)
            loss_factor = 1.0 / (val_loss + 0.1)  # Add small epsilon to avoid division by zero
            
            # Compute combined weight
            weight = (auc * 0.5) + (data_quality * 0.3) + (loss_factor * 0.2)
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
        memory_retention = 0.2  # Keep 20% of the old model
        
        # Perform weighted FedAvg
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            # Initialize with memory retention of previous global model
            global_state[key] = global_state[key] * memory_retention
            
            # Add weighted contributions from client models
            for i, client_model in enumerate(client_models):
                # Convert to float for precision in averaging
                client_param = client_model.state_dict()[key].float()
                global_state[key] += (1.0 - memory_retention) * normalized_weights[i] * client_param
            
            # Convert back to original dtype
            global_state[key] = global_state[key].to(global_state[key].dtype)
        
        self.global_model.load_state_dict(global_state)
        print("[Server completed weighted model aggregation with memory retention]")
        return self.global_model
    
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
        save_dir = os.path.join(OUTPUT_PATH, 'models')
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
                    print(f"{sensor}: Mean Anomaly = {sensor_metrics['mean_anomaly']:.4f}, AUC = {sensor_metrics['anomaly_auc']:.4f}")
            
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