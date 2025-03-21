import torch
from src.models.model import GearboxCNNLSTM as GearboxCNN
import numpy as np
import os
from config.config import *
from src.training.train import train_epoch, evaluate

def train_client(model, X_train, y_train, X_val, y_val, loc_train=None, loc_val=None, config=None):
    """Train a client's local model with fault localization support"""
    if config is None:
        config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Create data loaders
    train_data = torch.FloatTensor(X_train)
    train_labels = torch.FloatTensor(y_train)
    val_data = torch.FloatTensor(X_val)
    val_labels = torch.FloatTensor(y_val)
    
    # Create dummy locations (not used in new approach)
    train_locations = torch.zeros((len(train_labels), 8))
    val_locations = torch.zeros((len(val_labels), 8))
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        train_data,
        train_labels,
        train_locations
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        val_data,
        val_labels,
        val_locations
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size']
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = None
    
    for epoch in range(config['num_epochs']):
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion=None,  # Will use default FaultLocalizationLoss
            device=device
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader,
            criterion=None,  # Will use default FaultLocalizationLoss
            device=device
        )
        
        # Save best model
        current_val_loss = val_metrics['total_loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict().copy()
            best_metrics = {
                'val_loss': val_metrics['total_loss'],
                'detection_auc': val_metrics['detection_auc'],
                'sensor_metrics': val_metrics['sensor_metrics']
            }
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"Detection AUC: {val_metrics['detection_auc']:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, best_metrics

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GearboxCNN(dropout_rate=0.3).to(self.device)  # Initialize model with dropout
        self.has_location_labels = False  # Track whether we have real location labels
        
        # Introduce client heterogeneity factors (realistic FL)
        self.data_quality = np.random.uniform(0.7, 1.0)  # Each client has different data quality
        self.convergence_rate = np.random.uniform(0.8, 1.2)  # How quickly this client converges
        self.regularization_strength = np.random.uniform(0.8e-4, 1.5e-4)  # Client-specific regularization
        
        # Add operating condition awareness for domain adaptation
        self.operating_conditions = {
            "rpm_range": np.random.uniform(1000, 2000),
            "load_profile": np.random.choice(["high", "medium", "low"]),
            "environment": np.random.choice(["normal", "dusty", "humid"])
        }
        
        # Track client's specialty for better contribution in certain damage types
        self.specialized_damage_types = np.random.choice(
            ["ring_gear", "high_speed", "bearings", "low_speed"], 
            size=np.random.randint(1, 3),  # Each client specializes in 1-2 damage types
            replace=False
        )
        
        print(f"Client {client_id} initialized with data quality {self.data_quality:.2f}")
        print(f"Operating conditions: {self.operating_conditions}")
        print(f"Specialized in: {self.specialized_damage_types}")
        
        self.load_data()
    
    def load_data(self):
        """Load or generate client data"""
        client_path = os.path.join(OUTPUT_PATH, self.client_id)
        os.makedirs(client_path, exist_ok=True)
        
        try:
            # Try to load existing data
            self.X_train = np.load(os.path.join(client_path, "train_features.npy"))
            self.y_train = np.load(os.path.join(client_path, "train_labels.npy"))
            self.X_val = np.load(os.path.join(client_path, "val_features.npy"))
            self.y_val = np.load(os.path.join(client_path, "val_labels.npy"))
            
            # Try to load existing location labels
            try:
                self.loc_train = np.load(os.path.join(client_path, "train_locations.npy"))
                self.loc_val = np.load(os.path.join(client_path, "val_locations.npy"))
                self.has_location_labels = True
            except FileNotFoundError:
                # Generate location labels if not found
                self._generate_synthetic_locations()
                
            # Simulate real-world client data quality issues
            # Add varying levels of noise based on client data quality
            self._add_realistic_noise()
                
        except FileNotFoundError:
            print(f"No data found for client {self.client_id}, generating synthetic data")
            # Generate synthetic data
            num_train = 1000
            num_val = 200
            num_features = 9  # 8 sensors + 1 RPM
            
            # Generate training data
            self.X_train = np.random.randn(num_train, 256, num_features)  # Random sensor readings
            self.y_train = np.random.binomial(1, 0.3, num_train)  # 30% fault rate
            
            # Generate validation data
            self.X_val = np.random.randn(num_val, 256, num_features)
            self.y_val = np.random.binomial(1, 0.3, num_val)
            
            # Save synthetic data
            np.save(os.path.join(client_path, "train_features.npy"), self.X_train)
            np.save(os.path.join(client_path, "train_labels.npy"), self.y_train)
            np.save(os.path.join(client_path, "val_features.npy"), self.X_val)
            np.save(os.path.join(client_path, "val_labels.npy"), self.y_val)
            
            # Generate location labels
            self._generate_synthetic_locations()
            
            # Add realistic noise
            self._add_realistic_noise()
        
        # Save location labels
        np.save(os.path.join(client_path, "train_locations.npy"), self.loc_train)
        np.save(os.path.join(client_path, "val_locations.npy"), self.loc_val)
    
    def _add_realistic_noise(self):
        """Add realistic noise based on client data quality"""
        noise_level = (1.0 - self.data_quality) * 0.2  # Scale noise by data quality
        
        # Add sensor-specific noise (some sensors more noisy than others)
        for sensor_idx in range(self.X_train.shape[2] - 1):  # Exclude RPM
            sensor_noise = np.random.uniform(0.5, 1.5) * noise_level
            self.X_train[:, :, sensor_idx] += np.random.normal(0, sensor_noise, self.X_train[:, :, sensor_idx].shape)
            self.X_val[:, :, sensor_idx] += np.random.normal(0, sensor_noise, self.X_val[:, :, sensor_idx].shape)
            
        # Add label noise (incorrect labels)
        label_noise_rate = noise_level * 0.1  # 0-10% of labels might be wrong
        train_flip_mask = np.random.rand(len(self.y_train)) < label_noise_rate
        val_flip_mask = np.random.rand(len(self.y_val)) < label_noise_rate
        
        self.y_train[train_flip_mask] = 1 - self.y_train[train_flip_mask]
        self.y_val[val_flip_mask] = 1 - self.y_val[val_flip_mask]
    
    def _generate_synthetic_locations(self):
        """Generate synthetic location labels for testing"""
        num_locations = 8  # Number of possible fault locations
        
        # Generate synthetic locations for faulty samples in training set
        faulty_train_mask = self.y_train == 1
        num_faulty_train = np.sum(faulty_train_mask)
        self.loc_train = np.zeros((len(self.y_train), num_locations))
        if num_faulty_train > 0:
            # Randomly assign locations to faulty samples
            faulty_indices = np.where(faulty_train_mask)[0]
            random_locations = np.random.randint(0, num_locations, size=num_faulty_train)
            self.loc_train[faulty_indices, random_locations] = 1
        
        # Generate synthetic locations for faulty samples in validation set
        faulty_val_mask = self.y_val == 1
        num_faulty_val = np.sum(faulty_val_mask)
        self.loc_val = np.zeros((len(self.y_val), num_locations))
        if num_faulty_val > 0:
            # Randomly assign locations to faulty samples
            faulty_indices = np.where(faulty_val_mask)[0]
            random_locations = np.random.randint(0, num_locations, size=num_faulty_val)
            self.loc_val[faulty_indices, random_locations] = 1
        
        print(f"Generated synthetic locations for {num_faulty_train} training and {num_faulty_val} validation samples")
        self.has_location_labels = True
    
    def receive_model(self, model_state):
        """Receive model from server with improved transfer learning"""
        print(f"[{self.client_id} received global model]")
        if self.model is None:
            self.model = GearboxCNN(dropout_rate=0.3).to(self.device)
        
        # Store previous local knowledge (before applying global model)
        local_knowledge = {}
        if hasattr(self.model, 'sensor_anomaly_heads') and hasattr(self.model, 'fault_detector'):
            # Save local knowledge from key layers that should preserve client adaptation
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    # Focus on client-specific knowledge in output layers
                    if ('sensor_anomaly_heads' in name or 'fault_detector' in name or 
                        'anomaly_calibration' in name or 'joint_anomaly_head' in name):
                        local_knowledge[name] = param.clone()
        
        # Load the global model state
        self.model.load_state_dict(model_state)
        
        # Apply domain adaptation based on operating conditions and specialties
        with torch.no_grad():
            # Apply personalization for client-specific adaptation
            for name, param in self.model.named_parameters():
                # 1. Restore partial local knowledge in output layers (preserve client adaptations)
                if name in local_knowledge:
                    # Blend global and local knowledge (70% global, 30% local)
                    param.data = 0.7 * param.data + 0.3 * local_knowledge[name]
                
                # 2. Apply condition-specific adaptation to relevant layers
                if 'sensor_relationship_matrix' in name:
                    # Adjust relationship matrix based on client's specialization
                    for specialty in self.specialized_damage_types:
                        if specialty == "ring_gear":
                            # Strengthen ring gear relationships (AN3-AN4 are indices 0-1)
                            param.data[0, 1] *= 1.2
                            param.data[1, 0] *= 1.2
                        elif specialty == "high_speed":
                            # Strengthen high-speed section (AN7-AN9 are indices 4-6)
                            for i in range(4, 7):
                                for j in range(4, 7):
                                    if i != j:
                                        param.data[i, j] *= 1.2
                
                # 3. Add minimal noise for other parameters (avoids exact convergence)
                if 'norm' not in name and name not in local_knowledge:
                    # Add small Gaussian noise to model weights (simulates local adaptation)
                    drift_factor = np.random.uniform(0.0005, 0.005)  # Reduced from original
                    param.add_(torch.randn_like(param) * drift_factor)
        
        # Add client-specific operating condition embedding if model supports it
        if hasattr(self.model, 'set_operating_conditions'):
            self.model.set_operating_conditions(self.operating_conditions)
    
    def train(self, config=None):
        """Train the local model with transfer learning enhancements"""
        print(f"[{self.client_id} starting training]")
        
        # Create client-specific config based on convergence rate and regularization
        if config is None:
            config = {
                'batch_size': 32,
                'learning_rate': 0.001 * self.convergence_rate,  # Client-specific learning rate
                'weight_decay': self.regularization_strength,  # Client-specific regularization
                'num_epochs': int(10 * (1.0 / self.convergence_rate)),  # Faster clients need fewer epochs
                'patience': 5,  # Add early stopping patience
                'device': self.device,
                'sensor_coupling_weight': 0.3,  # Default value for sensor coupling loss
                'transfer_learning': True  # Enable transfer learning
            }
        
        # Add operating conditions to the config
        config['operating_conditions'] = self.operating_conditions
        
        # Selective fine-tuning based on transfer learning principles
        if config.get('transfer_learning', False):
            # First freeze the feature extraction layers to preserve global knowledge
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['sensor_cnns', 'lstm', 'rpm_cnn', 'freq_extractors']):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
            # Train for a few epochs with frozen feature extractors
            trained_model, _ = train_client(
                self.model, 
                self.X_train, 
                self.y_train,
                self.X_val,
                self.y_val,
                self.loc_train,
                self.loc_val,
                {**config, 'num_epochs': max(1, int(config['num_epochs'] * 0.3))}
            )
            
            # Then unfreeze all layers for fine-tuning with lower learning rate
            for param in trained_model.parameters():
                param.requires_grad = True
                
            config['learning_rate'] = config['learning_rate'] * 0.5
            
            # Enable Monte Carlo dropout for more realistic evaluation
            trained_model.enable_mc_dropout()
            
            # Continue training with all layers unfrozen
            trained_model, metrics = train_client(
                trained_model, 
                self.X_train, 
                self.y_train,
                self.X_val,
                self.y_val,
                self.loc_train,
                self.loc_val,
                {**config, 'num_epochs': max(1, int(config['num_epochs'] * 0.7))}
            )
        else:
            # Standard training without transfer learning
            self.model.enable_mc_dropout()
            
            trained_model, metrics = train_client(
                self.model, 
                self.X_train, 
                self.y_train,
                self.X_val,
                self.y_val,
                self.loc_train,
                self.loc_val,
                config
            )
        
        # Disable Monte Carlo dropout after training
        trained_model.disable_mc_dropout()
        
        print(f"[{self.client_id} completed training]")
        
        # Add client-specific metadata to metrics
        metrics['data_quality'] = self.data_quality
        metrics['convergence_rate'] = self.convergence_rate
        metrics['operating_conditions'] = self.operating_conditions
        metrics['specialized_damage_types'] = self.specialized_damage_types
        
        return trained_model, metrics 