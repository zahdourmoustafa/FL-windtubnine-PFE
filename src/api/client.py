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
        
        print(f"Client {client_id} initialized with data quality {self.data_quality:.2f}")
        
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
        """Receive model from server"""
        print(f"[{self.client_id} received global model]")
        if self.model is None:
            self.model = GearboxCNN(dropout_rate=0.3).to(self.device)
        
        # Load the model state
        self.model.load_state_dict(model_state)
        
        # Introduce realistic model personalization/drift
        # This simulates how each client might slightly modify weights based on local conditions
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Skip batch norm parameters to maintain stability
                if 'norm' not in name:
                    # Add small Gaussian noise to model weights (simulates local adaptation)
                    drift_factor = np.random.uniform(0.001, 0.01)
                    param.add_(torch.randn_like(param) * drift_factor)
    
    def train(self, config=None):
        """Train the local model"""
        print(f"[{self.client_id} starting training]")
        
        # Create client-specific config based on convergence rate and regularization
        if config is None:
            config = {
                'batch_size': 32,
                'learning_rate': 0.001 * self.convergence_rate,  # Client-specific learning rate
                'weight_decay': self.regularization_strength,  # Client-specific regularization
                'num_epochs': int(5 * (1.0 / self.convergence_rate)),  # Faster clients need fewer epochs
                'device': self.device
            }
        
        # Enable Monte Carlo dropout for more realistic evaluation
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
        self.model.disable_mc_dropout()
        
        print(f"[{self.client_id} completed training]")
        
        # Add client-specific metrics
        metrics['data_quality'] = self.data_quality
        metrics['convergence_rate'] = self.convergence_rate
        
        return trained_model, metrics 