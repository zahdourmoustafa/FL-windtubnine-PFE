import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import argparse

# Attempt to import from local structure (common for scripts in project root)
try:
    from config import config as cfg  # Assuming a config.py or config/config.py
    from src.models.model import GearboxCNNLSTM
    from src.training.train import train as train_model # Renamed to avoid conflict
    from src.training.train import evaluate as evaluate_model # Renamed
except ModuleNotFoundError:
    # Fallback for different project structures or if run from a sub-directory
    # This assumes your PYTHONPATH is set up, or you adjust sys.path
    import sys
    # Add project root to sys.path if the script is in a subdirectory like 'experiments'
    # Adjust the number of os.path.dirname calls based on script location relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__)) # If script is in root
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # If script is in experiments/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from config import config as cfg
    from src.models.model import GearboxCNNLSTM
    from src.training.train import train as train_model
    from src.training.train import evaluate as evaluate_model

class GearboxDataset(Dataset):
    """Dataset for loading preprocessed gearbox data for a single client."""
    def __init__(self, client_data_path: str, dataset_type='train', file_limit=None):
        """
        Args:
            client_data_path: Path to the processed data directory for a specific client 
                              (e.g., 'data/processed/Client_1').
            dataset_type: 'train', 'val', or 'test'.
            file_limit (int, optional): Maximum number of samples to load from files. For quick testing.
        """
        self.features_path = os.path.join(client_data_path, f'{dataset_type}_features.npy')
        self.labels_path = os.path.join(client_data_path, f'{dataset_type}_labels.npy')
        self.locations_path = os.path.join(client_data_path, f'{dataset_type}_locations.npy')

        if not (os.path.exists(self.features_path) and \
                os.path.exists(self.labels_path) and \
                os.path.exists(self.locations_path)):
            raise FileNotFoundError(f"Processed data files not found for {dataset_type} in {client_data_path}")

        self.features = np.load(self.features_path)
        self.labels = np.load(self.labels_path)         # Binary labels (0 or 1)
        self.locations = np.load(self.locations_path)   # Per-sensor anomaly GT (float 0.0-1.0)

        if file_limit is not None and file_limit > 0:
            self.features = self.features[:file_limit]
            self.labels = self.labels[:file_limit]
            self.locations = self.locations[:file_limit]
            print(f"Limited {dataset_type} dataset to {file_limit} samples.")

        assert len(self.features) == len(self.labels) == len(self.locations), \
            "Mismatch in number of samples between features, labels, and locations."
        
        print(f"Loaded {dataset_type} data for client: {client_data_path}")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Labels shape: {self.labels.shape}, Unique labels: {np.unique(self.labels)}")
        print(f"  Locations shape: {self.locations.shape}")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features are expected to be (window_length, num_channels=8) by the model after permutation
        # Preprocess.py saves them as (num_samples, window_length, num_channels)
        # Model expects (batch, time_steps, features/channels)
        
        features_sample = torch.from_numpy(self.features[idx]).float()
        label_sample = torch.tensor(self.labels[idx], dtype=torch.float) # BCEWithLogitsLoss expects float
        locations_sample = torch.from_numpy(self.locations[idx]).float()
        
        return features_sample, label_sample, locations_sample

def main(args):
    # --- Configuration ---
    print("Loading configuration...")
    DEVICE = args.device if args.device else (cfg.DEVICE if hasattr(cfg, 'DEVICE') else ('cuda' if torch.cuda.is_available() else 'cpu'))
    CLIENT_ID_TO_TEST = args.client_id
    # Ensure BASE_PROCESSED_DATA_PATH is correctly fetched from cfg
    if not hasattr(cfg, 'OUTPUT_PATH') or not cfg.OUTPUT_PATH:
        print("ERROR: cfg.OUTPUT_PATH is not defined in your config/config.py. This is required.")
        return
    BASE_PROCESSED_DATA_PATH = cfg.OUTPUT_PATH
    
    if not hasattr(cfg, 'WINDOW_SIZE') or not cfg.WINDOW_SIZE:
        print("ERROR: cfg.WINDOW_SIZE is not defined in your config/config.py. This is required.")
        return
    WINDOW_SIZE = cfg.WINDOW_SIZE
    NUM_SENSORS = getattr(cfg, 'SENSORS', 8)
    
    NUM_EPOCHS = args.epochs if args.epochs is not None else getattr(cfg, 'EPOCHS', 20)
    BATCH_SIZE = args.batch_size if args.batch_size is not None else getattr(cfg, 'BATCH_SIZE', 32)
    LEARNING_RATE = args.lr if args.lr is not None else getattr(cfg, 'LR', 0.001)
    
    class TrainingConfig:
        def __init__(self):
            self.device = DEVICE
            self.num_epochs = NUM_EPOCHS
            self.learning_rate = LEARNING_RATE # train.py might look for config.learning_rate or config.LR
            self.LR = LEARNING_RATE 
            
            # Parameters for FaultLocalizationLoss and train/evaluate functions
            self.LOCATION_LOSS_WEIGHT = getattr(cfg, 'LOCATION_LOSS_WEIGHT', 0.5) # Default if not in main cfg
            self.FOCAL_GAMMA = getattr(cfg, 'FOCAL_GAMMA', 2.0)
            self.POS_WEIGHT = getattr(cfg, 'POS_WEIGHT', 3.0)
            self.label_smoothing = getattr(cfg, 'label_smoothing', 0.05)
            self.weight_decay = getattr(cfg, 'WEIGHT_DECAY', 1e-4) # Example default
            self.scheduler_patience = getattr(cfg, 'scheduler_patience', 5)
            self.use_augmentation = getattr(cfg, 'use_augmentation', True)
            self.MIN_RECALL = getattr(cfg, 'MIN_RECALL', 0.5) # from train.py
            self.DEFAULT_THRESHOLD = getattr(cfg, 'DEFAULT_THRESHOLD', 0.5) # from train.py


    training_config = TrainingConfig()

    print(f"Using device: {DEVICE}")
    print(f"Testing with Client ID: {CLIENT_ID_TO_TEST}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")

    client_data_path = os.path.join(BASE_PROCESSED_DATA_PATH, CLIENT_ID_TO_TEST)
    if not os.path.isdir(client_data_path):
        print(f"ERROR: Processed data directory not found for client {CLIENT_ID_TO_TEST} at {client_data_path}")
        print("Please ensure preprocess.py has been run and OUTPUT_PATH in config.py is correct.")
        return

    # --- Datasets and DataLoaders ---
    try:
        train_dataset = GearboxDataset(client_data_path, dataset_type='train', file_limit=args.file_limit)
        val_dataset = GearboxDataset(client_data_path, dataset_type='val', file_limit=args.file_limit)
    except FileNotFoundError as e:
        print(e)
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True if DEVICE=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=True if DEVICE=='cuda' else False)

    # --- Model Initialization ---
    # Parameters for GearboxCNNLSTM from config or defaults
    lstm_hidden_size = getattr(cfg, 'LSTM_HIDDEN_SIZE', 32)
    num_lstm_layers = getattr(cfg, 'NUM_LSTM_LAYERS', 1)
    dropout_rate = getattr(cfg, 'DROPOUT_RATE', 0.3) # Assuming a DROPOUT_RATE in your config

    model = GearboxCNNLSTM(
        window_size=WINDOW_SIZE, # This is more about data shape; model adapts via pooling
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_sensors=NUM_SENSORS, # Should be 8
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    print("Model Initialized:")
    # print(model) # Printing the whole model can be very verbose
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --- Training ---
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, training_config)
    print("Training finished.")

    # --- Save the best model ---
    model_save_path = f"best_model_{CLIENT_ID_TO_TEST}.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # --- Final Evaluation on Test Set (Optional) ---
    if args.run_test_eval:
        print("\nStarting final evaluation on TEST set...")
        try:
            test_dataset = GearboxDataset(client_data_path, dataset_type='test', file_limit=args.file_limit)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
            
            print("Evaluating trained model on test set...")
            test_metrics = evaluate_model(trained_model, test_loader, device=DEVICE, num_sensors=NUM_SENSORS)
            # print_evaluation_summary is called within evaluate_model in train.py
            # print("Test Set Evaluation Metrics:") # Already printed by evaluate_model

        except FileNotFoundError:
            print(f"Test data not found for client {CLIENT_ID_TO_TEST}. Skipping test set evaluation.")
        except Exception as e:
            print(f"Error during test set evaluation: {e}")
    
    print("\nCentralized training script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run centralized training for Gearbox Fault Detection.")
    parser.add_argument('--client_id', type=str, default="Client_1", help="Client ID to use for training and validation (e.g., Client_1).")
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size for training (overrides config).")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument('--device', type=str, default=None, help="Device to use ('cuda' or 'cpu', overrides config/auto-detect).")
    parser.add_argument('--file_limit', type=int, default=None, help="Limit number of samples loaded from .npy files for quick testing.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader. Set to >0 for parallel data loading if beneficial.")
    parser.add_argument('--run_test_eval', action='store_true', help="Run evaluation on the test set after training.")
    
    args = parser.parse_args()
    main(args) 