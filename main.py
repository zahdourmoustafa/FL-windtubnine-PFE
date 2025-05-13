import os
from src.api.server import FederatedServer
import torch
from config.config import BASE_DIR # Import BASE_DIR

def main():
    # Initialize server with 5 clients instead of 2
    server = FederatedServer(num_clients=5)
    
    # Run federated training for 5 rounds (as per server.py default)
    server.train(num_rounds=5)
    
    # Define the save directory and ensure it exists
    save_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the full path for the final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    
    # Save the final model
    torch.save(server.global_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main() 