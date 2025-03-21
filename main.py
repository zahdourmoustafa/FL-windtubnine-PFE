from src.api.server import FederatedServer
import torch

def main():
    # Initialize server with 5 clients instead of 2
    server = FederatedServer(num_clients=5)
    
    # Run federated training for 5 rounds
    server.train(num_rounds=15)
    
    # Save the final model
    torch.save(server.global_model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    main() 