from src.api.server import FederatedServer
import torch

def main():
    # Initialize server with 2 clients
    server = FederatedServer(num_clients=2)
    
    # Run federated training for 5 rounds
    server.train(num_rounds=5)
    
    # Save the final model
    torch.save(server.global_model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    main() 