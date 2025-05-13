import os
import numpy as np
import sys

# Add the project root to Python path to allow importing from config
# Assumes this script is in src/ or src/utils/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config.config import OUTPUT_PATH, GLOBAL_STATS_PATH
except ImportError:
    print("Error: Could not import OUTPUT_PATH or GLOBAL_STATS_PATH from config.config.")
    print("Please ensure config/config.py is correctly set up and accessible.")
    print(f"Attempted to add {project_root} to sys.path")
    sys.exit(1)

CLIENT_NAMES = ["Client_1", "Client_2", "Client_3", "Client_4", "Client_5", "Client_Seiko"]

def calculate_and_save_global_stats():
    """
    Calculates global mean and standard deviation from all client metadata
    and saves them to a file.
    """
    total_samples_all_clients = 0
    weighted_sum_of_means = None
    weighted_sum_of_squared_means_plus_variances = None # Sum of N_client * (var_client + mean_client^2)

    print(f"Starting calculation of global statistics using data from: {OUTPUT_PATH}")

    for client_name in CLIENT_NAMES:
        metadata_file_path = os.path.join(OUTPUT_PATH, client_name, "metadata.npz")

        if not os.path.exists(metadata_file_path):
            print(f"Warning: Metadata file not found for {client_name} at {metadata_file_path}. Skipping.")
            continue

        try:
            metadata = np.load(metadata_file_path, allow_pickle=True) # allow_pickle=True for older npz files with objects
            local_mean = metadata['local_mean']
            local_std = metadata['local_std']
            
            # Check if 'num_healthy_windows_original' and 'num_damaged_windows_original' exist
            # These keys were added in a later version of preprocess.py
            # Fallback to 'train_samples', 'val_samples', 'test_samples' if they don't, 
            # though this is less accurate for original sample count before splitting.
            # The most accurate would be from the original window counts.
            if 'num_healthy_windows_original' in metadata and 'num_damaged_windows_original' in metadata:
                num_healthy = metadata['num_healthy_windows_original']
                num_damaged = metadata['num_damaged_windows_original']
            elif 'num_healthy_windows' in metadata and 'num_damaged_windows' in metadata: # older key names
                print(f"Warning: Using older keys 'num_healthy_windows' for {client_name}.")
                num_healthy = metadata['num_healthy_windows']
                num_damaged = metadata['num_damaged_windows']
            else:
                print(f"Warning: Original window counts not found for {client_name}. Approximating with split counts.")
                num_healthy = metadata.get('train_samples_healthy', 0) + metadata.get('val_samples_healthy', 0) + metadata.get('test_samples_healthy', 0)
                num_damaged = metadata.get('train_samples_damaged', 0) + metadata.get('val_samples_damaged', 0) + metadata.get('test_samples_damaged', 0)
                if num_healthy == 0 and num_damaged == 0 and 'train_samples' in metadata: # if only total samples are there
                     print(f"Further Warning: Using total split samples for {client_name} as num_healthy/num_damaged keys are missing.")
                     num_healthy = metadata['train_samples'] # Crude approximation, assuming it's mixed or mostly one type
                     num_damaged = 0 # Cannot distinguish

            N_client = int(num_healthy) + int(num_damaged)


            if N_client == 0:
                print(f"Warning: Client {client_name} has 0 samples. Skipping.")
                continue
            
            print(f"Processing {client_name}: {N_client} total original windows. Mean shape: {local_mean.shape}, Std shape: {local_std.shape}")


            if weighted_sum_of_means is None:
                weighted_sum_of_means = N_client * local_mean
                # Ensure local_std is correctly shaped if it was scalar
                local_var_term = local_std**2
                if local_mean.ndim > 0 and local_std.ndim == 0 and local_mean.shape[-1] > 1: # If mean is vector, std was scalar
                     local_var_term = np.full(local_mean.shape, local_std**2)

                weighted_sum_of_squared_means_plus_variances = N_client * (local_var_term + local_mean**2)
            else:
                weighted_sum_of_means += N_client * local_mean
                local_var_term = local_std**2
                if local_mean.ndim > 0 and local_std.ndim == 0 and local_mean.shape[-1] > 1:
                     local_var_term = np.full(local_mean.shape, local_std**2)
                weighted_sum_of_squared_means_plus_variances += N_client * (local_var_term + local_mean**2)
            
            total_samples_all_clients += N_client

        except KeyError as e:
            print(f"Error loading metadata for {client_name}: Missing key {e}. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred processing {client_name}: {e}. Skipping.")
            continue


    if total_samples_all_clients == 0:
        print("Error: No samples found across all clients. Cannot calculate global statistics.")
        return

    if weighted_sum_of_means is None or weighted_sum_of_squared_means_plus_variances is None:
        print("Error: Could not accumulate statistics. weighted_sum_of_means or weighted_sum_of_squared_means_plus_variances is None.")
        return

    global_mean = weighted_sum_of_means / total_samples_all_clients
    
    # Pooled variance: E[X^2] - (E[X])^2
    # E[X^2] = weighted_sum_of_squared_means_plus_variances / total_samples_all_clients
    # (E[X])^2 = global_mean^2
    global_variance = (weighted_sum_of_squared_means_plus_variances / total_samples_all_clients) - (global_mean**2)
    
    # Ensure variance is not negative due to floating point inaccuracies for near-zero variances
    global_variance[global_variance < 0] = 0
    global_std = np.sqrt(global_variance + 1e-8) # Add epsilon for numerical stability

    print(f"\nGlobal Mean shape: {global_mean.shape}")
    print(f"Global Std shape: {global_std.shape}")

    os.makedirs(GLOBAL_STATS_PATH, exist_ok=True)
    save_path = os.path.join(GLOBAL_STATS_PATH, "global_stats.npz")
    np.savez(save_path, global_mean=global_mean, global_std=global_std)
    print(f"\nSuccessfully calculated and saved global statistics to: {save_path}")
    print(f"Total original windows processed: {total_samples_all_clients}")

if __name__ == "__main__":
    calculate_and_save_global_stats() 