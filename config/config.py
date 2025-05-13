import os

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")  # data/raw for .mat files
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed")  # data/processed for processed data
GLOBAL_STATS_PATH = os.path.join(BASE_DIR, "data", "global_stats")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GLOBAL_STATS_PATH, exist_ok=True)

# --- Data processing parameters (used by preprocess.py) ---
WINDOW_SIZE = 128  # Size of each time-series window
OVERLAP = 64       # Overlap between consecutive windows
RANDOM_STATE = 42  # For reproducible data shuffling and splitting

# Data Splitting Ratios for preprocess.py (e.g., for a 60% train, 10% validation, 30% test split)
TRAIN_RATIO = 0.6  # Proportion of data to use for training
# VALIDATION_RATIO_OF_REMAINING defines how much of the (1 - TRAIN_RATIO) part goes to validation.
# e.g., if TRAIN_RATIO = 0.6, remaining is 0.4. If VALIDATION_RATIO_OF_REMAINING = 0.25, then val_size = 0.25 * 0.4 = 0.1 (10% of total)
# The rest of the remaining data (0.4 - 0.1 = 0.3) becomes the test set (30% of total)
VALIDATION_RATIO_OF_REMAINING = 0.25

# TEST_SIZE is kept for reference or other scripts, but preprocess.py derives test size from the above.
TEST_SIZE = 0.3 # Proportion of data for the final test set (should be consistent with above)


# Fault Class Information (used by preprocess.py)
NUM_FAULT_CLASSES = 8 # Example: 0 for Healthy, 1-7 for different fault types (as per PRD Table 2 + Healthy)
# **IMPORTANT**: Customize this mapping based on your actual client data and fault types.
# This maps a client_name (from preprocess.py) to an integer fault ID for their 'damaged' dataset.
# Fault IDs should range from 1 to (NUM_FAULT_CLASSES - 1). Healthy is automatically labeled as 0.
FAULT_TYPE_MAPPING_CONFIG = {
    "Client_1": 1,  # Example: Client_1's damaged data corresponds to fault type 1 (e.g., HS-ST Scuffing)
    "Client_2": 2,  # Example: Client_2's damaged data corresponds to fault type 2 (e.g., HS-SH Overheating)
    "Client_3": 3,
    "Client_4": 4,
    "Client_5": 5,
    # "Client_N": X, # Add all your clients and map to appropriate fault IDs from PRD Table 2
}

# Feature Extraction Toggles (used by preprocess.py)
APPLY_FFT_FEATURES = False          # Set to True if you implement and want to use FFT features
APPLY_STATISTICAL_FEATURES = False  # Set to True if you implement and want to use statistical features

# --- Specific Sensor Damage Profiles ---
# Maps a FILENAME (exact match for the 'damaged' file) to a list of 0-indexed sensor indices
# that are considered damaged. Other sensors in that file will be treated as healthy for GT.
# Sensor indices correspond to AN3=0, AN4=1, ..., AN10=7 after extraction.
SPECIFIC_SENSOR_DAMAGE_PROFILES = {
    "seiko.mat": { # For seiko.mat
        "damaged_indices": [0, 4, 6],  # AN3 (idx 0), AN7 (idx 4), AN9 (idx 6)
        "target_healthy_score": 0.05,
        "target_damaged_score": 0.95
    }
    # Add other specific profiles here if needed, e.g.:
    # "another_mixed_file.mat": {"damaged_indices": [1, 5], "target_healthy_score": 0.05, "target_damaged_score": 0.95},
}

# --- End of parameters directly used by preprocess.py ---

SENSORS = 8 # Number of sensor channels (AN3-AN10)
RPM_SENSOR_INDEX = -1 # Not directly used by current preprocess.py logic for channel selection

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
WEIGHT_DECAY = 0.0001
MAX_ROUNDS = 50

# Early stopping parameters
early_stopping_patience = 10
metric_for_best_model = 'f1_score'

# Training configuration
MOMENTUM = 0.9

# New architecture parameters
LSTM_HIDDEN_SIZE = 32
NUM_LSTM_LAYERS = 1

# New federation parameters
MIN_IMPROVEMENT = 0.1
WARMUP_ROUNDS = 5

# Loss function parameters
LOCATION_LOSS_WEIGHT = 1.0
FOCAL_GAMMA = 2.0
POS_WEIGHT = 5.0

# Detection threshold parameters
DEFAULT_THRESHOLD = 0.5
MIN_RECALL = 0.5

# Scheduler settings
scheduler_patience = 5

# Augmentation
use_augmentation = True

# Gradient clipping
gradient_clip_val = 1.0

# Label smoothing
label_smoothing = 0.05