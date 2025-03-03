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

# Data processing parameters
WINDOW_SIZE = 128
OVERLAP = 64
TEST_SIZE = 0.3
RANDOM_STATE = 42
SENSORS = 8
RPM_SENSOR_INDEX = -1

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0005
WEIGHT_DECAY = 0.1
MAX_ROUNDS = 50

# Early stopping parameters
PATIENCE = 3
MIN_ROUNDS = 10
TARGET_F1 = 90.0

# Training configuration
MOMENTUM = 0.9

# New architecture parameters
LSTM_HIDDEN_SIZE = 32
NUM_LSTM_LAYERS = 1

# New federation parameters
MIN_IMPROVEMENT = 0.1
WARMUP_ROUNDS = 5