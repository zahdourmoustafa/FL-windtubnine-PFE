# Gearbox Fault Detection System

This project implements a machine learning system for detecting faults in gearboxes using sensor data. The system uses a CNN-LSTM hybrid model trained with federated learning to analyze vibration patterns and identify potential damage.

## Project Structure

```
├── config/                 # Configuration files
│   └── config.py           # Main configuration parameters
│
├── data/                   # Data directory
│   ├── raw/                # Raw .mat files with sensor data
│   ├── processed/          # Processed and windowed data
│   └── global_stats/       # Global normalization statistics
│
├── output/                 # Output directory
│   ├── models/             # Trained models and thresholds
│   ├── plots/              # Generated plots and visualizations
│   └── logs/               # Log files
│
├── src/                    # Source code
│   ├── api/                # API and communication components
│   │   ├── client.py       # Federated learning client
│   │   └── server.py       # Federated learning server
│   │
│   ├── data_processing/    # Data processing modules
│   │   └── preprocess.py   # Data preprocessing functions
│   │
│   ├── evaluation/         # Model evaluation modules
│   │   ├── evaluate_best_model.py  # Evaluation on test data
│   │   ├── model_calibration.py    # Threshold calibration
│   │   └── test_unseen_data.py     # Testing on unseen data
│   │
│   ├── models/             # Model definitions
│   │   └── model.py        # CNN-LSTM model architecture
│   │
│   ├── training/           # Training modules
│   │   └── federated_train.py  # Federated learning implementation
│   │
│   └── utils/              # Utility functions
│       └── utils.py        # Common utility functions
│
└── main.py                 # Main entry point
```

## Getting Started

1. Place your raw gearbox data (.mat files) in the `data/raw/` directory
2. Run preprocessing: `python -m src.data_processing.preprocess`
3. Train the model: `python main.py`
4. Evaluate the model: `python -m src.evaluation.evaluate_best_model`
5. Test on unseen data: `python -m src.evaluation.test_unseen_data --dataset H1`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yourrepository.git

# Navigate to the project directory
cd yourrepository

# Install dependencies
pip install -r requirements.txt
```

## Usage

Provide examples of how to use your code.

```python
# Example code
```

## Dataset

This project uses a dataset that is not included in the repository due to size constraints.
You can download the dataset from [source link] and place it in the `dataset/` directory.

## Model Architecture

The system uses a hybrid CNN-LSTM architecture:

- CNN layers extract spatial features from vibration signals
- LSTM layers capture temporal dependencies in the signal patterns
- Fully connected layers for final classification

## Federated Learning

The model is trained using federated learning, which allows:

- Training across multiple data sources without sharing raw data
- Improved privacy and data security
- Better generalization across different operating conditions

## Evaluation

The system provides multiple evaluation tools:

- ROC curves and confusion matrices
- Probability distribution analysis
- Signal pattern analysis
- Threshold calibration for optimal performance

## License

[Your License]
