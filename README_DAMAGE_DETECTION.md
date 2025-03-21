# Wind Turbine Gearbox Damage Detection

This document explains how to use the advanced damage detection feature in the Wind Turbine Gearbox Monitoring System. This feature analyzes sensor data to identify which specific sensors are showing anomalies and determines the likely failure mode (e.g., overheating, scuffing, fretting corrosion).

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Place your test data (.mat files) in the `data/unseen_data` directory

## Running Damage Detection

### Quick Start

The simplest way to run damage detection is using the convenience script:

```bash
python test_gearbox.py your_dataset.mat
```

You can also just provide the filename if it's in the default directory:

```bash
python test_gearbox.py D3
```

### Advanced Options

```bash
python test_gearbox.py your_dataset.mat --threshold 0.6 --no_mc --output ./custom_output
```

Options:
- `--threshold`: Set a custom detection threshold (default: auto-calibrated)
- `--no_mc`: Disable Monte Carlo dropout for faster analysis (less accurate)
- `--output`: Specify a custom output directory
- `--model`: Path to a custom model file
- `--adaptive`: Use adaptive thresholding for better sensor detection
- `--full_scan`: Enable comprehensive scan to detect subtle damages
- `--sensor_threshold`: Set specific threshold for sensor anomaly detection
- `--no_classify`: Disable the supervised classifier

### Using the Python Module

For more control, you can use the module directly:

```bash
python -m src.evaluation.test_unseen_data --dataset data/unseen_data/D3.mat --format html
```

Additional options:
- `--format`: Choose output format (text, html, json)
- `--sensor_threshold`: Set threshold for detecting anomalous sensors (default: 0.5)
- `--adaptive_threshold`: Enable adaptive thresholds for sensors
- `--full_scan`: More sensitive detection to find all possible issues

## Advanced Damage Detection Features

### Supervised Learning Classifier

The system includes a supervised machine learning classifier that improves damage detection accuracy based on historical patterns. To train the classifier:

```bash
python train_damage_classifier.py
```

This creates a model that understands which sensor patterns correspond to which damage modes, resulting in more accurate diagnosis.

### Adaptive Thresholding

Enable adaptive thresholding for more intelligent sensor analysis:

```bash
python test_gearbox.py D3 --adaptive
```

This feature automatically adjusts detection thresholds based on known sensor relationships and mechanical coupling in the gearbox.

## Understanding the Results

The damage detection system provides output in several formats:

1. **Console output**: Shows a summary of all detected issues, highlighting damaged sensors and their failure modes

2. **HTML report**: A comprehensive report with visual indicators of damage severity, component information, and recommended actions

3. **Text summary**: A simple text file with key findings 

4. **JSON data**: Structured data for programmatic use or API integration

5. **Visualizations**: Bar charts showing anomaly scores for each sensor

## Example Output

When a damaged sensor is detected, you'll see output like:

