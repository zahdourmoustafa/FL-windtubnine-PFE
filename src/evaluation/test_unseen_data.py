"""
Unseen Data Testing Module for Wind Turbine Gearbox Fault Detection

This module provides functionality to test the trained model on unseen data,
detecting both faulty sensors and their specific damage modes.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime
import json
from tabulate import tabulate
from termcolor import colored
import time
import scipy.io as sio

# Add project root to Python path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model import GearboxCNNLSTM
from src.utils.utils import load_unseen_dataset, plot_confusion_matrix
from src.data_processing.damage_mappings import (
    SENSOR_TO_COMPONENT, 
    COMPONENT_FAILURE_MODES, 
    DIAGNOSTIC_RULES,
    SENSOR_SENSITIVITY,
    SENSOR_MODE_CORRELATIONS,
    SEVERITY_THRESHOLDS
)
from src.data_processing.ground_truth import SENSOR_GROUND_TRUTH
from config.config import BASE_DIR, DEFAULT_THRESHOLD

# Import torch if not imported
try:
    import torch
except ImportError:
    print("Error: PyTorch is required but not installed.")
    sys.exit(1)

# Create a custom JSON encoder to handle numpy types and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bool):
            return int(obj)  # Convert boolean to integer (0 or 1)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

class GearboxDamageDetector:
    """Class for detecting and analyzing gearbox damage in unseen data."""
    
    def __init__(self, model_path=None, threshold=DEFAULT_THRESHOLD, use_mc_dropout=True, 
                 sensor_threshold=0.5, adaptive_thresholds=False):
        """
        Initialize the damage detector with a trained model.
        
        Args:
            model_path: Path to the trained model (.pth file)
            threshold: Threshold for fault detection (0-1)
            use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
            sensor_threshold: Threshold for sensor anomaly detection
            adaptive_thresholds: Whether to use adaptive thresholds for sensors
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set default model path if none provided
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "final_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(BASE_DIR, "models", "global_model.pt")
        
        # Load model
        self.model = GearboxCNNLSTM().to(self.device)
        
        try:
            print(f"Loading model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Proceeding with default model initialization")
        
        self.model.eval()
        self.threshold = threshold
        self.sensor_threshold = sensor_threshold
        self.adaptive_thresholds = adaptive_thresholds
        self.use_mc_dropout = use_mc_dropout
        
        # Sensor names for reference
        self.sensor_names = [f'AN{i}' for i in range(3, 11)]
        
        # Setup adaptive thresholds based on sensor sensitivity if enabled
        if self.adaptive_thresholds:
            self.sensor_thresholds = {
                sensor: self.sensor_threshold * SENSOR_SENSITIVITY.get(sensor, 1.0) 
                for sensor in self.sensor_names
            }
        else:
            self.sensor_thresholds = {sensor: self.sensor_threshold for sensor in self.sensor_names}
            
        print(f"Detection threshold: {self.threshold}")
        print(f"Sensor anomaly thresholds: {'Adaptive' if adaptive_thresholds else 'Fixed'}")
        
    def process_file(self, file_path):
        """
        Process a single unseen data file.
        
        Args:
            file_path: Path to the .mat file with sensor data
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'-' * 50}")
        print(f"Processing file: {os.path.basename(file_path)}")
        print(f"{'-' * 50}")
        
        # Load and preprocess the data
        windows, filename = load_unseen_dataset(file_path)
        
        if windows is None:
            return {"error": f"Failed to load {file_path}"}
        
        # Convert to tensor and move to device
        tensor_data = torch.FloatTensor(windows).to(self.device)
        
        # Run inference with MC dropout for uncertainty estimation
        if self.use_mc_dropout:
            self.model.enable_mc_dropout()
            num_samples = 10  # Number of MC samples
            all_detections = []
            all_anomalies = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = self.model(tensor_data)
                    all_detections.append(torch.sigmoid(outputs['fault_detection']).cpu().numpy())
                    all_anomalies.append(outputs['sensor_anomalies'].cpu().numpy())
            
            # Average the predictions
            fault_probs = np.mean(np.concatenate(all_detections, axis=1), axis=1)
            sensor_anomalies = np.mean(np.stack(all_anomalies), axis=0)
            
            # Calculate uncertainty
            fault_uncertainty = np.std(np.concatenate(all_detections, axis=1), axis=1)
            sensor_uncertainty = np.std(np.stack(all_anomalies), axis=0)
            
            self.model.disable_mc_dropout()
        else:
            # Standard inference without MC dropout
            with torch.no_grad():
                outputs = self.model(tensor_data)
                fault_probs = torch.sigmoid(outputs['fault_detection']).cpu().numpy()
                sensor_anomalies = outputs['sensor_anomalies'].cpu().numpy()
                fault_uncertainty = np.zeros_like(fault_probs)
                sensor_uncertainty = np.zeros_like(sensor_anomalies)
        
        # Calculate average anomaly score for each sensor
        avg_sensor_anomalies = sensor_anomalies.mean(axis=0)
        
        # Determine if any sensor is damaged based on threshold
        is_damaged = {
            sensor: avg_sensor_anomalies[i] >= self.sensor_thresholds[sensor]
            for i, sensor in enumerate(self.sensor_names)
        }
        
        # Calculate overall fault probability
        overall_fault_prob = float(np.mean(fault_probs))
        overall_fault = overall_fault_prob >= self.threshold
        
        # Calculate number of detected anomalies
        num_anomalies = sum(1 for sensor in is_damaged if is_damaged[sensor])
        
        # Identify the most anomalous sensors
        sorted_indices = np.argsort(-avg_sensor_anomalies)
        top_anomalies = [(self.sensor_names[idx], float(avg_sensor_anomalies[idx])) 
                          for idx in sorted_indices[:3]]
        
        # Determine damage modes based on detected anomalies
        damaged_sensors = [sensor for sensor in is_damaged if is_damaged[sensor]]
        damage_analysis = self._analyze_damage(damaged_sensors, avg_sensor_anomalies)
        
        # Compile results
        results = {
            "filename": filename,
            "overall_fault_probability": overall_fault_prob,
            "overall_fault_detected": overall_fault,
            "num_anomalies": num_anomalies,
            "sensor_anomalies": {sensor: float(avg_sensor_anomalies[i]) 
                                for i, sensor in enumerate(self.sensor_names)},
            "is_damaged": is_damaged,
            "damaged_sensors": damaged_sensors,
            "top_anomalies": top_anomalies,
            "damage_analysis": damage_analysis,
            "uncertainty": {
                "fault_uncertainty": float(np.mean(fault_uncertainty)),
                "sensor_uncertainty": {sensor: float(sensor_uncertainty.mean(axis=0)[i]) 
                                      for i, sensor in enumerate(self.sensor_names)}
            }
        }
        
        return results
    
    def _analyze_damage(self, damaged_sensors, anomaly_scores):
        """
        Analyze the damage based on the detected anomalous sensors.
        Maps sensor anomalies to specific components and failure modes.
        
        Args:
            damaged_sensors: List of sensor names detected as damaged
            anomaly_scores: Array of anomaly scores for each sensor
            
        Returns:
            List of damage assessments with component and mode information
        """
        if not damaged_sensors:
            return []
        
        damage_assessments = []
        
        # First apply specific diagnostic rules
        for rule in DIAGNOSTIC_RULES:
            # Check if rule sensors match the damaged sensors
            rule_sensors = set(rule['sensors'])
            detected_sensors = set(damaged_sensors)
            
            # If the rule is a subset of detected sensors or exact match
            if rule_sensors.issubset(detected_sensors) or rule_sensors == detected_sensors:
                damage_assessments.append({
                    "component": rule["component"],
                    "failure_mode": rule["failure_mode"],
                    "confidence": self._calculate_rule_confidence(rule_sensors, anomaly_scores),
                    "matched_rule": True,
                    "sensors": list(rule_sensors)
                })
        
        # If no rules matched, do individual sensor analysis
        if not damage_assessments:
            for sensor in damaged_sensors:
                component = SENSOR_TO_COMPONENT.get(sensor, "Unknown component")
                
                # Get sensor index for anomaly score
                sensor_idx = self.sensor_names.index(sensor)
                anomaly_score = anomaly_scores[sensor_idx]
                
                # Determine failure mode for this sensor based on correlations
                if sensor in SENSOR_MODE_CORRELATIONS:
                    mode_correlations = SENSOR_MODE_CORRELATIONS[sensor]
                    # Get the mode with highest correlation, weighted by anomaly score
                    weighted_correlations = {mode: corr * anomaly_score 
                                           for mode, corr in mode_correlations.items()}
                    
                    # Find most likely mode
                    most_likely_mode = max(weighted_correlations, 
                                          key=weighted_correlations.get)
                    confidence = weighted_correlations[most_likely_mode]
                    
                    damage_assessments.append({
                        "component": component,
                        "failure_mode": most_likely_mode,
                        "confidence": min(confidence, 0.95),  # Cap at 0.95
                        "matched_rule": False,
                        "sensors": [sensor]
                    })
                else:
                    # Fallback if no correlations defined
                    possible_modes = COMPONENT_FAILURE_MODES.get(component, ["Unknown"])
                    damage_assessments.append({
                        "component": component,
                        "failure_mode": possible_modes[0],
                        "confidence": 0.6,  # Moderate confidence as this is a fallback
                        "matched_rule": False,
                        "sensors": [sensor]
                    })
        
        # Sort by confidence
        damage_assessments.sort(key=lambda x: x["confidence"], reverse=True)
        
        return damage_assessments
    
    def _calculate_rule_confidence(self, rule_sensors, anomaly_scores):
        """Calculate confidence in a matched diagnostic rule"""
        # Convert rule sensors to indices
        indices = [self.sensor_names.index(sensor) for sensor in rule_sensors]
        
        # Average the anomaly scores for the sensors in the rule
        avg_score = np.mean([anomaly_scores[idx] for idx in indices])
        
        # Apply sigmoid-like scaling for confidence: 0.7-0.95 range
        confidence = 0.7 + 0.25 * (1 / (1 + np.exp(-10 * (avg_score - 0.6))))
        return min(confidence, 0.95)  # Cap at 0.95
    
    def visualize_results(self, results, output_dir=None):
        """Visualize anomaly detection results and save to files"""
        if output_dir is None:
            # Default output directory
            output_dir = os.path.join("output", "analysis", f"{os.path.splitext(results['filename'])[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sensor values and anomaly scores
        anomaly_scores = np.array([results['sensor_anomalies'][sensor] for sensor in self.sensor_names])
        
        # Create heatmap of anomaly scores
        plt.figure(figsize=(12, 6))
        heatmap_data = anomaly_scores.reshape(1, -1)
        sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", 
                    xticklabels=self.sensor_names, 
                    yticklabels=["Anomaly Score"],
                    cbar_kws={'label': 'Anomaly Severity'})
        plt.title(f"Sensor Anomaly Detection - {results['filename']}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"anomaly_heatmap_{results['filename']}.png"), dpi=300)
        plt.close()
        
        # Bar chart of anomaly scores
        plt.figure(figsize=(14, 8))
        bars = plt.bar(np.arange(len(self.sensor_names)), anomaly_scores)
        
        # Color bars based on threshold
        for i, bar in enumerate(bars):
            if anomaly_scores[i] >= self.sensor_threshold:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        plt.axhline(y=self.sensor_threshold, color='r', linestyle='--', label=f'Threshold ({self.sensor_threshold})')
        plt.xlabel('Sensors')
        plt.ylabel('Anomaly Score')
        plt.title(f'Sensor Anomaly Scores - {results["filename"]}')
        plt.xticks(np.arange(len(self.sensor_names)), self.sensor_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"anomaly_scores_{results['filename']}.png"), dpi=300)
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")

    def export_results_json(self, results, output_dir):
        """Export analysis results as JSON for programmatic use"""
        json_path = os.path.join(output_dir, f"results_{results['filename']}.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, cls=CustomJSONEncoder)
            
        print(f"JSON results saved to: {json_path}")

def format_results_for_display(results):
    """Format results for console display"""
    if "error" in results:
        return colored(f"Error: {results['error']}", "red")
    
    output = []
    
    # File info and overall status
    output.append(colored(f"Analysis Results for: {results['filename']}", "cyan", attrs=["bold"]))
    output.append("-" * 60)
    
    if results['overall_fault_detected']:
        output.append(colored("⚠️  FAULT DETECTED", "red", attrs=["bold"]))
    else:
        output.append(colored("✅ NO FAULT DETECTED", "green", attrs=["bold"]))
    
    output.append(f"Overall Fault Probability: {results['overall_fault_probability']:.3f}")
    output.append(f"Anomalous Sensors: {results['num_anomalies']}")
    output.append("-" * 60)
    
    # Damage analysis
    if results['damage_analysis']:
        output.append(colored("Damage Assessment:", "yellow", attrs=["bold"]))
        damage_table = []
        for i, damage in enumerate(results['damage_analysis'], 1):
            confidence_str = f"{damage['confidence']*100:.1f}%"
            if damage['confidence'] > 0.8:
                confidence_str = colored(confidence_str, "green")
            elif damage['confidence'] > 0.6:
                confidence_str = colored(confidence_str, "yellow")
            else:
                confidence_str = colored(confidence_str, "red")
                
            damage_table.append([
                i,
                damage['component'],
                damage['failure_mode'],
                confidence_str,
                ', '.join(damage['sensors'])
            ])
        
        output.append(tabulate(damage_table, 
                              headers=["#", "Component", "Failure Mode", "Confidence", "Sensors"],
                              tablefmt="grid"))
    else:
        output.append(colored("No specific damage patterns identified.", "green"))
    
    output.append("-" * 60)
    
    # Sensor anomaly details
    output.append(colored("Sensor Anomaly Details:", "yellow", attrs=["bold"]))
    
    sensor_table = []
    for sensor in results['sensor_anomalies']:
        anomaly_score = results['sensor_anomalies'][sensor]
        is_damaged = sensor in results['damaged_sensors']
        
        # Format for display
        sensor_str = colored(sensor, "red") if is_damaged else sensor
        score_str = f"{anomaly_score:.3f}"
        if is_damaged:
            score_str = colored(score_str, "red")
        status = colored("ANOMALOUS", "red") if is_damaged else "NORMAL"
        
        sensor_table.append([
            sensor_str,
            SENSOR_TO_COMPONENT.get(sensor, "Unknown"),
            score_str,
            status
        ])
    
    output.append(tabulate(sensor_table,
                          headers=["Sensor", "Component", "Anomaly Score", "Status"],
                          tablefmt="grid"))
    
    return "\n".join(output)

def process_unseen_data(dataset_path, output_format='text', output_dir=None, model_path=None, 
                      sensor_threshold=0.5, detection_threshold=DEFAULT_THRESHOLD, 
                      adaptive_threshold=True, use_mc_dropout=True, full_scan=False):
    """
    Process unseen data from a dataset path.
    
    Args:
        dataset_path: Path to .mat file or directory with multiple .mat files
        output_format: Format for results ('text', 'json', 'html', or 'all')
        output_dir: Directory to save outputs
        model_path: Path to the trained model
        sensor_threshold: Threshold for sensor anomaly detection
        detection_threshold: Threshold for fault detection
        adaptive_threshold: Whether to use adaptive thresholds
        use_mc_dropout: Whether to use Monte Carlo dropout
        full_scan: Whether to perform a more sensitive scan
        
    Returns:
        Dictionary with analysis results
    """
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_DIR, "output", "analysis", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = GearboxDamageDetector(
        model_path=model_path,
        threshold=detection_threshold,
        use_mc_dropout=use_mc_dropout,
        sensor_threshold=sensor_threshold if not full_scan else sensor_threshold * 0.8,
        adaptive_thresholds=adaptive_threshold
    )
    
    # Prepare dataset path
    if os.path.isdir(dataset_path):
        # Process all .mat files in directory
        file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                     if f.endswith('.mat')]
    else:
        # Process single file
        file_paths = [dataset_path]
    
    # Process all files
    all_results = {}
    for file_path in file_paths:
        # Process the file
        results = detector.process_file(file_path)
        
        # Format and display results
        print(format_results_for_display(results))
        
        # Generate visualizations
        detector.visualize_results(results, output_dir)
        
        # Export JSON if requested
        if output_format in ['json', 'all']:
            detector.export_results_json(results, output_dir)
        
        # Store results
        all_results[os.path.basename(file_path)] = results
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    return all_results

def main():
    """Main entry point for command line use"""
    parser = argparse.ArgumentParser(description="Test trained model on unseen gearbox data")
    
    parser.add_argument("dataset", help="Path to .mat file or directory with .mat files")
    parser.add_argument("--model", help="Path to trained model (.pth file)")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--format", choices=["text", "json", "all"], 
                      default="all", help="Output format")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                      help="Detection threshold")
    parser.add_argument("--sensor_threshold", type=float, default=0.5,
                      help="Sensor anomaly threshold")
    parser.add_argument("--adaptive", action="store_true",
                      help="Use adaptive thresholds for sensors")
    parser.add_argument("--no_mc", action="store_true",
                      help="Disable Monte Carlo dropout")
    parser.add_argument("--full_scan", action="store_true",
                      help="More sensitive scan (lower thresholds)")
    
    args = parser.parse_args()
    
    # Process dataset path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Try adding .mat extension
        if not dataset_path.endswith('.mat'):
            test_path = dataset_path + '.mat'
            if os.path.exists(test_path):
                dataset_path = test_path
            else:
                # Try looking in the unseen_data directory
                default_dir = os.path.join(BASE_DIR, "data", "unseen_data")
                test_path = os.path.join(default_dir, dataset_path + '.mat')
                if os.path.exists(test_path):
                    dataset_path = test_path
                else:
                    print(f"Error: Could not find dataset at {args.dataset}")
                    return
    
    # Process the data
    process_unseen_data(
        dataset_path=dataset_path,
        output_format=args.format,
        output_dir=args.output,
        model_path=args.model,
        sensor_threshold=args.sensor_threshold,
        detection_threshold=args.threshold,
        adaptive_threshold=args.adaptive,
        use_mc_dropout=not args.no_mc,
        full_scan=args.full_scan
    )

if __name__ == "__main__":
    main()
