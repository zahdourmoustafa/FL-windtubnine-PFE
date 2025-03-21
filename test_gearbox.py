#!/usr/bin/env python3
"""
Wind Turbine Gearbox Testing Tool

This script provides a simple command-line interface for testing
gearbox data for faults using the trained machine learning model.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from src.evaluation.test_unseen_data import process_unseen_data, GearboxDamageDetector
from src.utils.damage_classifier_utils import load_or_train_classifier
from config.config import BASE_DIR, DEFAULT_THRESHOLD

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test wind turbine gearbox data for faults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_gearbox.py D1.mat
  python test_gearbox.py data/unseen_data/H5.mat
  python test_gearbox.py D3 --threshold 0.4 --adaptive
  python test_gearbox.py D2 --full_scan --output ./my_analysis
        """
    )
    
    # Required arguments
    parser.add_argument(
        "dataset", 
        help="Path to .mat file or dataset name (e.g., 'D1', 'H2')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=DEFAULT_THRESHOLD,
        help="Detection threshold (default: from config)"
    )
    
    parser.add_argument(
        "--sensor_threshold",
        type=float,
        default=0.5,
        help="Threshold for sensor anomaly detection (default: 0.5)"
    )
    
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive thresholds for sensors"
    )
    
    parser.add_argument(
        "--no_mc",
        action="store_true",
        help="Disable Monte Carlo dropout for uncertainty estimation"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--model",
        help="Path to custom model file"
    )
    
    parser.add_argument(
        "--full_scan",
        action="store_true",
        help="Enable comprehensive scan to detect subtle damages"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "all"],
        default="all",
        help="Output format (default: all)"
    )
    
    parser.add_argument(
        "--no_classify",
        action="store_true",
        help="Disable the supervised classifier"
    )
    
    return parser.parse_args()

def main():
    """Main function to process command line arguments and run analysis"""
    args = parse_args()
    
    # Process dataset path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Check different variations to help the user
        variations = [
            dataset_path,
            dataset_path + '.mat',
            os.path.join(BASE_DIR, "data", "unseen_data", dataset_path),
            os.path.join(BASE_DIR, "data", "unseen_data", dataset_path + '.mat')
        ]
        
        found = False
        for path in variations:
            if os.path.exists(path):
                dataset_path = path
                found = True
                break
                
        if not found:
            print(f"Error: Could not find dataset '{args.dataset}'.")
            print("Please provide a valid path or dataset name.")
            print("Expected locations:")
            for path in variations:
                print(f"  - {path}")
            return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join(BASE_DIR, "output", "analysis", f"{dataset_name}_{timestamp}")
    
    # Load damage classifier if requested
    damage_clf = None
    if not args.no_classify:
        try:
            print("Loading damage classifier...")
            damage_clf = load_or_train_classifier()
            print("Damage classifier loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load damage classifier: {e}")
            print("Continuing without supervised classification.")
    
    # Print analysis configuration
    print("\nGearbox Analysis Configuration:")
    print(f"  Dataset: {os.path.basename(dataset_path)}")
    print(f"  Detection Threshold: {args.threshold}")
    print(f"  Sensor Threshold: {args.sensor_threshold}")
    print(f"  Adaptive Thresholds: {'Enabled' if args.adaptive else 'Disabled'}")
    print(f"  Monte Carlo Dropout: {'Disabled' if args.no_mc else 'Enabled'}")
    print(f"  Full Scan Mode: {'Enabled' if args.full_scan else 'Disabled'}")
    print(f"  Output Format: {args.format}")
    print(f"  Output Directory: {output_dir}")
    if args.model:
        print(f"  Custom Model: {args.model}")
    if damage_clf:
        print("  Damage Classifier: Loaded")
    print("\nStarting analysis...")
    
    try:
        # Run the analysis
        results = process_unseen_data(
            dataset_path=dataset_path,
            output_format=args.format,
            output_dir=output_dir,
            model_path=args.model,
            sensor_threshold=args.sensor_threshold,
            detection_threshold=args.threshold,
            adaptive_threshold=args.adaptive,
            use_mc_dropout=not args.no_mc,
            full_scan=args.full_scan
        )
        
        # Apply supervised classifier if available
        if damage_clf and not args.no_classify and results:
            print("\nApplying supervised damage classification...")
            for filename, result in results.items():
                # Skip entries with errors
                if "error" in result:
                    continue
                    
                # Extract sensor anomalies as feature vector
                sensor_anomalies = []
                for sensor in [f'AN{i}' for i in range(3, 11)]:
                    sensor_anomalies.append(result['sensor_anomalies'][sensor])
                
                # Only analyze if there are damaged sensors
                if result['damaged_sensors']:
                    # Get predictions from classifier
                    predictions = damage_clf.predict(
                        [sensor_anomalies]  # Needs to be a 2D array
                    )
                    
                    if predictions:
                        # Add classifier predictions to results
                        predicted_mode, mode_probs = predictions[0]
                        result['classifier_prediction'] = {
                            'primary_mode': predicted_mode,
                            'probabilities': {k: float(v) for k, v in mode_probs.items()}
                        }
                        
                        # Print the prediction
                        print(f"\nSupervised classifier results for {filename}:")
                        print(f"  Primary damage mode: {predicted_mode}")
                        print("  Mode probabilities:")
                        for mode, prob in sorted(mode_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
                            print(f"    {mode}: {prob:.3f}")
        
        print(f"\nAnalysis complete. Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
