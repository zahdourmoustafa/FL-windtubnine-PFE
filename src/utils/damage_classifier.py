"""
Damage Classifier Module

Provides supervised classification capabilities to improve damage mode detection
based on sensor anomaly patterns.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from config.config import BASE_DIR

class DamageClassifier:
    """
    Supervised classifier for identifying gearbox damage types
    based on sensor anomaly patterns.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the damage classifier.
        
        Args:
            model_dir: Directory where model files are stored
        """
        if model_dir is None:
            model_dir = os.path.join(BASE_DIR, "models", "damage_classifiers")
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize classifier and scaler
        self.sensor_clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y):
        """
        Train the classifier on sensor anomaly patterns.
        
        Args:
            X: Sensor anomaly patterns [n_samples, n_sensors]
            y: Damage mode labels [n_samples]
            
        Returns:
            Self
        """
        # Scale the input data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the classifier
        self.sensor_clf.fit(X_scaled, y)
        
        # Save the trained model
        self.save(self.model_dir)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict damage modes from sensor anomaly patterns.
        
        Args:
            X: Sensor anomaly patterns [n_samples, n_sensors]
            
        Returns:
            Array of predicted damage modes and probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Please train or load a model first.")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.sensor_clf.predict(X_scaled)
        probabilities = self.sensor_clf.predict_proba(X_scaled)
        
        # Return both predictions and probabilities
        return list(zip(predictions, probabilities))
    
    def save(self, model_dir=None):
        """
        Save the trained model.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model directory
        """
        if model_dir is None:
            model_dir = self.model_dir
            
        os.makedirs(model_dir, exist_ok=True)
        
        # Save classifier and scaler
        joblib.dump(self.sensor_clf, os.path.join(model_dir, "sensor_clf.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        
        print(f"Model saved to {model_dir}")
        return model_dir
    
    def load(self, model_dir=None):
        """
        Load a trained model.
        
        Args:
            model_dir: Directory with saved model files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if model_dir is None:
            model_dir = self.model_dir
            
        clf_path = os.path.join(model_dir, "sensor_clf.joblib")
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        
        if not (os.path.exists(clf_path) and os.path.exists(scaler_path)):
            print(f"Model files not found in {model_dir}")
            return False
        
        try:
            self.sensor_clf = joblib.load(clf_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def create_synthetic_training_data(ground_truth, n_synthetic=10):
    """
    Create synthetic training data based on ground truth patterns.
    
    Args:
        ground_truth: Dictionary mapping dataset names to sensor statuses
        n_synthetic: Number of synthetic examples per pattern
        
    Returns:
        X: Sensor anomaly patterns [n_samples, n_sensors]
        y: Damage mode labels [n_samples]
    """
    X = []
    y = []
    
    # Import damage mappings
    from src.data_processing.damage_mappings import SENSOR_MODE_CORRELATIONS
    
    # Extract damage modes
    all_damage_modes = set()
    for sensor, mode_dict in SENSOR_MODE_CORRELATIONS.items():
        for mode in mode_dict:
            all_damage_modes.add(mode)
    
    # Map damage modes to integers
    mode_to_int = {mode: i for i, mode in enumerate(sorted(all_damage_modes))}
    
    # Use ground truth patterns to generate synthetic data
    for dataset, sensor_status in ground_truth.items():
        # Create base pattern from ground truth
        base_pattern = np.array([sensor_status[f'AN{i}'] for i in range(3, 11)])
        
        # Get most likely damage mode for this pattern
        damage_mode = identify_damage_mode(sensor_status)
        if damage_mode not in mode_to_int:
            continue  # Skip if damage mode not in our mapping
        
        mode_label = mode_to_int[damage_mode]
        
        # Generate synthetic variations
        for _ in range(n_synthetic):
            # Add realistic variations
            noise = np.random.normal(0, 0.1, base_pattern.shape)
            variation = np.clip(base_pattern + noise, 0, 1)
            
            X.append(variation)
            y.append(mode_label)
    
    return np.array(X), np.array(y)

def identify_damage_mode(sensor_status):
    """
    Identify the most likely damage mode from a sensor status dictionary.
    
    Args:
        sensor_status: Dictionary mapping sensor names to damage status (0-1)
        
    Returns:
        Most likely damage mode
    """
    # Import damage mappings
    from src.data_processing.damage_mappings import SENSOR_MODE_CORRELATIONS, DIAGNOSTIC_RULES
    
    # First check if any diagnostic rule matches
    active_sensors = [sensor for sensor, status in sensor_status.items() if status > 0.5]
    
    for rule in DIAGNOSTIC_RULES:
        if set(rule['sensors']).issubset(set(active_sensors)):
            return rule['failure_mode']
    
    # If no rule matches, find most likely mode based on correlations
    mode_scores = {}
    
    for sensor, status in sensor_status.items():
        if status > 0.5 and sensor in SENSOR_MODE_CORRELATIONS:
            for mode, correlation in SENSOR_MODE_CORRELATIONS[sensor].items():
                if mode not in mode_scores:
                    mode_scores[mode] = 0
                mode_scores[mode] += correlation * status
    
    # Return the mode with highest score, or "Unknown" if no match
    if mode_scores:
        return max(mode_scores.items(), key=lambda x: x[1])[0]
    else:
        return "Unknown"
