"""
Damage Classifier Module

This module implements a supervised classifier to accurately identify
damage modes in wind turbine gearboxes from sensor anomaly patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Add project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.damage_mappings import (
    SENSOR_TO_COMPONENT, 
    COMPONENT_FAILURE_MODES,
    SENSOR_MODE_CORRELATIONS
)
from config.config import BASE_DIR

class DamageClassifier:
    """
    Enhanced classifier for identifying gearbox damage types
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
        
        # Initialize ensemble of classifiers for better performance
        self.sensor_clf = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.damage_type_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Map damage modes to integers and back
        self.damage_modes = None
        self.mode_to_idx = None
        self.idx_to_mode = None
        
    def _initialize_mode_mappings(self, training_modes):
        """Initialize mappings between damage modes and indices"""
        self.damage_modes = sorted(list(set(training_modes)))
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(self.damage_modes)}
        self.idx_to_mode = {idx: mode for idx, mode in enumerate(self.damage_modes)}
        
        # Save mappings
        mappings = {
            "damage_modes": self.damage_modes,
            "mode_to_idx": self.mode_to_idx,
            "idx_to_mode": self.idx_to_mode
        }
        joblib.dump(mappings, os.path.join(self.model_dir, "mode_mappings.joblib"))
    
    def fit(self, X, y, optimize=False):
        """
        Train the classifier on sensor anomaly patterns.
        
        Args:
            X: Sensor anomaly patterns [n_samples, n_sensors]
            y: Damage mode labels as strings [n_samples]
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Self
        """
        # Initialize mode mappings
        if isinstance(y[0], str):
            self._initialize_mode_mappings(y)
            # Convert string labels to indices
            y_encoded = np.array([self.mode_to_idx[mode] for mode in y])
        else:
            # Assume y is already encoded
            y_encoded = y
            
        # Scale the input data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        if optimize:
            # Hyperparameter optimization
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            
            grid_search = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy'
            )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            
            # Update classifier with best params
            self.sensor_clf = GradientBoostingClassifier(
                **best_params,
                random_state=42
            )
        
        # Train the classifier
        self.sensor_clf.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.sensor_clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        self._plot_confusion_matrix(cm, list(self.mode_to_idx.keys()))
        
        # Save the trained model
        self.save()
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict damage modes from sensor anomaly patterns.
        
        Args:
            X: Sensor anomaly patterns [n_samples, n_sensors]
            
        Returns:
            List of tuples (predicted_mode, probability_dict)
        """
        if not self.is_trained:
            # Try to load the model
            if not self.load():
                raise ValueError("Classifier not trained. Please train or load a model first.")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        pred_indices = self.sensor_clf.predict(X_scaled)
        probabilities = self.sensor_clf.predict_proba(X_scaled)
        
        # Convert indices to damage modes
        results = []
        for idx, probs in zip(pred_indices, probabilities):
            # Convert predicted index to damage mode
            predicted_mode = self.idx_to_mode[idx]
            
            # Create probability dictionary
            prob_dict = {self.idx_to_mode[i]: prob for i, prob in enumerate(probs)}
            
            results.append((predicted_mode, prob_dict))
            
        return results
    
    def save(self):
        """
        Save the trained model.
            
        Returns:
            Path to saved model directory
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save classifier and scaler
        joblib.dump(self.sensor_clf, os.path.join(self.model_dir, "sensor_clf.joblib"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        
        # Save mode mappings if they exist
        if self.damage_modes is not None:
            mappings = {
                "damage_modes": self.damage_modes,
                "mode_to_idx": self.mode_to_idx,
                "idx_to_mode": self.idx_to_mode
            }
            joblib.dump(mappings, os.path.join(self.model_dir, "mode_mappings.joblib"))
        
        print(f"Model saved to {self.model_dir}")
        return self.model_dir
    
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
        mappings_path = os.path.join(model_dir, "mode_mappings.joblib")
        
        missing_files = []
        if not os.path.exists(clf_path):
            missing_files.append("classifier")
        if not os.path.exists(scaler_path):
            missing_files.append("scaler")
        if not os.path.exists(mappings_path):
            missing_files.append("mode mappings")
            
        if missing_files:
            print(f"Missing model files: {', '.join(missing_files)}")
            return False
        
        try:
            self.sensor_clf = joblib.load(clf_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load mode mappings
            mappings = joblib.load(mappings_path)
            self.damage_modes = mappings["damage_modes"]
            self.mode_to_idx = mappings["mode_to_idx"]
            self.idx_to_mode = mappings["idx_to_mode"]
            
            self.is_trained = True
            print(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix for model evaluation"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.model_dir, "confusion_matrix.png"))
        plt.close()

def create_synthetic_training_data(ground_truth, n_synthetic=100):
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
    
    # Extract damage modes from SENSOR_MODE_CORRELATIONS for realistic training data
    sensor_names = [f'AN{i}' for i in range(3, 11)]
    
    # Get all unique damage modes for synthetic data generation
    damage_modes = set()
    for modes in COMPONENT_FAILURE_MODES.values():
        for mode in modes:
            damage_modes.add(mode)
    
    # Generate data for each ground truth pattern and for random patterns
    for dataset, sensor_status in ground_truth.items():
        # Get active sensors
        active_sensors = [sensor for sensor, status in sensor_status.items() 
                         if status > 0.5]
        
        if not active_sensors:
            continue  # Skip if no active sensors
        
        # Determine damage mode from active sensors using rules or correlations
        damage_mode = _identify_damage_mode_from_sensors(active_sensors)
        
        # Generate base anomaly pattern
        base_pattern = np.array([sensor_status.get(sensor, 0) for sensor in sensor_names])
        
        # Generate synthetic variations
        for _ in range(n_synthetic):
            # Add realistic variations with correlation between sensors
            noise = np.random.normal(0, 0.1, base_pattern.shape)
            
            # Ensure correlated noise between related sensors
            for i in range(len(noise) - 1):
                # Add some correlation between adjacent sensors
                if np.random.rand() < 0.4:
                    noise[i+1] = noise[i] * 0.7 + noise[i+1] * 0.3
            
            # Generate variation with controlled noise
            variation = np.clip(base_pattern + noise, 0, 1)
            
            # For damaged sensors, ensure they have high enough anomaly scores
            for i, sensor in enumerate(sensor_names):
                if sensor in active_sensors:
                    # Ensure damaged sensors have high enough anomaly score
                    variation[i] = max(variation[i], 0.5 + np.random.random() * 0.3)
                else:
                    # Ensure healthy sensors have low enough anomaly score
                    variation[i] = min(variation[i], 0.3 + np.random.random() * 0.2)
            
            X.append(variation)
            y.append(damage_mode)
    
    # Generate additional synthetic samples for all possible damage modes
    for mode in damage_modes:
        # Find sensors associated with this mode based on correlations
        relevant_sensors = []
        for sensor, modes in SENSOR_MODE_CORRELATIONS.items():
            if mode in modes and modes[mode] > 0.6:  # Only use strong correlations
                relevant_sensors.append(sensor)
        
        if not relevant_sensors:
            continue
            
        # Generate pattern
        for _ in range(n_synthetic // 2):  # Generate fewer samples for these
            pattern = np.random.uniform(0.1, 0.3, len(sensor_names))  # Base low anomaly
            
            # Increase anomaly for relevant sensors
            for sensor in relevant_sensors:
                if sensor in sensor_names:
                    idx = sensor_names.index(sensor)
                    pattern[idx] = np.random.uniform(0.6, 0.9)  # High anomaly
            
            X.append(pattern)
            y.append(mode)
    
    return np.array(X), np.array(y)

def _identify_damage_mode_from_sensors(active_sensors):
    """
    Identify the most likely damage mode from a list of active sensors.
    
    Args:
        active_sensors: List of active sensor names
        
    Returns:
        Most likely damage mode
    """
    # Import damage mappings
    from src.data_processing.damage_mappings import DIAGNOSTIC_RULES
    
    # First check if any diagnostic rule matches
    for rule in DIAGNOSTIC_RULES:
        if set(rule['sensors']).issubset(set(active_sensors)):
            return rule['failure_mode']
    
    # If no direct rule matches, check components and their common failure modes
    if len(active_sensors) == 1:
        sensor = active_sensors[0]
        component = SENSOR_TO_COMPONENT.get(sensor)
        if component in COMPONENT_FAILURE_MODES:
            return COMPONENT_FAILURE_MODES[component][0]  # Return first/most common mode
    
    # Fallback to most common mode
    return "Scuffing"

def main():
    """Main function for training and testing the classifier"""
    from src.data_processing.ground_truth import SENSOR_GROUND_TRUTH
    
    # Create synthetic training data
    print("Creating synthetic training data...")
    X_train, y_train = create_synthetic_training_data(SENSOR_GROUND_TRUTH, n_synthetic=100)
    
    print(f"Generated {len(X_train)} training samples")
    
    # Train classifier
    print("Training damage classifier...")
    classifier = DamageClassifier()
    classifier.fit(X_train, y_train)
    
    # Test prediction
    print("\nTesting prediction on sample data...")
    test_pattern = np.array([0.8, 0.7, 0.2, 0.3, 0.1, 0.2, 0.9, 0.3])  # High AN3, AN4, AN9
    predictions = classifier.predict(test_pattern.reshape(1, -1))
    
    print(f"Test Pattern: {test_pattern}")
    print(f"Predicted Mode: {predictions[0][0]}")
    print("Probabilities:")
    for mode, prob in sorted(predictions[0][1].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {mode}: {prob:.3f}")

if __name__ == "__main__":
    main()
