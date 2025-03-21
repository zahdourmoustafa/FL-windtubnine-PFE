"""
Utilities for training and using the supervised damage classifier.
"""
import os
import numpy as np
from src.evaluation.damage_classifier import DamageClassifier, create_synthetic_training_data
from src.data_processing.ground_truth import SENSOR_GROUND_TRUTH
from config.config import BASE_DIR

def train_and_save_classifier():
    """
    Train a damage classifier using ground truth data and save it
    """
    # Create synthetic training data from ground truth
    X_train, y_train = create_synthetic_training_data(SENSOR_GROUND_TRUTH, n_synthetic=10)
    
    # Create and train classifier
    model_dir = os.path.join(BASE_DIR, "models", "damage_classifiers")
    os.makedirs(model_dir, exist_ok=True)
    
    clf = DamageClassifier(model_dir=model_dir)
    clf.fit(X_train, y_train)
    
    print(f"Damage classifier trained and saved to {model_dir}")
    return clf

def load_or_train_classifier():
    """
    Load a pre-trained classifier if available, otherwise train a new one
    """
    model_dir = os.path.join(BASE_DIR, "models", "damage_classifiers")
    clf = DamageClassifier(model_dir=model_dir)
    
    if clf.load(model_dir):
        print("Loaded pre-trained damage classifier")
        return clf
    else:
        print("No pre-trained damage classifier found, training a new one...")
        return train_and_save_classifier()

def evaluate_classifier_accuracy():
    """
    Evaluate the accuracy of the damage classifier
    """
    # Create test data
    X_test, y_true = create_synthetic_training_data(SENSOR_GROUND_TRUTH, n_synthetic=3)
    
    # Load classifier
    clf = load_or_train_classifier()
    
    # Make predictions
    y_pred = np.array([pred[0] for pred in clf.sensor_clf.predict(X_test)])
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    
    print(f"Damage classifier accuracy: {accuracy:.4f}")
    return accuracy
