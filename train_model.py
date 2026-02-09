import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features

def load_dataset(data_dir):
    """
    Load all images and extract features
    
    Returns:
        X: Feature matrix (each row = one image's features)
        y: Labels (disease names)
    """
    
    features_list = []
    labels_list = []
   
    disease_folders = [f for f in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, f))]
    
    print("Found disease classes:", disease_folders)
    
    for disease_class in disease_folders:
        disease_path = os.path.join(data_dir, disease_class)
      
        image_files = [f for f in os.listdir(disease_path) 
                      if f.endswith(('.jpg', '.JPG', '.jpeg', '.png'))]
        
        print(f"Processing {disease_class}: {len(image_files)} images")
        
        # Process each image
        for img_file in image_files[:100]:  # Limit to 100 images per class for speed
            img_path = os.path.join(disease_path, img_file)
            
            try:
                # Extract features
                features = extract_features(img_path)
                features_list.append(features)
                labels_list.append(disease_class)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nDataset loaded: {X.shape[0]} images, {X.shape[1]} features per image")
    
    return X, y


def train_and_save_model():
    """
    Train ML model and save it
    """
    
    print("=" * 50)
    print("STEP 1: Loading Dataset...")
    print("=" * 50)
    
    X, y = load_dataset('data/plantvillage')
    
    print("\n" + "=" * 50)
    print("STEP 2: Splitting Data (80% train, 20% test)...")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n" + "=" * 50)
    print("STEP 3: Training Random Forest Model...")
    print("=" * 50)
    
    # Random Forest: Creates many decision trees and combines their predictions
    # n_estimators = number of trees
    # Good for: Handling complex patterns, resistant to overfitting
    
    model = RandomForestClassifier(
        n_estimators=100,      # 100 decision trees
        max_depth=20,          # Limit tree depth to prevent overfitting
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    print("\n" + "=" * 50)
    print("STEP 4: Evaluating Model...")
    print("=" * 50)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 50)
    print("STEP 5: Saving Model...")
    print("=" * 50)
    
    # Create models folder if not exists
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/plant_model.pkl')
    
    # Save class names for later use
    class_names = list(np.unique(y))
    joblib.dump(class_names, 'models/class_names.pkl')
    
    print("✓ Model saved to models/plant_model.pkl")
    print("✓ Training complete!")
    
    return model, accuracy


if __name__ == "__main__":
    train_and_save_model()