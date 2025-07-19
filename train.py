import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ASLRecognitionSystem:
    def __init__(self, dataset_path='asl_dataset'):
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_data = []
        self.labels = []
        
    def extract_hand_landmarks(self, image_path):
        """Extract hand landmarks from an image using MediaPipe"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Take first hand
                
                # Extract x, y coordinates for all 21 landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                return np.array(landmarks)
            else:
                return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load dataset and extract features from all images"""
        print("Loading dataset and extracting hand landmarks...")
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path {self.dataset_path} does not exist!")
            return False
        
        class_folders = [f for f in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        if not class_folders:
            print("No class folders found in dataset!")
            return False
        
        print(f"Found {len(class_folders)} classes: {class_folders}")
        
        total_processed = 0
        successful_extractions = 0
        
        for class_name in class_folders:
            class_path = os.path.join(self.dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Processing class '{class_name}': {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                landmarks = self.extract_hand_landmarks(img_path)
                
                if landmarks is not None:
                    self.feature_data.append(landmarks)
                    self.labels.append(class_name)
                    successful_extractions += 1
                
                total_processed += 1
                
                if total_processed % 50 == 0:
                    print(f"Processed {total_processed} images...")
        
        print(f"\nDataset loading complete!")
        print(f"Total images processed: {total_processed}")
        print(f"Successful landmark extractions: {successful_extractions}")
        print(f"Success rate: {successful_extractions/total_processed*100:.2f}%")
        
        # Convert to numpy arrays
        self.feature_data = np.array(self.feature_data)
        self.labels = np.array(self.labels)
        
        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        print(f"Feature data shape: {self.feature_data.shape}")
        print(f"Labels shape: {self.labels_encoded.shape}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Show class distribution
        class_counts = Counter(self.labels)
        print(f"Class distribution: {dict(class_counts)}")
        
        return True
    
    def save_features(self, filepath='asl_features.pkl'):
        """Save extracted features to pickle file"""
        data_to_save = {
            'features': self.feature_data,
            'labels': self.labels_encoded,
            'label_encoder': self.label_encoder,
            'class_names': list(self.label_encoder.classes_)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Features saved to {filepath}")
    
    def load_features(self, filepath='asl_features.pkl'):
        """Load features from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.feature_data = data['features']
            self.labels_encoded = data['labels']
            self.label_encoder = data['label_encoder']
            
            print(f"Features loaded from {filepath}")
            print(f"Feature data shape: {self.feature_data.shape}")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            return True
        except FileNotFoundError:
            print(f"File {filepath} not found!")
            return False
    
    def train_model(self):
        """Train Random Forest model with GridSearchCV optimization"""
        print("Training Random Forest model with GridSearchCV...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_data, self.labels_encoded, 
            test_size=0.2, random_state=42, stratify=self.labels_encoded
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(random_state=42)
        
        # Perform GridSearchCV
        print("Performing GridSearchCV (this may take a while)...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Calculate mAP-like score (mean accuracy per class)
        class_accuracies = []
        for class_idx in range(len(self.label_encoder.classes_)):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracies.append(class_acc)
        
        map_score = np.mean(class_accuracies)
        print(f"mAP Score (mean accuracy per class): {map_score:.4f}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        # Get unique labels in test set to avoid mismatch
        unique_test_labels = np.unique(y_test)
        test_class_names = self.label_encoder.inverse_transform(unique_test_labels)
        
        print(classification_report(
            y_test, y_pred, 
            labels=unique_test_labels,
            target_names=test_class_names,
            zero_division=0
        ))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'map_score': map_score
        }
        
        with open('asl_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved to asl_model.pkl")
        
        return accuracy, map_score
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_trained_model(self, filepath='asl_model.pkl'):
        """Load trained model from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            
            print(f"Model loaded from {filepath}")
            print(f"Model accuracy: {data.get('accuracy', 'N/A')}")
            print(f"mAP Score: {data.get('map_score', 'N/A')}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found!")
            return False
    
    def predict_image(self, image_path):
        """Predict sign language from a single image"""
        if self.model is None:
            print("No model loaded! Please train or load a model first.")
            return None
        
        # Extract landmarks from the image
        landmarks = self.extract_hand_landmarks(image_path)
        
        if landmarks is None:
            print("No hand landmarks detected in the image!")
            return None
        
        # Reshape for prediction
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(landmarks)[0]
        prediction_proba = self.model.predict_proba(landmarks)[0]
        
        # Get class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(prediction_proba)
        
        print(f"Predicted sign: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_3_indices, 1):
            class_name = self.label_encoder.inverse_transform([idx])[0]
            prob = prediction_proba[idx]
            print(f"{i}. {class_name}: {prob:.4f}")
        
        # Display image with prediction
        self.display_prediction(image_path, predicted_class, confidence)
        
        return predicted_class, confidence
    
    def display_prediction(self, image_path, predicted_class, confidence):
        """Display image with prediction overlay"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw hand landmarks
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        # Display image with prediction
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.title(f'Predicted Sign: {predicted_class} (Confidence: {confidence:.3f})', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Initialize the ASL recognition system
    asl_system = ASLRecognitionSystem('asl_dataset')
    
    print("=== ASL Sign Language Recognition System ===\n")
    
    # Step 1: Load dataset and extract features
    print("Step 1: Loading dataset and extracting hand landmarks...")
    if not asl_system.load_dataset():
        print("Failed to load dataset!")
        return
    
    # Step 2: Save features to pickle file
    print("\nStep 2: Saving features to pickle file...")
    asl_system.save_features('asl_features.pkl')
    
    # Step 3: Train model with GridSearchCV
    print("\nStep 3: Training Random Forest model with GridSearchCV...")
    accuracy, map_score = asl_system.train_model()
    
    print(f"\n=== Training Results ===")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"mAP Score: {map_score:.4f}")
    
    # Step 4: Test with a sample image
    print("\n=== Testing with Sample Image ===")
    
    # You can test with any image from your dataset or a new image
    # Example: test with the first image found in the dataset
    test_image_path = None
    for class_folder in os.listdir('asl_dataset'):
        class_path = os.path.join('asl_dataset', class_folder)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image_path = os.path.join(class_path, images[0])
                break
    
    if test_image_path:
        print(f"Testing with image: {test_image_path}")
        asl_system.predict_image(test_image_path)
    else:
        print("No test image found. Please provide an image path to test.")
        print("Usage: asl_system.predict_image('path/to/your/image.jpg')")

if __name__ == "__main__":
    main()

# Example usage for testing with your own image:
# asl_system = ASLRecognitionSystem()
# asl_system.load_trained_model('asl_model.pkl')  # Load pre-trained model
# asl_system.predict_image('path/to/your/test/image.jpg')