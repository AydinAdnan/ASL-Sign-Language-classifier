from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pickle
import mediapipe as mp
import io
from PIL import Image
import uvicorn
from typing import List, Dict
import os
from datetime import datetime

app = FastAPI(title="ASL Sign Language Recognition API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ASLPredictor:
    def __init__(self, model_path='asl_model.pkl'):
        self.model_path = model_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.model = None
        self.label_encoder = None
        self.load_model()
        
        # For sentence building
        self.sentence_words = []
        self.last_prediction_time = None
        self.prediction_cooldown = 1.0  # seconds
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found!")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_hand_landmarks(self, image):
        """Extract hand landmarks from image"""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert RGB to BGR for OpenCV (MediaPipe expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Process the image
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract x, y coordinates for all 21 landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                return np.array(landmarks)
            else:
                return None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def predict(self, image):
        """Predict sign from image"""
        if self.model is None or self.label_encoder is None:
            return None, 0.0, []
        
        landmarks = self.extract_hand_landmarks(image)
        
        if landmarks is None:
            return None, 0.0, []
        
        # Reshape for prediction
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(landmarks)[0]
        prediction_proba = self.model.predict_proba(landmarks)[0]
        
        # Get class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(prediction_proba)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_predictions = []
        for idx in top_3_indices:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            prob = prediction_proba[idx]
            top_predictions.append({"class": class_name, "confidence": float(prob)})
        
        return predicted_class, float(confidence), top_predictions
    
    def add_to_sentence(self, predicted_sign, confidence_threshold=0.8):
        """Add predicted sign to sentence if confidence is high enough"""
        current_time = datetime.now()
        
        # Check if enough time has passed and confidence is high
        if (self.last_prediction_time is None or 
            (current_time - self.last_prediction_time).total_seconds() > self.prediction_cooldown):
            
            if confidence_threshold <= 0.8:  # You can adjust this threshold
                self.sentence_words.append(predicted_sign)
                self.last_prediction_time = current_time
                return True
        return False
    
    def get_sentence(self):
        """Get current sentence"""
        return " ".join(self.sentence_words)
    
    def clear_sentence(self):
        """Clear current sentence"""
        self.sentence_words = []
        self.last_prediction_time = None
    
    def remove_last_word(self):
        """Remove last word from sentence"""
        if self.sentence_words:
            self.sentence_words.pop()

# Initialize the predictor
predictor = ASLPredictor()

@app.get("/")
async def root():
    return {"message": "ASL Sign Language Recognition API", "status": "running"}

@app.get("/health")
async def health_check():
    model_loaded = predictor.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "classes": list(predictor.label_encoder.classes_) if predictor.label_encoder else []
    }

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    """Predict sign language from uploaded image"""
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Make prediction
        predicted_class, confidence, top_predictions = predictor.predict(image_np)
        
        if predicted_class is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No hand landmarks detected in the image"}
            )
        
        return {
            "predicted_sign": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-and-add")
async def predict_and_add_to_sentence(file: UploadFile = File(...), auto_add: bool = True):
    """Predict sign and optionally add to sentence"""
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Make prediction
        predicted_class, confidence, top_predictions = predictor.predict(image_np)
        
        if predicted_class is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No hand landmarks detected in the image"}
            )
        
        # Add to sentence if auto_add is True and confidence is high
        added_to_sentence = False
        if auto_add:
            added_to_sentence = predictor.add_to_sentence(predicted_class, confidence)
        
        return {
            "predicted_sign": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "added_to_sentence": added_to_sentence,
            "current_sentence": predictor.get_sentence(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/add-word")
async def add_word_to_sentence(word: str):
    """Manually add a word to the sentence"""
    predictor.sentence_words.append(word.lower())
    return {
        "added_word": word,
        "current_sentence": predictor.get_sentence()
    }

@app.get("/sentence")
async def get_current_sentence():
    """Get the current sentence"""
    return {
        "sentence": predictor.get_sentence(),
        "word_count": len(predictor.sentence_words),
        "words": predictor.sentence_words
    }

@app.post("/sentence/clear")
async def clear_sentence():
    """Clear the current sentence"""
    predictor.clear_sentence()
    return {"message": "Sentence cleared", "sentence": ""}

@app.post("/sentence/remove-last")
async def remove_last_word():
    """Remove the last word from the sentence"""
    predictor.remove_last_word()
    return {
        "message": "Last word removed",
        "current_sentence": predictor.get_sentence()
    }

@app.get("/classes")
async def get_available_classes():
    """Get list of available sign classes"""
    if predictor.label_encoder is None:
        return {"error": "Model not loaded"}
    
    return {
        "classes": list(predictor.label_encoder.classes_),
        "total_classes": len(predictor.label_encoder.classes_)
    }

if __name__ == "__main__":
    # Check if model file exists
    if not os.path.exists("asl_model.pkl"):
        print("Warning: asl_model.pkl not found. Please train the model first.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)