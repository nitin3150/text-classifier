from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
from typing import List
import os

# Define the FastAPI app
app = FastAPI(title="StackOverflow Text Classifier")
origins = [
    "https://cedar-style-444500-q5.uc.r.appspot.com","*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontend when visiting the root URL
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("index.html", "r") as file:
        return file.read()

# Load the pre-trained model
try:
    model = joblib.load("new_model.joblib")  # Use the pre-trained model
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Define the mapping
mapping = {
    1: 'wordpress',
    2: 'oracle',
    3: 'svn',
    4: 'apache',
    5: 'excel',
    6: 'matlab',
    7: 'visual-studio',
    8: 'cocoa',
    9: 'osx',
    10: 'bash',
    11: 'spring',
    12: 'hibernate',
    13: 'scala',
    14: 'sharepoint',
    15: 'ajax',
    16: 'qt',
    17: 'drupal',
    18: 'linq',
    19: 'haskell',
    20: 'magento'
}

# Define the request schema
class PredictionRequest(BaseModel):
    texts: List[str]  # List of texts to classify

# Define the response schema
class PredictionResponse(BaseModel):
    predictions: List[str]  # Predicted categories

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Get the input texts
        texts = request.texts

        # Validate input
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided for prediction.")

        # Predict using the loaded model
        predictions = model.predict(texts)

        # Map predictions to categories
        mapped_predictions = [mapping.get(pred, "unknown") for pred in predictions]

        # Return predictions as a list
        return PredictionResponse(predictions=mapped_predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
