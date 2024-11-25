from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

import json
import os
from io import BytesIO

import numpy as np
from PIL import Image

from keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_NAME = 'ai_id_vehicle.keras'
MODEL_VERSION = '1'
MODEL_DIR = os.path.splitext(MODEL_NAME)[0]

def init():

  global model
  global MODEL_NAME
  global MODEL_VERSION
  global MODEL_DIR
  
  # Get the current working directory
  current_directory = os.getcwd()
  print("Current Directory:", current_directory)

  # List the contents of the current directory
  contents = os.listdir(current_directory)
  print("Contents of the Directory:")
  for item in contents:
    print(item)
    
  model_name = MODEL_NAME # os.getenv('MODEL_NAME')
  model_version = MODEL_VERSION # os.getenv('MODEL_VERSION')
  model_label = os.path.splitext(model_name)[0]
  model_path = os.path.join(MODEL_DIR, model_version, model_name)

  # Check if the file exists
  if os.path.exists(model_path):
    print(f"File exists {model_path}.")
  else:
    print(f"File does not exist {model_path}.")

  print(f"model_path: {model_path}")
  
  model = load_model('ai_id_vehicle.keras') #(model_path)
  if model:
    print('Model {model_label} at path {model_path} loaded')
    return True
  return False

def prepare_image(image):

    dimension = 224
    shape = (dimension, dimension)
    # Resize the image, convert to RGB
    image = image.resize(shape, resample=Image.BILINEAR)
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Reshape to add batch dimension: (1, 224, 224, 3)
    image = image.reshape(1, dimension, dimension, 3)
    return image

app = FastAPI(title="AI Vehicle ID")
model = None
init_success = init()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    global init_success
    if init_success:
        return {"status": "ok"}
    else:
        return {"status": "not ready"}

@app.post("/predict", response_model=None)
async def predict(image: UploadFile):
    try:
        # Read the image bytes from the uploaded file
        image_bytes = await image.read()
        # Open the image using PIL
        pil_image = Image.open(BytesIO(image_bytes))

        # Preprocess the image for the model
        prepared_image = prepare_image(pil_image)

        # Make a prediction using the pre-trained model
        prediction = model.predict(prepared_image)

        # Assuming you have a list `category_tags` to map indices to categories
        category_tags = ["Negative","cab", "convertible", "coupe", "hatchback", "minivan", "sedan", "suv", "truck", "van", "wagon"]

        # Extract the prediction results
        index_max_prob = np.argmax(prediction[0])
        predicted_category = category_tags[index_max_prob]
        prediction_probability = float(prediction[0][index_max_prob])
        predictions_dict = {tag: float(prob) for tag, prob in zip(category_tags, prediction[0])}

        # Create a result dictionary and include the form data
        result_dict = {
            "message": "Image and form data received and processed successfully!",
            "prediction": {
                "category": predicted_category,
                "probability": prediction_probability
            },
            "predictions": predictions_dict
        }

        return JSONResponse(content=result_dict)

    except Exception as e:

        return JSONResponse(content={"error": str(e), "status": "500"})
        
