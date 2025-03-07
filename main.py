# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import bcrypt
from typing import List, Optional
import os
import shutil
from firestore_db import get_firestore_client
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
import traceback
from geopy.distance import geodesic
import requests
from textblob import TextBlob
# from chat import init_chat

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat module
# socket_manager = init_chat(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Db connection
db = get_firestore_client()

# Google Maps Configs
GOOGLE_API_KEY = 'AIzaSyDAsJYZSQ92_NQAz9kiSpW1XpyuCxRl_uI'
GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


# Load MOdels Health
MODEL_HEALTH = joblib.load('model_health_random_f_classifier.joblib')
LABEL_ENCODER =  joblib.load('label_encoder_health_status.joblib')

# Load Milk Quality Checker
try:
    MODEL_ML_QUALITY = joblib.load("milk_model_dt.joblib")  
    MODEL_ML_FORECAST = joblib.load("arima_milk_production_model.joblib")
except FileNotFoundError:
    raise Exception("Model file not found. Ensure the decision_tree_model.joblib file is in the same directory.")


# Load Pest detection Models
# Load Pest detection Models
MODEL_PESTS = load_model("model_pests_detect.h5")
CLASS_PESTS = ['Mastitis', ' Tick Infestation', 'Dermatophytosis (RINGWORM)', 'Fly Strike (MYIASIS)', 'Foot and Mouth disease', 'Lumpy Skin', 'Black Quarter (BQ)', 'Parasitic Mange']



class User(BaseModel):
    username: str
    full_name: str
    email:str
    contact: str
    password: str
    nic: str

class LoginUser(BaseModel):
    username: str
    password: str

class FaceID(BaseModel):
    username: str

users_db = {}

@app.post("/register")
async def register_user(user: User):
    user_ref = db.collection("users").document(user.username)
    if user_ref.get().exists:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Hash the password before storing it
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_data = user.dict()
    user_data["password"] = hashed_password.decode('utf-8')

    user_ref.set(user_data)
    return {"message": "User registered successfully", "user": user_data}

@app.post("/login")
async def login_user(user: LoginUser):
    user_ref = db.collection("users").document(user.username)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data = user_doc.to_dict()
    
    # Check the hashed password
    if not bcrypt.checkpw(user.password.encode('utf-8'), user_data["password"].encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data.pop("password")  # Remove the password field from the response

    return {"message": "Login successful", "user": user_data}


# Cattle Details
class Cattle(BaseModel):
    name: str
    breed: str
    birth: str
    health: str
    status: str
    image: str
    owner: str

@app.post("/cattle")
async def create_cattle(cattle: Cattle):
    cattle_ref = db.collection("cattle").document()
    cattle_data = cattle.dict()
    cattle_ref.set(cattle_data)
    return {"message": "Cattle added successfully", "id": cattle_ref.id}

@app.get("/cattle", response_model=List[dict])
async def get_all_cattle():
    cattle_docs = db.collection("cattle").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in cattle_docs]

@app.get("/cattle/owner/{owner}", response_model=List[dict])
async def get_cattle_by_owner(owner: str):
    cattle_docs = db.collection("cattle").where("owner", "==", owner).stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in cattle_docs]

@app.delete("/cattle/{cattle_id}")
async def delete_cattle(cattle_id: str):
    cattle_ref = db.collection("cattle").document(cattle_id)
    if not cattle_ref.get().exists:
        raise HTTPException(status_code=404, detail="Cattle not found")
    cattle_ref.delete()
    return {"message": "Cattle deleted successfully"}

class CattleUpdate(BaseModel):
    health: str | None = None
    status: str | None = None

@app.put("/cattle/{cattle_id}")
async def update_cattle(cattle_id: str, cattle_data: CattleUpdate):
    try:
        cattle_ref = db.collection("cattle").document(cattle_id)
        
        # Check if the cattle document exists
        if not cattle_ref.get().exists:
            raise HTTPException(status_code=404, detail="Cattle not found")

        # Prepare the update data
        update_data = cattle_data.dict(exclude_unset=True)

        # Ensure there is at least one field to update
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")

        # Perform the update
        cattle_ref.update(update_data)
        return {"message": "Cattle updated successfully"}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {e}\nTraceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


# Appoinement s 
class Appointment(BaseModel):
    title: str
    date: str  # Stored in "YYYY-MM-DD" format
    time: str  # Stored as a string, e.g., "14:30"
    message: str | None = None
    username: str  # Dummy user field
    accepted: bool = False  # Default value

# Function to sort by latest date
def sort_appointments_by_date(appointments):
    return sorted(appointments, key=lambda x: x["date"], reverse=True)

# ✅ Endpoint to Create an Appointment
@app.post("/appointments")
async def create_appointment(appointment: Appointment):
    try:
        appointment_data = appointment.dict()
        doc_ref = db.collection("appointments").document()
        doc_ref.set(appointment_data)

        return {"message": "Appointment created successfully", "appointment_id": doc_ref.id}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {e}\nTraceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ✅ Endpoint to Get All Appointments (Sorted by Date)
@app.get("/appointments", response_model=List[Appointment])
async def get_all_appointments():
    try:
        appointments_ref = db.collection("appointments").stream()
        appointments = [doc.to_dict() for doc in appointments_ref]

        if not appointments:
            raise HTTPException(status_code=404, detail="No appointments found")

        return sort_appointments_by_date(appointments)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {e}\nTraceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ✅ Endpoint to Get Appointments for a Specific User (Sorted by Date)
@app.get("/appointments/user/{username}", response_model=List[Appointment])
async def get_appointments_by_user(username: str):
    try:
        appointments_ref = db.collection("appointments").where("username", "==", username).stream()
        user_appointments = [doc.to_dict() for doc in appointments_ref]

        if not user_appointments:
            raise HTTPException(status_code=404, detail="No appointments found for this user")

        return sort_appointments_by_date(user_appointments)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {e}\nTraceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Predict Pests and Diseases
@app.post("/predict-pest")
async def predict_pest(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image for pest attack detection.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the image
        image = Image.open(file_path).convert("RGB")  # Ensure RGB format
        image = image.resize((48, 48))  # Resize to model's input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the pest attack label
        predictions = MODEL_PESTS.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))  # Highest probability

        # Map the predicted index to the pest attack label
        predicted_label = CLASS_PESTS[predicted_index]

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        os.remove(file_path)  # Clean up the uploaded file in case of error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Predict Cow Health
class HealthStatusInput(BaseModel):
    body_temperature: float
    milk_production: float
    respiratory_rate: int
    walking_capacity: int
    sleeping_duration: float
    body_condition_score: int
    heart_rate: int
    eating_duration: float
    lying_down_duration: float
    ruminating: float
    rumen_fill: int

@app.post("/predict-health-status")
async def predict_health_status(input_data: HealthStatusInput):
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Predict using the loaded model
        predicted_class = MODEL_HEALTH.predict(input_df)[0]

        # Decode the predicted class
        health_status = LABEL_ENCODER.inverse_transform([predicted_class])[0]

        return {"health_status": health_status}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    


# Milk Quality Monitor
class MilkQualityInput(BaseModel):
    pH: float
    Temperature: float
    Taste: int
    Odor: int
    Fat: float
    Turbidity: int
    Colour: int

# Define grade mapping
grade_mapping = {0: "high", 1: "low", 2: "medium"}

@app.post("/predict-milk-grade")
async def predict_milk_grade(input_data: MilkQualityInput):

    try:
        # Prepare the input for prediction
        input_array = np.array([
            [
                input_data.pH,
                input_data.Temperature,
                input_data.Taste,
                input_data.Odor,
                input_data.Fat,
                input_data.Turbidity,
                input_data.Colour,
            ]
        ])

        # Perform the prediction
        predicted_grade = MODEL_ML_QUALITY.predict(input_array)

        # Map the predicted grade to a category
        predicted_grade_category = grade_mapping.get(predicted_grade[0])

        if predicted_grade_category is None:
            raise ValueError("Invalid prediction received from the model.")

        return {"predicted_grade": predicted_grade_category}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Milk Production Forecast
class MilkProductionRequest(BaseModel):
    year: int
    month: int

@app.post("/predict-milk-production")
async def predict_milk_production(request: MilkProductionRequest):
    """
    Predicts milk production for a given year and month.

    Input:
    - year: int, the year for which to predict milk production.
    - month: int, the month for which to predict milk production.

    Returns:
    - predicted_value: float, predicted milk production value for the given month.
    """
    try:
        # Extract year and month from the request
        year = request.year
        month = request.month

        # Validate month input
        if month < 1 or month > 12:
            raise HTTPException(status_code=400, detail="Invalid month. Must be between 1 and 12.")

        # Convert input year and month into a datetime object
        prediction_date = pd.Timestamp(year=year, month=month, day=1)

        # Get the last available date in the model data (assumed to be the last training date)
        last_date = MODEL_ML_FORECAST.data.dates[-1]

        # Calculate the number of months between the last training date and the requested prediction date
        months_ahead = (prediction_date.year - last_date.year) * 12 + (prediction_date.month - last_date.month)

        if months_ahead < 0:
            raise HTTPException(
                status_code=400,
                detail="The requested date is within the training data range. Please provide a future date for prediction."
            )

        # Forecast the milk production for the given number of months ahead
        forecast = MODEL_ML_FORECAST.forecast(steps=months_ahead)
        print(year)
        print(month)

        # Return the predicted value for the requested month
        predicted_value = forecast[-1]
        print(forecast)
        return {"predicted_milk_production": predicted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Milk Production x 2
MODEL_MILK_DAILY = joblib.load("fine_tuned_random_forest.joblib")

# Dictionary of encoded rainfall labels
rainfall_encoding = {
    'Clear': 0,
    'Overcast': 1,
    'Partially cloudy': 2,
    'Rain': 3,
    'Rain, Overcast': 4,
    'Rain, Partially cloudy': 5
}

class MilkPredictionRequest(BaseModel):
    year: float
    month: float
    date: float
    temperature: float
    humidity: float
    rainfall: str

@app.post("/milk-weather-predict")
def predict_milk_liters(request: MilkPredictionRequest):
    print(request)
    """
    Endpoint to predict milk liters based on input features.
    """
    encoded_rainfall = rainfall_encoding.get(request.rainfall, 6)  # Default to 6 if unknown
    input_features = np.array([[request.year, request.month, request.date, request.temperature, request.humidity, encoded_rainfall]])

    predicted_value = MODEL_MILK_DAILY.predict(input_features)[0]
    
    return {"predicted_milk_liters": predicted_value}

class MilkRecord(BaseModel):
    cattle_id: str
    amount: float = 0.0
    status: str = "ok"

class DailyMilkSummary(BaseModel):
    predicted_total: Optional[float] = 0.0
    actual_total: Optional[float] = 0.0

@app.post("/milk-summary/{username}/{date_str}")
async def create_or_update_milk_summary(username: str, date_str: str, summary: DailyMilkSummary):
    date_doc_ref = db.collection("users").document(username).collection("daily-milk-summary").document(date_str)
    date_doc_ref.set(summary.dict(), merge=True)
    return {"message": "Daily milk summary updated", "summary": summary.dict()}


@app.get("/milk-summary/{username}/{date_str}")
async def get_milk_summary(username: str, date_str: str):
    date_doc_ref = db.collection("users").document(username).collection("daily-milk-summary").document(date_str)
    doc = date_doc_ref.get()

    if doc.exists:
        summary = doc.to_dict()
        return {"message": "Daily milk summary retrieved", "summary": summary}
    else:
        return {"message": "No milk summary found for the given date"}

@app.post("/milk-record/{username}/{date_str}")
async def add_milk_record(username: str, date_str: str, record: MilkRecord):
    milk_records_ref = db.collection("users").document(username).collection("daily-milk-summary").document(date_str).collection("milk-records")
    record_ref = milk_records_ref.document()
    record_ref.set(record.dict())
    return {"message": "Milk record added", "record": record.dict()}

@app.delete("/milk-record/{username}/{date_str}/{record_id}")
async def delete_milk_record(username: str, date_str: str, record_id: str):
    record_ref = db.collection("users").document(username).collection("daily-milk-summary").document(date_str).collection("milk-records").document(record_id)
    if not record_ref.get().exists:
        raise HTTPException(status_code=404, detail="Milk record not found")
    record_ref.delete()
    return {"message": "Milk record deleted successfully"}


# Locate Vets   
class Location(BaseModel):
    latitude: float
    longitude: float

class Review(BaseModel):
    author_name: str
    rating: int
    text: str
    location_name: str
    polarity: Optional[float] = None


outlets_db = {}
outlet_ref = db.collection("outlets")

def analyze_sentiment(text: str) -> float:
    """Calculate polarity score using TextBlob"""
    return TextBlob(text).sentiment.polarity if text else 0.0

@app.post("/submit_review")
async def submit_review(review: Review):
    try:
        # Calculate polarity score
        review.polarity = analyze_sentiment(review.text)

        # Save review to Firestore
        review_data = review.dict()  
        db.collection("reviews").add(review_data)

        return {"message": "Review submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class PlaceDetails(BaseModel):
    name: str
    address: str
    location: Location
    rating: Optional[float] = None
    reviews: List[Review] = []
    phone_number: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None





def get_nearby_locations(latitude: float, longitude: float, radius: int = 5000):
    location = f"{latitude},{longitude}"
    params = {
        "location": location,
        "radius": radius,
        "type": "veterinary_care",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(GOOGLE_PLACES_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        locations = []

        for result in data.get('results', []):
            place_id = result.get("place_id")
            details = get_place_details(place_id) if place_id else {}
            firestore_reviews = []

            reviews_ref = db.collection("reviews").where("location_name", "==", result.get("name")).stream()

            firestore_reviews = [review.to_dict() for review in reviews_ref]
            pol = 0
            if len(firestore_reviews) > 0:
                total_polarity = sum(review.get("polarity", 0) for review in firestore_reviews)
                pol = total_polarity

            # Merge Google reviews and Firestore reviews
            all_reviews = details.get("reviews", []) + firestore_reviews

            locations.append({
                "name": result.get("name"),
                "address": result.get("vicinity"),
                "location": result.get("geometry", {}).get("location"),
                "rating": result.get("rating"),
                "reviews": all_reviews,
                "phone_number": details.get("formatted_phone_number"),
                "email": details.get("email"),
                "website": details.get("website"),
                "polarity_score": pol
            })

        # Sort locations by polarity_score, highest to lowest (undefined comes last)
        # locations.sort(key=lambda x: (x['polarity_score'] is None, -x['polarity_score'] if x['polarity_score'] is not None else float('inf')))
        
        return locations
    else:
        return {"error": "Failed to retrieve data from Google Places API"}

def get_place_details(place_id):
    """Fetch detailed information about a place using Google Place Details API."""
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,reviews,email,website",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(GOOGLE_DETAILS_URL, params=params)
    
    if response.status_code == 200:
        data = response.json().get("result", {})
        
        # Add polarity score to Google reviews
        if "reviews" in data:
            for review in data["reviews"]:
                review["polarity"] = analyze_sentiment(review["text"])

        return data
    return {}

@app.post("/nearby_locations")
async def nearby_locations(location: Location):
    locations = get_nearby_locations(location.latitude, location.longitude)
    return {"locations": locations}


@app.post("/outlets")
async def create_outlet(outlet: PlaceDetails):
    """Create a new outlet and store it in Firestore."""
    outlet_dict = outlet.dict()
    doc_ref = outlet_ref.add(outlet_dict)  # Add the document to Firestore
    return {"message": "Outlet created successfully", "id": doc_ref[1].id}

@app.get("/outlets/{outlet_id}")
async def get_outlet(outlet_id: str):
    """Retrieve an outlet by its Firestore document ID."""
    doc = outlet_ref.document(outlet_id).get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Outlet not found")
    
    return doc.to_dict()




