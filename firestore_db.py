import os
from google.cloud import firestore

# Set the environment variable for the service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cattleproject-fbb10-firebase-adminsdk-kvejw-dabce09a49.json"

# Initialize Firestore DB
db = firestore.Client()

def get_firestore_client():
    return db