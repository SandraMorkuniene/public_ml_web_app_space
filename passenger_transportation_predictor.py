
import streamlit as st
import pandas as pd
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

# Access AWS credentials from Streamlit 
aws_access_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)


# Define the S3 bucket and model file key
bucket_name = 'my.spacemodel.s3.bucket'
model_file_key = 'best_model.joblib'

# Function to download the model from S3
def download_model_from_s3():
    try:
        # Download the model file from S3 to local storage
        s3.download_file(bucket_name, model_file_key, 'best_model.joblib')
        # Load the model
        model = joblib.load('best_model.joblib')
        return model
    except NoCredentialsError:
        st.error("AWS credentials not available. Please set your credentials.")
        return None
    except Exception as e:
        st.error(f"Error downloading the model from S3: {e}")
        return None

# Load the model
model = download_model_from_s3()

# Streamlit app title and description
st.title("Passenger Transportation Predictor")

st.write("""
Upload a CSV file containing passenger details to predict whether each passenger was transported or not.
The file should include the required columns:
- PassengerId
- HomePlanet
- CryoSleep (0 = No, 1 = Yes)
- Cabin
- Destination
- Age
- VIP (0 = No, 1 = Yes)
- RoomService, FoodCourt, ShoppingMall, Spa, VRDeck (Expenditure columns)
""")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())  # Display a preview of the data

        # Validate the required columns
        required_columns = [
            "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination",
            "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
        ]
        if not all(column in data.columns for column in required_columns):
            st.error("The uploaded file is missing required columns.")
        else:
            # Process and predict
            if st.button("Predict"):
                try:
                    
                    processed_data = data.copy()  

                    # Make predictions
                    predictions = model.predict(processed_data)
                    prediction_probs = model.predict_proba(processed_data)[:, 1]

                    
                    data["Transported"] = predictions
                    data["Transported_Probability"] = prediction_probs

                    st.write("Predictions:")
                    st.dataframe(data[["PassengerID", "Transported", "Transported_Probability"]])

                    
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
