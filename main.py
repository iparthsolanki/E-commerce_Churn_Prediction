from contextlib import asynccontextmanager
from pathlib import Path
import logging

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, field_validator


# 1. LOGGING -> “Logging is used to track application events, errors, and important activities for monitoring and debugging.”

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("churn_api")


# 2. CONFIGURATION ->“Configuration is used to centrally manage important application settings such as file paths and prediction thresholds.”


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "XGBoost_Pipeline.pkl"

CHURN_THRESHOLD = 0.50

API_PREFIX = "/api/v1"


# 3. MODEL LOADING ->“Model loading is used to load the trained machine learning model into memory when the FastAPI application starts, so it can be reused for predictions.”


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    try:
        logger.info("Loading XGBoost model...")

        model = joblib.load(MODEL_PATH)

        logger.info("XGBoost model loaded successfully.")

    except Exception as exc:

        logger.exception("Failed to load model.")

        raise RuntimeError(
            f"Could not load model from {MODEL_PATH}"
        ) from exc

    yield

    logger.info("Application shutting down.")



# 4. FASTAPI APPLICATION->“FastAPI creates the application instance and provides metadata such as the API title, description, version, and lifecycle management.”

app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description=(
        "Production-style REST API for predicting e-commerce "
        "customer churn using an XGBoost machine learning pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 5. CORS->“CORS middleware controls which frontend origins are allowed to communicate with the FastAPI backend.”


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",

        "https://e-commerce-churn-prediction-1.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 6. PYDANTIC REQUEST SCHEMA
# “Pydantic models are used to define the structure and validation rules for incoming API requests.”

class CustomerData(BaseModel):
    """
    Input schema for a single e-commerce customer.
    """

    membership_category: str = Field(
        ...,
        description="Customer membership category"
    )

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer age"
    )

    gender: str = Field(
        ...,
        min_length=1,
        description="Customer gender"
    )

    region_category: str = Field(
        ...,
        min_length=1,
        description="Customer region category"
    )

    joined_through_referral: str = Field(
        ...,
        min_length=1,
        description="Whether customer joined through referral"
    )

    preferred_offer_types: str = Field(
        ...,
        min_length=1,
        description="Preferred offer type"
    )

    medium_of_operation: str = Field(
        ...,
        min_length=1,
        description="Primary operating medium"
    )

    internet_option: str = Field(
        ...,
        min_length=1,
        description="Internet option"
    )

    avg_time_spent: float = Field(
        ...,
        ge=0,
        description="Average time spent"
    )

    avg_transaction_value: float = Field(
        ...,
        ge=0,
        description="Average transaction value"
    )

    points_in_wallet: float = Field(
        ...,
        ge=0,
        description="Points available in wallet"
    )

    used_special_discount: str = Field(
        ...,
        min_length=1,
        description="Whether special discount was used"
    )

    offer_application_preference: str = Field(
        ...,
        min_length=1,
        description="Offer application preference"
    )

    past_complaint: str = Field(
        ...,
        min_length=1,
        description="Whether customer had a past complaint"
    )

    complaint_status: str = Field(
        ...,
        min_length=1,
        description="Customer complaint status"
    )

    feedback: str = Field(
        ...,
        min_length=1,
        description="Customer feedback"
    )

    days_since_last_login: float = Field(
        ...,
        ge=0,
        description="Number of days since last login"
    )

    avg_frequency_login_days: float = Field(
        ...,
        ge=0,
        description="Average login frequency in days"
    )

    customer_tenure_days: float = Field(
        ...,
        ge=0,
        description="Customer tenure in days"
    )

    visit_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of customer's visit"
    )

    # Pydantic validation -> “This validator checks that categorical values are strings, removes extra spaces, and prevents empty values from entering the API.”


    @field_validator(
        "gender",
        "region_category",
        "membership_category",
        "joined_through_referral",
        "preferred_offer_types",
        "medium_of_operation",
        "internet_option",
        "used_special_discount",
        "offer_application_preference",
        "past_complaint",
        "complaint_status",
        "feedback",
        mode="before"
    )
    @classmethod
    def strip_string_values(cls, value):

        if not isinstance(value, str):
            raise ValueError("Value must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("Value cannot be empty.")

        return value



# 7. PYDANTIC RESPONSE SCHEMA -> “It defines and validates the structure and format of the data returned by the API.”


class PredictionResponse(BaseModel):

    predicted_churn: int = Field(
        ...,
        description="1 = Churn, 0 = No Churn"
    )

    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of customer churn"
    )

    risk_status: str = Field(
        ...,
        description="Customer churn risk status"
    )

    threshold: float = Field(
        ...,
        description="Classification threshold used"
    )



# 8. HEALTH CHECK -> “It checks whether the FastAPI application and the machine learning model are running and available.”


@app.get(
    "/health",
    tags=["Health"]
)
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": "XGBoost",
        "threshold": CHURN_THRESHOLD
    }

# 9. API ROOT --> “It provides a welcome message and basic information about the API, including its version and documentation link.”

@app.get(
    "/",
    tags=["System"]
)
def root():

    return {
        "message": "E-Commerce Customer Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

# 10. PREDICTION ENDPOINT --> “It receives customer data, processes it, and returns the predicted churn probability and risk status.”

@app.post(
    f"{API_PREFIX}/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict_churn(customer: CustomerData):

    if model is None:

        logger.error("Prediction requested but model is not loaded.")

        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not available."
        )

    try:

        # Convert Pydantic model → Dictionary

        customer_dict = customer.model_dump()
        
        # Dictionary → DataFrame

        input_df = pd.DataFrame([customer_dict])

        logger.info("Prediction request received.")
        
        # Get churn probability

        churn_probability = model.predict_proba(
            input_df
        )[0][1]

        churn_probability = float(churn_probability)

        # Apply final business threshold

        predicted_churn = int(
            churn_probability >= CHURN_THRESHOLD
        )
        
        # Risk classification

        if predicted_churn == 1:

            risk_status = "High Churn Risk"

        else:

            risk_status = "Low Churn Risk"

        logger.info(
            "Prediction completed | churn=%s | probability=%.4f",
            predicted_churn,
            churn_probability
        )
        
        # API Response

        return PredictionResponse(
            predicted_churn=predicted_churn,
            churn_probability=round(
                churn_probability,
                4
            ),
            risk_status=risk_status,
            threshold=CHURN_THRESHOLD
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the input data and model."
        ) from exc



# 11. GLOBAL EXCEPTION HANDLER --> “It catches unhandled exceptions, logs them, and returns a standardized error response to the client.”


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(
        "Unhandled exception: %s",
        exc
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred."
        }
    )

        