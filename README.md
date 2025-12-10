### 🚚 Food Delivery Time Prediction using Random Forest

This project predicts food delivery time (in minutes) based on multiple real-world factors such as delivery person ratings, weather, distance, vehicle type, and order conditions.
A Random Forest Regressor is trained using a labeled CSV dataset, and the final predictions help food delivery companies optimize ETA accuracy and improve customer satisfaction.

## 📘 **1. Project Summary**

This repository contains a complete **machine learning pipeline** designed to predict **food delivery time (in minutes)** using real-world operational features such as distance, traffic, weather, pickup times, and delivery rider performance.

The solution is built using a **Random Forest Regression model**, making it stable, highly interpretable, and effective for nonlinear ETA prediction tasks commonly used by food-delivery companies like Swiggy, Zomato, Uber Eats, and Bolt Food.

This project showcases:

* End-to-end ML development
* Feature engineering & preprocessing
* Model training & evaluation
* Saved production model (`joblib`)
* API-ready prediction system

---

## 📂 **2. Repository Structure**

```
Random-forest-food-delivery-main/
│
├── food_prediction.py        # Full ML training + prediction pipeline
├── random_forest.joblib      # Trained Random Forest model
├── food.csv                  # Dataset (cleaned, labeled)
└── README.md                 # Documentation (this file)
```

---

## 🧠 **3. Machine Learning Architecture**

### 🔧 **3.1 Complete Pipeline Flow**

```
           ┌──────────────────────────┐
           │     Load Dataset         │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │  Clean Missing Values    │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │ Encode Categorical Data  │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │     Train-Test Split     │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │ Random Forest Regressor  │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │ Save Model (joblib)      │
           └───────────────┬──────────┘
                           ▼
           ┌──────────────────────────┐
           │ Predict Delivery Time    │
           └──────────────────────────┘
```

---

## 🔍 **4. Dataset & Features**

The dataset includes essential operational features used by ETA prediction systems:

| Feature                   | Description                               |
| ------------------------- | ----------------------------------------- |
| `distance`                | Distance between restaurant & customer    |
| `Weather_conditions`      | Sunny / Foggy / Stormy                    |
| `Road_traffic_density`    | Low / Medium / High / Jam                 |
| `Delivery_person_Ratings` | Rider reliability rating                  |
| `Type_of_vehicle`         | Bike / Scooter / Bicycle                  |
| `Festival`                | Indicates holiday/high load               |
| `multiple_deliveries`     | Whether rider is handling multiple orders |
| `Time_Orderd`             | Order placement time                      |
| `Time_Order_picked`       | Time when rider picked the order          |
| `Delivery_time`           | **Target variable** — minutes taken       |

This variety of categorical & numeric data makes Random Forest an excellent fit.

---

## 🤖 **5. Model Explanation**

### 🟢 **Why Random Forest?**

* Handles **nonlinear patterns**
* Resilient to noisy & messy data
* Works well with **mixed feature types**
* Provides stable predictions
* Avoids overfitting via ensembling

### ⚙️ Model Training Includes:

* One-hot encoding
* Feature alignment
* Train-test split
* Random Forest training
* Saving the model for production use

---

## 🚀 **6. Running the Project**

### Install dependencies

```bash
pip install pandas numpy joblib scikit-learn fastapi uvicorn
```

### Train the model

```bash
python food_prediction.py
```

### Predict inside the script

Modify the `input_data = {...}` dictionary, then run:

```bash
python food_prediction.py
```

---

# 🌐 **7. REST API for Deployment (FastAPI)**

A lightweight API for real-time ETA predictions.

### Create `api.py`:

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("random_forest.joblib")

@app.get("/")
def root():
    return {"message": "Food Delivery Time Prediction API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"predicted_delivery_time_minutes": prediction}
```

### Start server:

```bash
uvicorn api:app --reload
```

### API Endpoint:

```
http://127.0.0.1:8000/docs
```

Swagger UI will appear for easy testing.

---

# 🧭 **8. Prediction Flow Overview**

```
User Input
   ↓
Preprocessing Pipeline
   ↓
Random Forest Model
   ↓
Predicted Delivery Time (minutes)
```

---

# 🔮 **9. Future Enhancements**

* Streamlit dashboard UI
* Feature importance visualizations
* Hyperparameter tuning (GridSearchCV)
* Integration with real GPS distance (Haversine)
* Model deployment using Docker
* CI/CD automation for model retraining

---

