# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/model_2024-09-21-14-54-01.pkl')
model = joblib.load(model_path)

# Define the input data model
class SalesPredictionInput(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: bool
    Open: int
   

# Define the prediction endpoint
@app.post("/predict_sales")
async def predict_sales(input_data: SalesPredictionInput):
    try:
        # Convert input data into a DataFrame for the model
        input_df = pd.DataFrame([input_data.dict()])
        
        # Generate predictions
        prediction = model.predict(input_df)
        
        # Return prediction result
        return {"predicted_sales": prediction.tolist()[0]}  # Return the first prediction
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
