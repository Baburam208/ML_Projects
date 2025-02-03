from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel


# Define the input data schema using Pydantic
# ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
#        'Mortgage', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard',
#        'Gender_M', 'Gender_O', 'Gender_Unknown', 'Home_Ownership_Home_Owner',
#        'Home_Ownership_Rent']

class LoanApplication(BaseModel):
    Age: int
    Experience: int
    Income: float
    Family: int
    CCAvg: float
    Mortgage: int
    Online: int
    Gender: str
    Home_Ownership: str


# Initialize FastAPI app
app = FastAPI()

# # Load the trained model
# model = joblib.load('loan_approval_model.pkl')

# Load the trained pipeline
pipeline = joblib.load('loan_approval_pipeline.pkl')


@app.get('/')
def root():
    return {
        'description': 'This is a loan approval machine learning project'
    }


@app.post("/predict")
async def predict(loan_application: LoanApplication):
    # try:
    #     # Convert input data to a DataFrame
    #     input_data = pd.DataFrame([loan_application.model_dump()])

    #     # Preprocess the input date
    #     input_data['Gender'] = input_data['Gender'].replace(['#', '-'], 'Unknown')
    #     input_data = pd.get_dummies(input_data, columns=['Gender', 'Home_Ownership'], drop_first=True)

    #     # Ensure the input data has the same columns as the training data
    #     # Add missing columns with default values (if necessary)
    #     expected_columns = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
    #                         'Mortgage', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard',
    #                         'Gender_M', 'Gender_O', 'Gender_Unknown', 'Home_Ownership_Home_Owner',
    #                         'Home_Ownership_Rent']
        
    #     for col in expected_columns:
    #         if col not in input_data.columns:
    #             input_data[col] = 0

    #     # Reorder columns to match the training data
    #     input_data = input_data[expected_columns]

    #     # Make predictions
    #     prediction = model.predict(input_data)
    #     return {"prediction": int(prediction[0])}

    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))

    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([loan_application.model_dump()])

        # The pipeline will handle preprocessing
        prediction = pipeline.predict(input_data)
        
        # Map the prediction to the corresponding message
        prediction_message = "Loan approved" if prediction[0] == 1 else "Loan not approved"
        
        return {"prediction": prediction_message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

# Run the FastAPI app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("model_serving:app", host='127.0.0.0', port=8000, reload=True)
