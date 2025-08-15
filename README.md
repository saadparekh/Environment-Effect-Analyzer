# Environment Impact Analyzer (EIA) - Full Stack Project

This is a full-stack web application that predicts the carbon footprint of products based on various input parameters. The application uses a machine learning model trained on environmental data to make predictions.

## Project Structure

```
EIA_FullStack_Project/
│
├── app.py                     # Flask main backend
├── model/
│   ├── model_train.py         # Model training + feature processing
│   └── eia_model.pkl          # Saved trained model (pickle)
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html             # Frontend main page
├── data/
│   └── eiadatasetAI.csv       # Dataset
└── requirements.txt           # Python dependencies
```

## How to Run the Project

### 1. Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### 2. Setup Instructions

#### Step 1: Create a Virtual Environment (Recommended)

```bash
# Navigate to the project directory
cd EIA_FullStack_Project

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

#### Step 3: Prepare the Dataset

1. Obtain the `eiadatasetAI.csv` file with the required data
2. Place it in the `data/` directory
3. Ensure the CSV file has the following columns:
   - `Product weight (kg)`
   - `*Carbon intensity`
   - `*Operations CO2e (fraction of total PCF)`
   - `Year of reporting`
   - `Product's carbon footprint (PCF, kg CO2e)` (target variable)

#### Step 4: Train the Model (Optional)

If you need to retrain the model or if `eia_model.pkl` doesn't exist:

```bash
# Navigate to the model directory
cd model

# Run the model training script
python model_train.py

# This will create/update the eia_model.pkl file
```

#### Step 5: Run the Flask Application

```bash
# Navigate back to the main project directory if you're in the model directory
cd ..

# Run the Flask app
python app.py
```

The application will start running on `http://127.0.0.1:5000/` by default.

### 3. Using the Application

1. Open your web browser and go to `http://127.0.0.1:5000/`
2. Enter the required information in the form:
   - Product weight (kg)
   - *Carbon intensity
   - *Operations CO2e (fraction of total PCF)
   - Year of reporting
3. Click "Predict PCF" to get the carbon footprint prediction

## Features

- Responsive web interface
- Real-time carbon footprint prediction
- Clean and professional UI
- Error handling for invalid inputs

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Scikit-learn (Gradient Boosting Regressor)
- **Data Processing**: Pandas, NumPy

## Model Information

The machine learning model uses a Gradient Boosting Regressor to predict the carbon footprint based on the input features. The model is evaluated using R² Score and RMSE metrics.