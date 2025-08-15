import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import datetime

# ------------------- Load dataset -------------------
df = pd.read_csv("../data/eiadatasetAI.csv", encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip().str.replace('\xa0','').str.replace("'", "")

# ------------------- Target Column -------------------
target_candidates = [col for col in df.columns if "carbon footprint" in col.lower()]
if len(target_candidates) == 0:
    raise ValueError("Target column not found in CSV")
target = target_candidates[0]
print("Target column:", target)

# ------------------- Missing Values -------------------
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# ------------------- Feature Engineering -------------------
current_year = datetime.datetime.now().year

# Years since reporting
if 'Year of reporting' in df.columns:
    df['Years_Since_Reporting'] = current_year - df['Year of reporting']

# Log product weight
if 'Product weight (kg)' in df.columns:
    df['Log_Product_Weight'] = np.log1p(df['Product weight (kg)'])

# PCF change category
if 'Relative change in PCF vs previous' in df.columns:
    df['Relative change in PCF vs previous'] = pd.to_numeric(df['Relative change in PCF vs previous'], errors='coerce').fillna(0)
    def categorize_change(x):
        if x > 0.05: return 'Increase'
        elif x < -0.05: return 'Decrease'
        else: return 'Stable'
    df['PCF_Change_Category'] = df['Relative change in PCF vs previous'].apply(categorize_change)
    le_change = LabelEncoder()
    df['PCF_Change_Category_Encoded'] = le_change.fit_transform(df['PCF_Change_Category'])
else:
    df['PCF_Change_Category_Encoded'] = 0

# Fraction columns
fraction_cols = [
    '*Upstream CO2e (fraction of total PCF)',
    '*Operations CO2e (fraction of total PCF)',
    '*Downstream CO2e (fraction of total PCF)',
    '*Transport CO2e (fraction of total PCF)',
    '*EndOfLife CO2e (fraction of total PCF)'
]
existing_fraction_cols = []
for col in fraction_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(str(x).strip('%'))/100 if pd.notnull(x) else 0)
        existing_fraction_cols.append(col)
if existing_fraction_cols:
    df['Total_CO2e_Fraction'] = df[existing_fraction_cols].sum(axis=1)
else:
    df['Total_CO2e_Fraction'] = 0

# Word counts
df['Product_Detail_Word_Count'] = df['Product detail'].apply(lambda x: len(str(x).split())) if 'Product detail' in df.columns else 0
df['Change_Reason_Word_Count'] = df['Company-reported reason for change'].apply(lambda x: len(str(x).split())) if 'Company-reported reason for change' in df.columns else 0

# ------------------- Feature Selection -------------------
selected_features = [
    'Year of reporting',
    'Log_Product_Weight',
    'PCF_Change_Category_Encoded',
    'Total_CO2e_Fraction',
    'Product_Detail_Word_Count',
    'Change_Reason_Word_Count'
]

# Keep only available features
selected_features = [f for f in selected_features if f in df.columns]

X = df[selected_features]
y = df[target]

# ------------------- Train-Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- Train Gradient Boosting -------------------
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# ------------------- Evaluate -------------------
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\nModel Evaluation:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# ------------------- Save Model -------------------
with open("eia_model.pkl", "wb") as f:
    pickle.dump(gb_model, f)

print("\nModel saved as eia_model.pkl ✅")
