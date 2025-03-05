import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Datafiniti_Womens_Shoes.csv')

# Compute average price and keep relevant columns
df['price'] = (df['prices.amountMax'] + df['prices.amountMin']) / 2
df = df[['brand', 'categories', 'colors', 'price']]

# Drop missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['brand', 'categories', 'colors']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df[['brand', 'categories', 'colors']]
y = df['price']

# Split dataset into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create directories if they don't exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save datasets
train_path = "datasets/train.csv"
val_path = "datasets/validation.csv"
test_path = "datasets/test.csv"

pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
pd.concat([X_val, y_val], axis=1).to_csv(val_path, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

print("✅ Datasets saved successfully in 'datasets/'.")

# Train model using train and validation sets
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f'📊 Validation RMSE: {rmse:.2f}')
print(f'📈 Validation R² Score: {r2:.3f}')

# Save trained model
model_path = "models/random_forest_model.pkl"
joblib.dump(model, model_path)

print(f"✅ Model saved to {model_path}")
