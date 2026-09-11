print("Loading dataset for Student ID: Saba_8005")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Student ID
STUDENT_ID = "Saba_8005"

# 1. Load dataset
data = pd.read_csv("/content/A1-MLOps_8005/data/dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

# 2. Separate features and target
X = data.drop("price", axis=1)
y = data["price"]

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

### changes start

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### changes end

# 4. Create and train the machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed successfully.")

# 5. Save the trained model
model_path = "model/house_price_model_8005.pkl"

joblib.dump(model, model_path)

print("Model saved successfully.")
print("Model location:", model_path)