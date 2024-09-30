import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
file_path = 'healthcare_dataset.csv/healthcare_dataset.csv'
data = pd.read_csv(file_path)

# Convert date columns to datetime format
data['Date of Admission'] = pd.to_datetime(data['Date of Admission'], format='%Y-%m-%d')
data['Discharge Date'] = pd.to_datetime(data['Discharge Date'], format='%Y-%m-%d')

# Calculate the stay period in days
data['Stay Period'] = (data['Discharge Date'] - data['Date of Admission']).dt.days

# Handle missing values
data.ffill(inplace=True)

# Drop unnecessary columns
data = data.drop(['Name', 'Room Number', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date'], axis=1)

# Split the data into features (X) and target (y)
X = data.drop('Billing Amount', axis=1)
y = data['Billing Amount']

# Define the categorical and numerical columns
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']
numerical_cols = ['Age', 'Stay Period']

# Label encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  # Save the encoder for future use

# Scale numerical columns
scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save the encoders and scaler to disk
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Convert to NumPy arrays
X = X.values
y = y.values

# Ensure target values are non-negative
y_min = np.min(y)
if y_min <= 0:
    y += abs(y_min) + 1  # Shift all values in y to be positive

# Split the data into training and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply log transformation to y_train and y_valid
y_train = np.log1p(y_train)  # log1p avoids log(0) issues
y_valid = np.log1p(y_valid)

# Reshape y_train and y_valid to be 2D arrays for TabNet
y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)

# Initialize TabNetRegressor with increased complexity
model = TabNetRegressor(
    n_d=64,  # Feature dimensionality
    n_a=64,  # Attention dimensionality
    n_steps=8,  # More decision steps
    gamma=1.5,
    n_independent=4,  # Increased number of independent layers
    n_shared=4,  # Increased number of shared layers
    optimizer_fn=torch.optim.LBFGS,  # Using second-order L-BFGS optimizer
    optimizer_params=dict(lr=0.01),  # Learning rate for L-BFGS
    mask_type='entmax',
    device_name='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize TabNetRegressor with Adam optimizer
model = TabNetRegressor(
    n_d=64,  # Feature dimensionality
    n_a=64,  # Attention dimensionality
    n_steps=8,  # More decision steps
    gamma=1.5,
    n_independent=4,  # Increased number of independent layers
    n_shared=4,  # Increased number of shared layers
    optimizer_fn=torch.optim.Adam,  # Using first-order Adam optimizer
    optimizer_params=dict(lr=0.001),  # Learning rate for Adam
    mask_type='entmax',
    device_name='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train the model using the built-in fit method
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Make predictions and reverse log transformation
preds = model.predict(X_valid)
preds = np.expm1(preds)  # Reverse log1p to get the original scale

# Custom RMSE function with dynamic tolerance
def custom_rmse_dynamic_tolerance(y_true, y_pred):
    """
    Custom RMSE with dynamic tolerance based on different cost brackets.
    
    Returns:
    - RMSE with different tolerance levels based on cost.
    """
    tolerance = np.where(y_true <= 10000, np.abs(y_true) * 0.05,  # 5% for low costs
                         np.where(y_true <= 50000, np.abs(y_true) * 0.10,  # 10% for mid-range costs
                                  np.abs(y_true) * 0.15))  # 15% for high costs
    
    errors = y_pred - y_true
    
    # Apply tolerance windows
    errors_within_tolerance = np.where(
        np.abs(errors) <= tolerance,
        0,  # No penalty for errors within tolerance
        errors
    )
    
    return np.sqrt(mean_squared_error(y_true, y_true + errors_within_tolerance))

# Evaluate the model with the custom RMSE based on dynamic tolerance
custom_rmse = custom_rmse_dynamic_tolerance(np.expm1(y_valid), preds)
print(f'Custom RMSE with dynamic tolerance: {custom_rmse:.4f}')


# Save the TabNet model
joblib.dump(model, 'tabnet_healthcare_model.pkl')
