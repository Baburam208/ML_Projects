import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import scipy.stats as stats

# Load dataset
file_path = 'ADBL.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: file '{file_path}' not found.")
    exit()

# Selecting features
selected_features = ['Open', 'High', 'Low', 'Close', 'Vol']
df = df[selected_features]

X = df[['Open', 'High', 'Low', 'Vol']]
y = df['Close']

# Check for missing values
if df.isnull().any().any():
    print("Warning: The dataset contains missing values. Handle them before proceeding.")
    exit()

# Split the data without shuffling
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Shuffle the training set but keep test set unchanged
train_indices = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(train_indices)

X_train = X_train.iloc[train_indices]
y_train = y_train.iloc[train_indices]

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)  # Use the same scaler fitted on training data


# Reshape y to 2D array for scaling, and `StandardScaler`` transforms `y_train` and `y_test` into a 2D array (shape (n_samples, 1))
# `SVR` expects the target variable (y) to be a 1D array (shape (n_samples,)),
# (and most other scikit-learn models) expects the target variable y to be a 1D array.
# `.ravel()` method flattens the 2D array into a 1D array, which is required by the SVR model.
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Train SVR model
# svr = SVR(kernel='rbf', C=0.1, gamma=0.01)
# svr = SVR(kernel='linear', C=0.1)
# C: regularization parameters
# gamma: Kernel Coefficient for RBF

models = {
    'linear': LinearRegression(),
    'svr': SVR(kernel='linear', C=0.1),
    'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
    'randomforest_regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

model_name = 'linear'
models[model_name].fit(X_train_scaled, y_train_scaled)

# Make predictions
predictions_scaled = models[model_name].predict(X_test_scaled)

# Inverse transform the predictions to original scale
y_pred_actual = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))


# Computing the metrics for the regression task.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_actual)
r2 = r2_score(y_test, y_pred_actual)
mape = np.mean(np.abs((y_test - y_pred_actual.ravel()) / y_test)) * 100  # Convert to percentage

# Print metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(y_test.values, label='Actual Close Price')
plt.plot(y_pred_actual, label='Predicted Close Price')
plt.xlabel("Timestamps")
plt.ylabel("Close Price")
plt.title("ADBL Actual vs Predicted Close Price")
plt.legend()
plt.show()

## Comparing the train and test metrics
# Predictions for training data
train_predictions_scaled = models[model_name].predict(X_train_scaled)
train_predictions_actual = scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1)).ravel()

# Compute training metrics
mse_train = mean_squared_error(y_train, train_predictions_actual)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predictions_actual)
r2_train = r2_score(y_train, train_predictions_actual)

# Compute test metrics
y_pred_actual = y_pred_actual.ravel()
mse_test = mean_squared_error(y_test, y_pred_actual)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_actual)
r2_test = r2_score(y_test, y_pred_actual)

# Print results
print(f"Training Set Metrics:")
print(f"  MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}")
print(f"Test Set Metrics:")
print(f"  MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}")


# Calculate residuals
residuals = y_test.values - y_pred_actual.ravel()  # Ensuring both are 1D

# 1 Residual Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test.values, residuals, alpha=0.6, edgecolors='k')
plt.axhline(0, color='red', linestyle='dashed')
plt.xlabel("Actual Close Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# 2️ Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='dashed')
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# 3️ QQ Plot
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()
