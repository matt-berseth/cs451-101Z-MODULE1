# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate a random dataset
# Create data with a simple linear relationship for demonstration purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and test sets (30% of data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the regressor
regressor = LinearRegression().fit(X_train, y_train)

# TODO: reminder Matt
# replace LogisticRegression with DecisionTreeClassifier, or RandomForestClassifier

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# If you'd like to make predictions:
# new_data = [[x1], ...]
# predictions = regressor.predict(new_data)
