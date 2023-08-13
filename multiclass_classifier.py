# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a random dataset
# We'll use a utility function from scikit-learn to create a multi-class dataset.
X, y = make_classification(n_samples=300, n_features=5, n_informative=3, 
                           n_redundant=1, n_classes=3, random_state=42)

# Split the data into training and test sets (30% of data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the classifier
# Note: Logistic Regression, when used with the 'ovr' (one-vs-rest) or 
# 'multinomial' option for multi_class, can handle multi-class classification.
clf = LogisticRegression(random_state=42, multi_class="ovr").fit(X_train, y_train)

# TODO: reminder Matt
# replace LogisticRegression with DecisionTreeClassifier, or RandomForestClassifier

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# If you'd like to make predictions:
# new_data = [[x1, x2, ...], ...]
# predictions = clf.predict(new_data)
