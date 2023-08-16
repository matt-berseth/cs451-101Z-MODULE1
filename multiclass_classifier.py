# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
X = np.zeros((300, 2))
X[:100, 0] = np.random.randint(-13, -6, (100,)) + np.random.random((100,))
X[:100, 1] = np.random.randint(-7, 0, (100,)) + np.random.random((100,))
X[100:200, 0] = np.random.randint(-10, -4, (100,)) + np.random.random((100,))
X[100:200, 1] = np.random.randint(-10, -5, (100,)) + np.random.random((100,))
X[200:, 0] = np.random.randint(-5, 2, (100,)) + np.random.random((100,))
X[200:, 1] = np.random.randint(2.5, 7.5, (100,)) + np.random.random((100,))
y = 100*[0] + 100*[1] + 100*[2]

# drop a plot of the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow")
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("Multiclass Classifier")
plt.savefig("./multiclass_classifier.png")

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
