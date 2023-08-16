# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Generate a random dataset
# Here, we're making up data just for demonstration. Replace this with your data.
np.random.seed(0)
X = np.zeros((200, 2))
X[:100, 0] = np.random.randint(-13, -6, (100,)) + np.random.random((100,))
X[:100, 1] = np.random.randint(-7, 0, (100,)) + np.random.random((100,))
X[100:, 0] = np.random.randint(-5, 3, (100,)) + np.random.random((100,))
X[100:, 1] = np.random.randint(0, 9, (100,)) + np.random.random((100,))
y = 100*[0] + 100*[1]

# drop a plot of the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow")
plt.title("Binary Classifier")
plt.xlabel("X0")
plt.ylabel("X1")
plt.savefig("./binary_classifier.png")

# Split the data into training and test sets (30% of data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the classifier
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# If you'd like to make predictions:
new_data = [[0, 1]]
print(clf.predict(new_data))
print(clf.predict_proba(new_data))

new_data = [[0.4, -0.5]]
print(clf.predict(new_data))
print(clf.predict_proba(new_data))