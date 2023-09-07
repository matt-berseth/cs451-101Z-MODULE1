from sklearn.model_selection import train_test_split
import numpy as np

# Sample data
X = np.random.rand(1000, 2) # 1000 rows, 2 columns/features
y = [0] * 1000 # default all lables to '0'
# set every 10th label to '1'
# severe class imbalance (9 '0' labels for every '1' label)
for i in range(100):
   y[i] = 1

# Split the data (don't stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Output the results
print(f"y_train has {np.sum(y_train)} '1's")
print(f"y_test has {np.sum(y_test)} '1's")


# Split the data (do stratify)
# Question: if we stratify, how many examples do we expect from each class/label?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Output the results
print(f"y_train has {np.sum(y_train)} '1's")
print(f"y_test has {np.sum(y_test)} '1's")