import logging
import os
import warnings

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
np.random.seed(0)

# Load MNIST dataset from OpenML
logging.info("Checking if mnist data exists ...")
if not os.path.exists(".data"):
    logging.info("mnist data does not exist, downloading and saving locally ...")
    mnist = fetch_openml("mnist_784")

    # for now, only interested in 1000 images
    n = 10000
    os.makedirs(".data", exist_ok=True)
    np.savetxt(os.path.join(".data", "y.txt"), mnist.target[:n], fmt="%s")
    np.savetxt(os.path.join(".data", "X.txt"), mnist.data[:n], fmt="%1.2f")

logging.info("loading mnist data ...")
X = np.loadtxt(os.path.join(".data", "X.txt"))
y = np.loadtxt(os.path.join(".data", "y.txt"))
logging.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

logging.info("Print out the count of each label")
for i in range(0, 10):
    count = np.sum(y==i)
    logging.info(f"{i}: {count} / {round(count/len(y)*100, 4)}%")

# Preprocess the data
# Scale pixel values to mean 0 and variance 1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create the MLP model 784x5x10
mlp = MLPClassifier(hidden_layer_sizes=(5), learning_rate_init=.001, random_state=0)

# Create the MLP model 784x20x10, more complexity
# mlp = MLPClassifier(hidden_layer_sizes=(20), learning_rate_init=.001, random_state=0)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# evaluate: more hidden nodes, more iterations
n_epochs = 10
minibatch_sz = 20
for i in range(n_epochs):
    for ii in range(X_train.shape[0] // minibatch_sz):
        # grab random elements from the minibatch ...
        inds = list(range(0, X_train.shape[0]))
        np.random.shuffle(inds)
        inds = inds[:minibatch_sz]

        # Fit the model
        mlp.partial_fit(X_train[inds], y_train[inds], np.unique(y))

        # Make predictions (train)
        yhat = mlp.predict(X_train)
        train_accuracy = round(np.sum(yhat==y_train) / len(y_train), 2)

        # Make predictions (test)
        yhat = mlp.predict(X_test)
        test_accuracy = round(np.sum(yhat==y_test) / len(y_test), 2)
        print(f"epoch: {i}, mini-batch {ii} Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
