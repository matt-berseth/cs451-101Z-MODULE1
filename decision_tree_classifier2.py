import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

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

# convert to true/false. True pixels are black pixels
X = X < (255//2)

logging.info("Print out the count of each label")
for i in range(0, 10):
    count = np.sum(y==i)
    logging.info(f"{i}: {count} / {round(count/len(y)*100, 4)}%")

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# train a classifier ...
# classify the digits based on testing only 3 pixels ...
# tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# classify the digits based on testing only 7 pixels ...
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)

# this is what builds the tree (i.e. identifies the optimal set of rules that
# splits the data)
tree_clf.fit(X_train, y_train)

# compute the accuracy
yhat = tree_clf.predict(X_train)
n_correct = np.sum(yhat==y_train)
print(f"TRAIN: {n_correct} out of {len(yhat)}. {round(n_correct/len(yhat), 2)}%")

yhat = tree_clf.predict(X_test)
n_correct = np.sum(yhat==y_test)
print(f"TEST: {n_correct} out of {len(yhat)}. {round(n_correct/len(yhat), 2)}%")

# to view dot file: copy `iris_tree.dot` into
# https://dreampuf.github.io/GraphvizOnline/
export_graphviz(
    tree_clf,
    out_file="digits.dot",
    proportion=True,
    rounded=True,
    filled=True
)

# below works when depth is 7

# 0 is black, white is 255
x = np.full((28*28, 3), 255, dtype=np.uint8)
# predict 9 with 85% probability if all of these pixels are black  
x[658] = (0,255,0)
x[400] = (0,255,0)
x[541] = (0,255,0)
x[319] = (0,255,0)
x[489] = (0,255,0)
x[378] = (0,255,0)

x = np.array(x)
x = x.reshape(28, 28, 3) # reshape into a 28x28 matrix
plt.imshow(x, interpolation="nearest")
plt.axis("off")
plt.savefig(f"./nine.png")

x = np.full((28*28, 3), 255, dtype=np.uint8)
# predict 7 with 85% probability if all of these pixels are white
x[153] = (255,0,0)
x[485] = (255,0,0)
x[433] = (255,0,0)
x[459] = (255,0,0)
x[542] = (255,0,0)
x[596] = (255,0,0)
x[378] = (255,0,0)

x = np.array(x)
x = x.reshape(28, 28, 3) # reshape into a 28x28 matrix
plt.imshow(x, interpolation="nearest")
plt.axis("off")
plt.savefig(f"./seven.png")