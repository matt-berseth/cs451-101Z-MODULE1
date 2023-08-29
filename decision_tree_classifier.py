import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# load the data
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# train a classifier ...
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)

# try some other variants with other hyperparameter values
#tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
#tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
#tree_clf = DecisionTreeClassifier(max_depth=15, random_state=42)

# this is what builds the tree (i.e. identifies the optimal set of rules that
# splits the data)
tree_clf.fit(X_iris, y_iris)

# compute the accuracy
yhat = tree_clf.predict(X_iris)
n_correct = np.sum(yhat==y_iris)
print(f"{n_correct} out of {len(yhat)}. {round(n_correct/len(yhat), 2)}%")

# show us the output label names. The index
# of these labels in the array maps to the label.
# i.e.
# 0 = setosa
# 1 = versicolor
# 2 = virginica
print(iris.target_names)

# some basic error analysis
incorrect_idx = np.argwhere(yhat!=y_iris).flatten()
print(f"Incorrect indices: {incorrect_idx}")

print("X data for incorrect examples:")
print(X_iris[incorrect_idx])

print("Y data for incorrect examples:")
print(y_iris[incorrect_idx])

print("yhat data for incorrect examples:")
print(yhat[incorrect_idx])


# to view dot file: copy `iris_tree.dot` into
# https://dreampuf.github.io/GraphvizOnline/
export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)