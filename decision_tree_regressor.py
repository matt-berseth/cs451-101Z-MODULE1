from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

# load the data
calf = fetch_california_housing(as_frame=True)

X = calf.data.values
y = calf.target

# train a classifier ...
tree = DecisionTreeRegressor(max_depth=2, random_state=42)
tree.fit(X, y)

# TODO: experiment with the parameters. 
# can we overfit and go really deep?
# min samples leaf?

# to view dot file: copy `iris_tree.dot` into
# https://dreampuf.github.io/GraphvizOnline/
export_graphviz(
    tree,
    out_file="calf.dot",
    feature_names=calf.data.columns.values.tolist(),
    rounded=True,
    filled=True
)