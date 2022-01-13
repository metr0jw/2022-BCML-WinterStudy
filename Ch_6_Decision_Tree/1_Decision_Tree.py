# 2022 code by @metr0jw
# https://metr0jw.studio

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'


# Load data
iris = load_iris()
data = iris.data[:, 2:]
target = iris.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=33)


# Model build
criterion = "gini"
"""
Gini impurity is usually used, but in case of our data probability distribution is exponential or Laplace
entropy outperform gini.
"""
decision_tree = DecisionTreeClassifier(criterion=criterion, max_depth=3)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("decision_tree", decision_tree)
])

# Train
pipe.fit(x_train, y_train)
export_graphviz(
    decision_tree,
    out_file="./img/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# Prediction
print("Predict probality of petal which have 5cm long and 1.5 cm wide: " + str(decision_tree.predict_proba([[5, 1.5]])))
y_pred = pipe.predict(x_test)
print(f"Expected target: {y_pred}")
print(f"Actual target:   {y_test}")
print(f"accuracy: {accuracy_score(y_test, y_pred)}")


# Visualization
plt.figure(figsize=(8, 4))
axes = [0, 7.5, 0, 3]

plt.plot(data[:, 0][target == 0], data[:, 1][target == 0], "yo", label="Iris-Setosa")
plt.plot(data[:, 0][target == 1], data[:, 1][target == 1], "bs", label="Iris-Versicolor")
plt.plot(data[:, 0][target == 2], data[:, 1][target == 2], "g^", label="Iris-Virginica")
plt.axis(axes)

plt.xlabel("꽃잎 길이", fontsize=14)
plt.ylabel("꽃잎 너비", fontsize=14)

plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "깊이=0", fontsize=15)
plt.text(3.2, 1.80, "깊이=1", fontsize=13)
plt.text(4.05, 0.5, "(깊이=2)", fontsize=11)

plt.savefig('./img/1_decision_tree.png')
plt.show()
