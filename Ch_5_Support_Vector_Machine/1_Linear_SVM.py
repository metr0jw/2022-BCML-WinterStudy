# 2022 code by @metr0jw
# https://metr0jw.studio

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# Load datasets
iris = datasets.load_iris()
data = iris["data"][:, (2, 3)]                      # petal length, petal width
target = (iris["target"] == 2).astype(np.float64)   # Iris-Virginica(or not)

# Compare C==1 and C==100
for i, C in enumerate([1, 100]):
    # Build a pipeline
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=C, loss="hinge"))
    ])
    """
    Option 1. SVC(kernel="linear", C=1)
    PROS: ?
    CONS: Much slower than LinearSVC, especially with large dataset.
    
    Option 2. SGDClassifier(loss="hinge", alpha=1/(m*C)). (== training a linear SVC)
    PROS: Good for handling huge datasets, or to handle online classification
    CONS: Slower than LinearSVC.
    """

    # Fitting
    svm_clf.fit(data, target)

    # Prediction
    print(svm_clf.predict([[5.5, 1.7]]))

    # Visualize sample data
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)

    # Visualize Hyper-Plane
    decision_function = svm_clf.decision_function(data)

    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = data[support_vector_indices]

    plt.subplot(2, 1, i+1)
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")
    plt.title("C=" + str(C))
plt.tight_layout()
plt.show()
